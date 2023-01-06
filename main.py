#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import random
import torch
import numpy as np
from uuid import uuid4
from typing import Dict

import hydra
from omegaconf import DictConfig, OmegaConf # @manual //github/third-party/omry/omegaconf:omegaconf

from iopath.common.file_io import PathManager, NativePathHandler
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning.profiler import (
    SimpleProfiler,
    AdvancedProfiler,
    PyTorchProfiler,
    PassThroughProfiler,
)
from pytorch_lightning.callbacks import TQDMProgressBar


from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

from torch.distributed.launcher import (
    LaunchConfig,
    elastic_launch as launch
)

from nlf import (
    INRTrainer,
    INRDataModule,
    INRSystem
)


class INRModelCheckpoint(ModelCheckpoint):
    """Like pytorch_lightning.callbacks.ModelCheckpoint but allowing saving last top k checkpoints.
    See https://github.com/PyTorchLightning/pytorch-lightning/discussions/10669
    """
    def _save_last_checkpoint(self, trainer: "pl.Trainer", monitor_candidates: Dict[str, torch.Tensor]) -> None:
        if self._every_n_epochs >= 1 and (trainer.current_epoch + 1) % self._every_n_epochs == 0:
            super()._save_last_checkpoint(trainer, monitor_candidates)


def run(cfg: DictConfig, log_dir: str, ckpt_dir: str, workflow_id: str) -> None:
    # Print
    print(OmegaConf.to_yaml(cfg))
    OmegaConf.set_struct(cfg, False)
    cfg = cfg.experiment

    # Seed
    if 'seed' in cfg.params \
        and not isinstance(cfg.params.seed, str) \
        and cfg.params.seed is not None:

        seed_everything(cfg.params.seed, workers=True)

    # CWD paths
    dir_path = os.path.normpath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../../'))
    os.chdir(dir_path)

    # PathManager
    pmgr = PathManager()
    pmgr.register_handler(NativePathHandler())

    # Logging and saving
    if log_dir is None or log_dir == "":
        log_dir = os.path.expanduser(cfg.params.log_dir)

    log_dir = os.path.join(log_dir, cfg.params.name)

    pmgr.mkdirs(log_dir)

    if cfg.params.save_results:
        cfg.params.save_video_dir = os.path.join(
            log_dir,
            cfg.params.save_video_dir
        )
        cfg.params.save_image_dir = os.path.join(
            log_dir,
            cfg.params.save_image_dir
        )
        pmgr.mkdirs(cfg.params.save_video_dir)
        pmgr.mkdirs(cfg.params.save_image_dir)

    logger = TensorBoardLogger(save_dir=log_dir, name=cfg.params.name)

    # Setup system and datamodule
    dm = INRDataModule(cfg)
    dm.prepare_data()

    if 'sample_with_replacement' in cfg.training and cfg.training.sample_with_replacement:
        cfg.training.iters_per_epoch = cfg.training.num_iters
    else:
        cfg.training.iters_per_epoch = int(np.ceil(dm.original_dataset_size / cfg.training.batch_size))

    # Checkpointing
    if ckpt_dir is None or ckpt_dir == "":
        ckpt_dir = os.path.expanduser(cfg.params.ckpt_dir)

    ckpt_dir = os.path.join(ckpt_dir, cfg.params.name)

    if 'ckpt_name' in cfg.params and cfg.params.ckpt_name != '' and cfg.params.ckpt_name is not None:
        ckpt_name = cfg.params.ckpt_name
    elif cfg.params.load_from_weights:
        ckpt_name = 'last-weights'
    else:
        ckpt_name = 'last'

    if cfg.params.load_from_weights:
        last_ckpt_path = f'{ckpt_dir}/{ckpt_name}.ckpt'
    else:
        last_ckpt_path = f'{ckpt_dir}/{ckpt_name}.ckpt'

    if not pmgr.exists(last_ckpt_path):
        last_ckpt_path = None

    checkpoint_callback = INRModelCheckpoint(
        dirpath=ckpt_dir,
        filename='{epoch:d}',
        monitor='val/loss',
        mode='min',
        save_top_k=-1,
        save_last=True,
        every_n_epochs=cfg.training.ckpt_every
    )
    checkpoint_callback.CHECKPOINT_NAME_LAST = 'last'

    weights_checkpoint_callback = INRModelCheckpoint(
        save_weights_only=True,
        dirpath=ckpt_dir,
        filename='{epoch:d}-weights',
        monitor='val/loss',
        mode='min',
        save_top_k=-1,
        save_last=True,
        every_n_epochs=cfg.training.ckpt_every
    )
    weights_checkpoint_callback.CHECKPOINT_NAME_LAST = 'last-weights'

    # Other callbacks
    callbacks = []
    callbacks.append(TQDMProgressBar(refresh_rate=10))

    # Load checkpoint
    if last_ckpt_path is not None and cfg.params.load_from_weights:
        system = INRSystem.load_from_checkpoint(last_ckpt_path, cfg=cfg, dm=dm)
    else:
        system = INRSystem(cfg, dm=dm)

    # Trainer
    if cfg.params.render_only:
        cfg.training.render_every = 1
        cfg.training.val_every = 1

    if cfg.params.test_only:
        cfg.training.test_every = 1
        cfg.training.val_every = 1

    trainer = INRTrainer(
        cfg,
        callbacks=[checkpoint_callback, weights_checkpoint_callback] + callbacks,
        resume_from_checkpoint=last_ckpt_path if not cfg.params.load_from_weights else None,
        logger=logger if cfg.params.tensorboard else False,
        accelerator='gpu',
        strategy='ddp' if cfg.training.num_gpus > 1 else None,
        check_val_every_n_epoch=cfg.training.val_every,
        benchmark=False,
        profiler=None,
        #profiler=AdvancedProfiler(dirpath='/home/benattal/logs/profiler', filename='logs.txt'),
        #profiler=PyTorchProfiler(dirpath='/home/benattal/logs/pytorch_profiler', filename='logs.txt', row_limit=-1),
    )

    # Fit
    trainer.fit(system, datamodule=dm)


def elastic_run(cfg: DictConfig):
    if cfg.experiment.training.num_gpus > 1:
        lc = LaunchConfig(
            # Assuming devgpu testing, min = max nodes = 1
            min_nodes=1,
            max_nodes=1,
            nproc_per_node=cfg.experiment.training.num_gpus,
            rdzv_backend="zeus",
            # run_id just has to be globally unique
            run_id=f"your_run_identifier_{uuid4()}",
            # for fault tolerance; for testing set it to 0 (no fault tolerance)
            max_restarts=0,
            start_method="spawn",
        )
        # The "run" function is called inside the elastic_launch
        ret = launch(lc, run)(cfg, "", "", "")
        print(f"Rank 0 results = {ret[0]}")
    else:
        run(cfg, "", "", "")


@hydra.main(config_path='conf', config_name='config')
def main(cfg: DictConfig):
    elastic_run(cfg)


if __name__ == '__main__':
    main()
