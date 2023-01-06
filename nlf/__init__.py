#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import time
import copy
from collections import namedtuple

from torch.utils.data import DataLoader, RandomSampler

import numpy as np
import torch

import imageio # noqa
from PIL import Image

from iopath.common.file_io import PathManager, NativePathHandler
from omegaconf import ListConfig

from pytorch_lightning import (
    LightningModule,
    LightningDataModule,
    Trainer
)

from datasets import dataset_dict
from datasets.base import Base5DDataset, Base6DDataset
from .subdivision import subdivision_dict
from .rendering import (
    render_chunked,
    render_fn_dict
)

from losses import loss_dict
from metrics import ( # noqa
    psnr_gpu,
    ssim_gpu,
    psnr,
    ssim,
    get_mean_outputs,
)

from utils.config_utils import replace_config, lambda_config
from utils.tensorf_utils import AlphaGridMask
from utils import (
    to8b,
    get_optimizer,
    get_scheduler,
    weight_init_dict
)

from .models import model_dict
from .regularizers import regularizer_dict
from .visualizers import visualizer_dict

from utils.gui_utils import NeRFGUI


class INRTrainer(Trainer):
    def __init__(
        self,
        cfg,
        **kwargs,
        ):
        super().__init__(
            gpus=cfg.training.num_gpus,
            max_epochs=cfg.training.num_epochs if 'num_epochs' in cfg.training else None,
            max_steps=-1,
            log_every_n_steps=cfg.training.flush_logs,
            **kwargs
        )

    def save_checkpoint(self, *args, **kwargs):
        if not self.is_global_zero:
            return

        super().save_checkpoint(*args, **kwargs)


class INRDataModule(LightningDataModule):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        self.current_epoch = 0

        # Replacement
        self.sample_with_replacement = getattr(self.cfg.training, 'sample_with_replacement', False)
        self.num_iters = getattr(self.cfg.training, 'num_iters', -1)

        # Testing
        self.test_every = cfg.training.test_every
        self.is_testing = False
        self.test_only = getattr(self.cfg.params, 'test_only', False)

        # Multiscale
        self.multiscale_training = getattr(self.cfg.training, 'multiscale', False)
        self.scale_epochs = getattr(self.cfg.training, 'scale_epochs', [])
        self.scales = getattr(self.cfg.training, 'scales', [])
        self.scale_batch_sizes = getattr(self.cfg.training, 'scale_batch_sizes', [])

        # TODO:
        #   - NOW: Same scale factor for all
        #   - LATER: Allow for completely different configs for multi-scale training
        self.scale_lrs = getattr(self.cfg.training, 'scale_lrs', [])

        self.cur_idx = -1
        self.cur_scale = 1.0
        self.cur_batch_size = self.cfg.training.batch_size
        self.cur_lr = 1.0
        self.prepared = False

    def get_cur_scale(self, epoch):
        if not self.multiscale_training:
            return {
                'idx': self.cur_idx,
                'scale': self.cur_scale,
                'batch_size': self.cur_batch_size,
                'lr': self.lr,
            }

        cur_idx = 0
        cur_scale = 1.0
        cur_batch_size = self.cfg.training.batch_size
        cur_lr =  1.0

        for idx in range(len(self.scales)):
            if epoch >= self.scale_epochs[idx]:
                cur_idx = idx
                cur_scale = self.scales[idx]
                cur_lr = self.scale_lrs[idx]

        return {
            'idx': cur_idx,
            'scale': cur_scale,
            'batch_size': cur_batch_size,
            'lr': cur_lr,
        }

    def prepare_data(self):
        if self.prepared:
            return

        self.prepared = True
        self.reload_data()
    
    def reload_data(self):
        ## Train, val, test datasets
        dataset_cl = dataset_dict[self.cfg.dataset.train.name] \
            if 'train' in self.cfg.dataset else dataset_dict[self.cfg.dataset.name]
        self.train_dataset = dataset_cl(self.cfg, split='train')
        dataset_cl = dataset_dict[self.cfg.dataset.val.name] \
            if 'val' in self.cfg.dataset else dataset_dict[self.cfg.dataset.name]
        self.val_dataset = dataset_cl(self.cfg, split='val')
        dataset_cl = dataset_dict[self.cfg.dataset.test.name] \
            if 'test' in self.cfg.dataset else dataset_dict[self.cfg.dataset.name]
        self.test_dataset = dataset_cl(self.cfg, split='test')
        dataset_cl = dataset_dict[self.cfg.dataset.render.name] \
            if 'render' in self.cfg.dataset else dataset_dict[self.cfg.dataset.name]
        self.render_dataset = dataset_cl(self.cfg, split='render')

        ## Stats
        self.original_dataset_size = len(self.train_dataset)

        ## Regularizer datasets
        self.create_regularizer_datasets()
        self.update_data()

    def setup(self, stage):
        pass

    def create_regularizer_datasets(self):
        self.regularizer_datasets = {}

        for key in self.cfg.regularizers.keys():
            cfg = self.cfg.regularizers[key]

            if cfg is not None and 'dataset' in cfg:
                dataset_cl = dataset_dict[cfg.dataset.name]
                self.regularizer_datasets[cfg.type] = dataset_cl(
                    cfg, train_dataset=self.train_dataset
                )

    def update_data(self):
        # Set iter
        self.train_dataset.cur_iter = self.current_epoch

        # Resize
        reset_dataloaders = False

        if self.multiscale_training:
            scale_params = self.get_cur_scale(self.current_epoch)

            if scale_params['idx'] != self.cur_idx:
                print(f"Scaling dataset to scale {scale_params['scale']} batch_size: {scale_params['batch_size']}")

                self.cur_idx = scale_params['idx']
                self.cur_scale = scale_params['scale']
                self.cur_batch_size = scale_params['batch_size']
                self.cur_lr = scale_params['lr']

                self.train_dataset.scale(self.cur_scale)
                self.val_dataset.scale(self.cur_scale)
                self.create_regularizer_datasets()
                reset_dataloaders = True

        # Crop
        self.train_dataset.crop()

        # Shuffle
        if self.train_dataset.use_full_image:
            self.train_dataset.shuffle()

        for dataset in self.regularizer_datasets.values():
            dataset.shuffle()

        return reset_dataloaders

    def train_dataloader(self):
        if self.sample_with_replacement:
            sampler = RandomSampler(
                self.train_dataset,
                replacement=True,
                num_samples=self.num_iters * self.cur_batch_size
            )

            return DataLoader(
                self.train_dataset,
                num_workers=self.cfg.training.num_workers,
                persistent_workers=True,
                sampler=sampler,
                batch_size=self.cur_batch_size,
                pin_memory=True
            )
        else:
            return DataLoader(
                self.train_dataset,
                shuffle=(not self.train_dataset.use_full_image),
                num_workers=self.cfg.training.num_workers,
                persistent_workers=True,
                batch_size=self.cur_batch_size,
                pin_memory=True
            )

    def val_dataloader(self):
        if ((self.current_epoch + 1) % self.test_every == 0) or self.test_only:
            print("Testing")
            dataset = self.test_dataset
            self.is_testing = True

            if hasattr(self.test_dataset, 'video_paths'):
                return DataLoader(
                    dataset,
                    shuffle=False,
                    num_workers=0,
                    persistent_workers=False,
                    batch_size=1,
                    pin_memory=True
                )
        else:
            print("Validating")
            self.is_testing = False
            dataset = self.val_dataset

        return DataLoader(
            dataset,
            shuffle=False,
            num_workers=self.cfg.training.num_workers,
            persistent_workers=True,
            batch_size=1,
            pin_memory=True
        )


class INRSystem(LightningModule):
    def __init__(self, cfg, dm):
        super().__init__()

        self.cfg = cfg
        self.dm = dm

        ## Settings ##

        # Path manager
        self.pmgr = PathManager()
        self.pmgr.register_handler(NativePathHandler())

        # Training and loss
        self.automatic_optimization = False
        self.training_started = False
        self.loss = loss_dict[self.cfg.training.loss.type](self.cfg.training.loss)

        # Data loading
        self.sample_with_replacement = getattr(self.cfg.training, 'sample_with_replacement', False)
        self.num_iters = getattr(self.cfg.training, 'num_iters', -1)

        # Test & render
        self.render_only = getattr(self.cfg.params, 'render_only', False)
        self.test_only = getattr(self.cfg.params, 'test_only', False)
        self.interact_only = getattr(self.cfg.params, 'interact_only', False)

        # Convert epochs -> iterations in config
        def set_iter(cfg, key):
            if isinstance(cfg[key], ListConfig):
                cfg[key.replace('epoch', 'iter')] = \
                    [[lii * self.cfg.training.iters_per_epoch for lii in li] for li in cfg[key]]
            else:
                cfg[key.replace('epoch', 'iter')] = cfg[key] * self.cfg.training.iters_per_epoch

        for key in ['max_freq', 'wait', 'stop', 'falloff', 'window', 'no_bias', 'window_bias', 'window_bias_start', 'decay', 'warmup']:
            lambda_config(self.cfg, f'{key}_epoch', set_iter)
            lambda_config(self.cfg, f'{key}_epochs', set_iter)

        ## Set-up rendering pipeline ##

        # Create subdivision (sampling) scheme
        self.is_subdivided = ('subdivision' in cfg.model) \
            and (cfg.model.subdivision.type is not None)

        if self.is_subdivided:
            self.subdivision = subdivision_dict[
                self.cfg.model.subdivision.type
            ](
                self,
                self.cfg.model.subdivision,
            )

            replace_config(
                self.cfg,
                voxel_size=float(self.subdivision.voxel_size.cpu())
            )

            if 'min_point' in self.subdivision.__dict__:
                replace_config(
                    self.cfg,
                    min_point=self.subdivision.min_point
                )

            if 'max_point' in self.subdivision.__dict__:
                replace_config(
                    self.cfg,
                    max_point=self.subdivision.max_point
                )
        else:
            self.subdivision = None

        # Model mapping samples -> color
        model = model_dict[self.cfg.model.type](
            self.cfg.model, system=self
        )

        # Render function that queries model using subdivision scheme
        self.rendering = False
        self.render_fn = render_fn_dict[
            self.cfg.model.render.type
        ](
            model,
            self.subdivision,
            cfg.model.render,
            net_chunk=self.cfg.training.net_chunk,
        )

        ## Optimizers ##

        self.optimizer_configs = {}

        for idx, key in enumerate(self.cfg.training.optimizers.keys()):
            opt_cfg = copy.deepcopy(self.cfg.training.optimizers[key])
            self.optimizer_configs[key] = opt_cfg

        self.optimizer_groups = {}

        for module in self.render_fn.modules():
            if 'opt_group' in module.__dict__:
                if isinstance(module.opt_group, str):
                    if module.opt_group in self.optimizer_groups:
                        self.optimizer_groups[module.opt_group] += [module]
                    else:
                        self.optimizer_groups[module.opt_group] = [module]
                else:
                    for k, v in module.opt_group.items():
                        if k in self.optimizer_groups:
                            self.optimizer_groups[k] += copy.copy(v)
                        else:
                            self.optimizer_groups[k] = copy.copy(v)

        self.reset_opt_list = getattr(self.cfg.training, 'reset_opt_list', [])
        self.skip_opt_list = []

        ## Additional objects used for training & visualization ##

        # Regularizers for additional losses during training
        self.regularizers = []
        self.regularizer_configs = []

        for key in self.cfg.regularizers.keys():
            cfg = self.cfg.regularizers[key]
            reg = regularizer_dict[cfg.type](
                self,
                cfg
            )

            self.regularizer_configs.append(cfg)
            self.regularizers.append(reg)
            setattr(self, f"reg_{cfg.type}", reg)

        # Number of regulariztion pretraining iterations
        self.num_regularizer_pretraining_iters = getattr(self.cfg.training, 'num_regularizer_pretraining_iters', 0)

        # Visualizers
        self.visualizers = []

        for key in self.cfg.visualizers.keys():
            cfg = self.cfg.visualizers[key]
            vis = visualizer_dict[cfg.type](
                self,
                cfg
            )

            self.visualizers.append(vis)

        ## Network weight initialization ##

        self.apply(
            weight_init_dict[self.cfg.training.weight_init.type](
                self.cfg.training.weight_init
            )
        )

    def load_state_dict(self, state_dict, strict=False):
        new_state_dict = {}

        # For loading subdivision variables (voxel grid, voxel size, etc.) #
        alpha_aabb = None
        alpha_volume = None

        for key in state_dict.keys():
            new_state_dict[key] = state_dict[key]
            
            # Update size of tensor components
            if 'alpha_aabb' in key:
                alpha_aabb = state_dict[key]
            elif 'alpha_volume' in key:
                alpha_volume = state_dict[key]
            elif 'gridSize' in key:
                self.render_fn.model.color_model.net.gridSize = state_dict[key]

                self.render_fn.model.color_model.net.init_svd_volume(
                    self.render_fn.model.color_model.net.gridSize[0],
                    self.render_fn.model.color_model.net.device
                )

        for key in state_dict.keys():
            if 'app_plane' in key or 'density_plane' in key or \
                'app_line' in key or 'density_line' in key:

                new_shape = self.state_dict()[key].shape

                if state_dict[key].shape != new_shape:
                    new_state_dict[key] = state_dict[key].view(*new_shape)

        super().load_state_dict(new_state_dict, strict=False)

        # Update other grid-size-dependent variables
        self.render_fn.model.color_model.net.update_stepSize(
            self.render_fn.model.color_model.net.gridSize
        )

        # Update alpha mask
        if alpha_volume is not None:
            device = self.render_fn.model.color_model.net.device
            self.render_fn.model.color_model.net.alphaMask = AlphaGridMask(
                device,
                alpha_aabb.to(device),
                alpha_volume.to(device)
            )

    def render(self, method_name, coords, **render_kwargs):
        return self.run_chunked(
            coords, getattr(self.render_fn, method_name), **render_kwargs
        )

    def forward(self, coords, **render_kwargs):
        return self.run_chunked(
            coords, self.render_fn, **render_kwargs
        )

    def run_chunked(self, coords, fn, **render_kwargs):
        if self.rendering:
            ray_chunk = self.cfg.training.render_ray_chunk if 'render_ray_chunk' in self.cfg.training else self.cfg.training.ray_chunk
        else:
            ray_chunk = self.cfg.training.ray_chunk

        return render_chunked(
            coords,
            fn,
            render_kwargs,
            chunk=ray_chunk
        )

    def configure_optimizers(self):
        print("Configuring optimizers")

        optimizers = []
        schedulers = []

        # Iterate over groups
        for idx, key in enumerate(self.optimizer_groups.keys()):
            opt_cfg = copy.deepcopy(self.optimizer_configs[key])
            opt_cfg.lr *= self.trainer.datamodule.cur_lr

            # Optimizer
            optimizer = get_optimizer(opt_cfg, self.optimizer_groups[key])
            optimizers.append(optimizer)

            # Scheduler
            scheduler = get_scheduler(
                opt_cfg,
                optimizer,
                self.cfg.training.iters_per_epoch
            )
            schedulers.append(scheduler)

        return optimizers, schedulers

    def needs_opt_reset(self, train_iter):
        # Check if reset needed
        needs_reset = False

        for idx, key in enumerate(self.optimizer_groups.keys()):
            opt_cfg = self.optimizer_configs[key]

            if 'reset_opt_list' in opt_cfg and (train_iter in opt_cfg.reset_opt_list):
                needs_reset = True

        return needs_reset or (train_iter == 0)

    def reset_optimizers(self, train_iter):
        # Perform opt reset
        optimizers = []
        schedulers = []

        # Iterate over groups
        for idx, key in enumerate(self.optimizer_groups.keys()):
            opt_cfg = copy.deepcopy(self.optimizer_configs[key])

            if 'skip_opt_list' in opt_cfg and (train_iter in opt_cfg.skip_opt_list):
                self.skip_opt_list.append(key)
            elif 'remove_skip_opt_list' in opt_cfg and (train_iter in opt_cfg.remove_skip_opt_list):
                self.skip_opt_list = [skip_key for skip_key in self.skip_opt_list if skip_key != key]

            if 'reset_opt_list' in opt_cfg and (train_iter in opt_cfg.reset_opt_list):
                print("Resetting optimizer", opt_cfg)
                opt_cfg.lr *= self.trainer.datamodule.cur_lr

                # Optimizer
                optimizer = get_optimizer(opt_cfg, self.optimizer_groups[key])
                optimizers.append(optimizer)

                # Scheduler
                scheduler = get_scheduler(
                    opt_cfg,
                    optimizer,
                    self.cfg.training.iters_per_epoch
                )
                schedulers.append(scheduler)
            else:
                optimizers.append(self.trainer.optimizers[idx])
                schedulers.append(self.trainer.lr_scheduler_configs[idx].scheduler)

        self.trainer.optimizers = optimizers
        self.trainer.strategy.lr_scheduler_configs = [
            namedtuple('scheduler_config', ('scheduler',))(s) \
                for s in schedulers
        ]

    def get_train_iter(self, epoch, batch_idx, val=False):
        # Get epoch
        if self.render_only or self.test_only:
            epoch = 10000000
        elif self.cfg.params.load_from_weights:
            epoch = self.cfg.params.start_epoch

        # Get number of iterations per epoch
        num_iters = self.num_iters

        if not self.sample_with_replacement:
            num_iters = len(self.trainer.datamodule.train_dataset) \
                // self.cfg.training.batch_size

        # Get train iteration
        train_iter = (
            num_iters
        ) * epoch + batch_idx * self.cfg.training.num_gpus

        # Multi-GPU
        if not val:
            train_iter += self.global_rank

        # Regularization
        train_iter -= self.num_regularizer_pretraining_iters

        return train_iter

    def set_train_iter(self, train_iter):
        # Set model iter
        self.render_fn.model.set_iter(train_iter)

        # Set regularizer iter
        for reg in self.regularizers:
            reg.set_iter(train_iter)

    @property
    def regularizer_render_kwargs(self):
        render_kwargs = {}

        for reg in self.regularizers:
            render_kwargs.update(reg.render_kwargs)

        return render_kwargs

    @property
    def visualizer_render_kwargs(self):
        render_kwargs = {}

        for vis in self.visualizers:
            render_kwargs.update(vis.render_kwargs)

        return render_kwargs

    def training_step(self, batch, batch_idx):
        #return {}
        if self.render_only or self.test_only or self.interact_only:
            return {}

        ## Flag indicating the training has started
        self.training_started = True

        ## Tell model what training iter it is
        train_iter = self.get_train_iter(self.current_epoch, batch_idx)
        self.set_train_iter(train_iter)

        # Reset optimizers if necessary
        if self.needs_opt_reset(train_iter):
            self.reset_optimizers(train_iter)

        # Input batch
        batch = self.trainer.datamodule.train_dataset.format_batch(batch)

        # Results
        #with torch.autocast("cuda"):
        outputs = {}
        coords, rgb, weight = batch['coords'], batch['rgb'], batch['weight']

        results = self(coords, **self.regularizer_render_kwargs)

        # Image loss
        loss = 0.0

        if train_iter >= 0:
            # Calculate image loss and PSNR
            image_loss = self.loss(results['rgb'] * weight, rgb * weight, **batch)
            loss = image_loss

            outputs['train/psnr'] = psnr_gpu(results['rgb'], rgb).detach()

            # Print
            if self.cfg.params.print_loss:
                print(f"Image loss: {loss:.04f}, PSNR: {outputs['train/psnr']:.04f}, Iter: f{train_iter}")

        # Regularization losses
        reg_loss = 0.0

        for reg, cfg in zip(self.regularizers, self.regularizer_configs):
            reg.batch_size = self.trainer.datamodule.cur_batch_size
            cur_loss = reg.loss(batch, results, batch_idx) * reg.loss_weight()
            reg_loss += cur_loss

            if not reg.warming_up():
                loss += cur_loss

        # Print
        if self.cfg.params.print_loss and reg_loss > 0.0:
            print(f"Regularization loss: {reg_loss:.04f}, Iter: f{train_iter}")

        # Optimizers
        optimizers = self.optimizers(use_pl_optimizer=True)
        if not isinstance(optimizers, list):
            optimizers = [optimizers]

        # Gradient descent step
        for opt in optimizers: opt.zero_grad()
        self.manual_backward(loss)
        for opt in optimizers: opt.step()

        ## Scheduler step
        #if self.training_started:
        #    schedulers = self.lr_schedulers()

        #    for sched in schedulers:
        #        sched.step()

        ## Return
        outputs['train/loss'] = loss.detach()

        return outputs

    def training_epoch_end(self, outputs):
        if ((self.current_epoch + 1) % self.cfg.training.log_every) == 0:
            # Log
            mean = get_mean_outputs(outputs)

            for key, val in mean.items():
                self.log(key, val, on_epoch=True, on_step=False, sync_dist=True)
                print(f"{key}: {val}")

        # Scheduler step
        if self.training_started:
            schedulers = self.lr_schedulers()

            for sched in schedulers:
                sched.step()

        # Dataset update & resize
        self.trainer.datamodule.current_epoch = self.current_epoch + 1

        reset_val = (
            (self.current_epoch + 2) % self.trainer.datamodule.test_every == 0 \
                or (self.current_epoch + 1) % self.trainer.datamodule.test_every == 0 \
                or (self.current_epoch) % self.trainer.datamodule.test_every == 0
        ) or self.test_only
        resized = False

        if ((self.current_epoch + 1) % self.cfg.training.update_data_every) == 0:
            print("Updating data")
            resized = self.trainer.datamodule.update_data()

        if resized:
            print("Resized data")
            self.trainer.reset_train_dataloader(self)

            if 'reset_after_resize' in self.cfg.training and self.cfg.training.reset_after_resize:
                optimizers, schedulers = self.configure_optimizers()
                self.trainer.optimizers = optimizers
                self.trainer.lr_schedulers = [{'scheduler': s} for s in schedulers]

        if reset_val or resized:
            print("Re-setting dataloaders")
            self.trainer.reset_val_dataloader(self)

    def interact(self): # noqa
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=True):
                gui = NeRFGUI(
                    self, W=self.cfg.dataset.img_wh[0], H=self.cfg.dataset.img_wh[1]
                )
                gui.render()

        # Record time-to-render
        all_times = []

        # Get rays
        coords = self.trainer.datamodule.render_dataset[0]['coords'].cuda()
        origins = coords[..., :3]
        directions = coords[..., 3:6]
        extra = coords[..., 6:]

        # Initial pose
        initial_pose = np.eye(4)
        initial_pose[:3, :4] = self.trainer.datamodule.render_dataset.poses[0]

        initial_pose_inv = np.linalg.inv(initial_pose)
        initial_pose_inv = torch.FloatTensor(initial_pose_inv).cuda()

        # Visualizer kwargs
        visualizer_render_kwargs = self.visualizer_render_kwargs

        for idx in range(len(self.trainer.datamodule.render_dataset)):
            # Set rendering
            self.rendering = True

            # Render
            torch.cuda.synchronize()
            start_time = time.time()

            cur_pose = torch.FloatTensor(
                self.trainer.datamodule.render_dataset.poses[idx]
            ).cuda()
            pose_offset = cur_pose @ initial_pose_inv

            cur_origins = (pose_offset[:3, :3] @ origins.permute(1, 0)).permute(1, 0) + pose_offset[None, :3, -1]
            cur_directions = (pose_offset[:3, :3] @ directions.permute(1, 0)).permute(1, 0)
            cur_extra = extra
            cur_coords = torch.cat([cur_origins, cur_directions, cur_extra], -1)
            cur_results = self(cur_coords)

            torch.cuda.synchronize()

            # Record time
            all_times.append(time.time() - start_time)
            print(idx, all_times[-1])

            # Set not rendering
            self.rendering = False

    def validation_video(self, batch, batch_idx): # noqa
        if not self.trainer.is_global_zero:
            return

        # Render outputs
        all_videos = {
            'videos/rgb': []
        }

        # Function for adding outputs
        def _add_outputs(outputs):
            for key in outputs:
                all_videos[key] = np.array(outputs[key])

        # Loop over all render poses
        all_times = []

        for idx in range(len(self.trainer.datamodule.render_dataset)):
            # Convert batch to CUDA
            cur_batch = self.trainer.datamodule.render_dataset[idx]
            W, H = cur_batch['W'], cur_batch['H']
            self.cur_wh = [int(W), int(H)]

            for k in cur_batch:
                if isinstance(cur_batch[k], torch.Tensor):
                    cur_batch[k] = cur_batch[k].cuda()

            self.rendering = True

            # Render current pose
            visualizer_render_kwargs = self.visualizer_render_kwargs

            torch.cuda.synchronize()
            start_time = time.time()

            #cur_results = self.render_fn.model.embedding_model(cur_batch['coords'], {})
            #cur_results = self.render_fn.model.embedding_model.embeddings[0]({'rays': cur_batch['coords']}, {})

            cur_results = self(cur_batch['coords'], rendering=True, **visualizer_render_kwargs)
            #cur_results = self.model_script(cur_batch['coords'])

            torch.cuda.synchronize()

            all_times.append(time.time() - start_time)

            self.rendering = False
            print(idx, all_times[-1])

            cur_img = cur_results['rgb'].view(H, W, 3).cpu().numpy()
            #cur_img = cur_results.view(H, W, 3).cpu().numpy()

            # Format output RGB
            cur_img = cur_img.transpose(2, 0, 1)
            all_videos['videos/rgb'] = cur_img

            # Visualizer outputs
            for vis in self.visualizers:
                outputs = vis.validation_video(cur_batch, idx)
                _add_outputs(outputs)

            # Save outputs
            if self.cfg.params.save_results:
                epoch = str(self.current_epoch + 1)

                if self.render_only:
                    save_video_dir = self.cfg.params.save_video_dir.replace('val_videos', 'render')
                else:
                    save_video_dir = os.path.join(self.cfg.params.save_video_dir, epoch)

                for key in all_videos:
                    cur_im = np.squeeze(all_videos[key])
                    vid_suffix = key.split('/')[-1]

                    self.pmgr.mkdirs(os.path.join(save_video_dir, vid_suffix))

                    with self.pmgr.open(
                        os.path.join(save_video_dir, vid_suffix, f'{idx:04d}.png'),
                        'wb'
                    ) as f:
                        if len(cur_im.shape) == 3:
                            Image.fromarray(to8b(cur_im.transpose(1, 2, 0))).save(f)
                        else:
                            Image.fromarray(to8b(cur_im)).save(f)

        print("Average time:", np.mean(all_times[1:-1]))

    def validation_image(self, batch, batch_idx): # noqa
        batch_idx = batch_idx * self.cfg.training.num_gpus + self.global_rank

        # Forward
        coords, rgb, = batch['coords'], batch['rgb']
        coords = torch.clone(coords.view(-1, coords.shape[-1]))
        rgb = rgb.view(-1, 3)
        results = self(coords, **self.visualizer_render_kwargs)

        # Setup
        W, H = batch['W'], batch['H']
        self.cur_wh = [int(W), int(H)]
        all_images = {}

        # Logging
        img = results['rgb'].view(H, W, 3).cpu().numpy()
        img = img.transpose(2, 0, 1)
        img_gt = rgb.view(H, W, 3).cpu().numpy()
        img_gt = img_gt.transpose(2, 0, 1)

        all_images['eval/pred'] = img
        all_images['eval/gt'] = img_gt

        # Helper for adding outputs
        def _add_outputs(outputs):
            for key in outputs:
                if key not in all_images:
                    all_images[key] = np.clip(np.array(outputs[key]), 0.0, 1.0)

        # Visualizer images
        for vis in self.visualizers:
            if not self.trainer.datamodule.is_testing or vis.run_on_test:
                outputs = vis.validation_image(batch, batch_idx)
                _add_outputs(outputs)

        # Log all images
        for key in all_images:
            if 'eval/' in key:
                continue

            if self.cfg.params.tensorboard and self.cfg.training.num_gpus <= 1 and self.cfg.params.log_images:
                self.logger.experiment.add_images(
                    f'{key}_{batch_idx}',
                    all_images[key][None],
                    self.global_step,
                )

        # Save outputs
        if self.cfg.params.save_results:
            epoch = str(self.current_epoch + 1)

            if self.test_only:
                save_image_dir = self.cfg.params.save_image_dir.replace('val_images', 'testset')
            else:
                save_image_dir = os.path.join(self.cfg.params.save_image_dir, epoch)

            for key in all_images:
                im_suffix = key.split('/')[0]
                im_name = key.split('/')[-1]

                self.pmgr.mkdirs(os.path.join(save_image_dir, im_suffix))

                if im_suffix == 'data':
                    with self.pmgr.open(
                        os.path.join(save_image_dir, im_suffix, f'{batch_idx:04d}_{im_name}.npy'),
                        'wb'
                    ) as f:
                        all_images[key] = np.squeeze(all_images[key])
                        np.save(f, all_images[key])
                else:
                    with self.pmgr.open(
                        os.path.join(save_image_dir, im_suffix, f'{batch_idx:04d}_{im_name}.png'),
                        'wb'
                    ) as f:
                        all_images[key] = np.squeeze(all_images[key])

                        if len(all_images[key].shape) == 3:
                            Image.fromarray(to8b(all_images[key].transpose(1, 2, 0))).save(f)
                        else:
                            Image.fromarray(to8b(all_images[key])).save(f)

        # Output metrics
        outputs = {}
        outputs['val/loss'] = self.loss(results['rgb'], rgb, **batch).detach().cpu().numpy()
        outputs['val/psnr'] = psnr(img.transpose(1, 2, 0), img_gt.transpose(1, 2, 0))
        outputs['val/ssim'] = ssim(img.transpose(1, 2, 0), img_gt.transpose(1, 2, 0))

        return outputs


    def validation_step(self, batch, batch_idx):
        #with torch.autocast("cuda"):
        self.render_fn.eval()

        with torch.no_grad():
            train_iter = self.get_train_iter(self.current_epoch + 1, 0, True)
            self.set_train_iter(max(train_iter, 0))

            # Interact
            if self.interact_only:
                self.interact()
                exit(0)

            # Render video
            if batch_idx == 0 and \
                ((self.current_epoch + 1) % self.cfg.training.render_every == 0 or self.render_only):
                self.validation_video(batch, batch_idx)

            # Render image
            log = self.validation_image(batch, batch_idx)

        # Do not train
        if self.render_only:
            exit(0)

        self.render_fn.train()

        # Return
        return log

    def validation_epoch_end(self, outputs):
        # Log
        mean = get_mean_outputs(outputs, cpu=True)
        epoch = str(self.current_epoch + 1)
        self.pmgr.mkdirs(os.path.join(self.cfg.params.save_image_dir, epoch))

        with self.pmgr.open(
            os.path.join(self.cfg.params.save_image_dir, epoch, 'metrics.txt'),
            'w'
        ) as f:
            for key, val in mean.items():
                self.log(key, val, on_epoch=True, on_step=False, sync_dist=True)
                print(f"{key}: {val}")
                f.write(f'{key}: {float(val)}\n')

        return {}
