#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn
import numpy as np

from nlf.rendering import (
    render_chunked,
)

from losses import loss_dict
from utils.ray_utils import get_weight_map


class BaseRegularizer(nn.Module):
    def __init__(
        self,
        system,
        cfg
    ):
        super().__init__()

        self.cfg = cfg
        self.net_chunk = cfg.net_chunk if 'net_chunk' in cfg else 32768
        self.batch_size = cfg.batch_size if 'batch_size' in cfg else 4096
        self.weight = cfg.weight if 'weight' in cfg else 0.0

        self.use_inp_freq = cfg.use_inp_freq if 'use_inp_freq' in cfg else 0
        self.wait_iters = cfg.wait_iters if 'wait_iters' in cfg else 0
        self.warmup_iters = cfg.warmup_iters if 'warmup_iters' in cfg else 0
        self.stop_iters = cfg.stop_iters if 'stop_iters' in cfg else float("inf")

        ## (Hack) Prevent from storing system variables
        self.systems = [system]

        ## Losses
        self.build_losses()

    def build_losses(self):
        self.loss_fns = {}

        for key in self.cfg.keys():
            if 'loss' in key:
                loss_fn = loss_dict[self.cfg[key].type](self.cfg[key])
                self.loss_fns[key] = loss_fn

            if 'weight_map' in key:
                for attr in self.cfg[key].keys():
                    if attr == 'angle_std':
                        angle_std = float(np.radians(self.cfg[key].angle_std))
                        self.cfg[key].angle_std = angle_std

    def warming_up(self):
        return self.cur_iter < self.warmup_iters

    def _do_loss(self, name):
        return self.cfg[name].wait_iters != 'inf' and \
            self.cur_iter >= self.cfg[name].wait_iters \
            and self.cfg[name].weight > 0

    def _loss_fn(self, name, *args):
        return self.loss_fns[name](
            *args
        ) * self.cfg[name].weight

    def get_system(self):
        return self.systems[0]

    def get_dataset(self):
        system = self.get_system()

        if 'dataset' in self.cfg:
            return system.trainer.datamodule.regularizer_datasets[self.cfg.type]
        else:
            return None

    def to_ndc(self, rays):
        dataset = self.get_dataset()
        return dataset.to_ndc(rays)

    def get_batch(self, train_batch, batch_idx, apply_ndc=False):
        system = self.get_system()
        dataset = self.get_dataset()
        batch = {}

        use_inp = self.use_inp_freq == 0 or \
            (
                (float(self.use_inp_freq) != float("inf")) and \
                (batch_idx % self.use_inp_freq == 0)
            )

        if dataset is not None and not use_inp:
            ## Use regularizer dataset
            if 'jitter' in self.cfg:
                batch = dataset.get_batch(batch_idx, self.batch_size, self.cfg.jitter)
            else:
                batch = dataset.get_batch(batch_idx, self.batch_size, None)

            ## Convert to correct device
            for k in batch.keys():
                batch[k] = batch[k].type_as(train_batch['coords'])

        else:
            ## Use training dataset
            batch['coords'] = train_batch['coords'][:self.batch_size]
            batch['rgb'] = train_batch['rgb'][:self.batch_size]

        ## Convert to correct device
        for k in batch.keys():
            batch[k] = batch[k].type_as(train_batch['coords'])

        return batch

    def forward(self, x):
        return x

    def run_chunked(self, x):
        return render_chunked(
            x,
            self,
            { },
            chunk=self.cfg.ray_chunk
        )

    def get_weight_map(
        self,
        rays,
        jitter_rays,
        name,
    ):
        with torch.no_grad():
            return get_weight_map(
                rays,
                jitter_rays,
                self.cfg[name],
                softmax=False,
            )

    def _loss(self, batch, batch_idx):
        return 0.0

    def loss(self, batch, batch_results, batch_idx):
        if self.cur_iter < 0:
            return 0.0
        elif self.cur_iter >= self.stop_iters:
            return 0.0

        return self._loss(batch, batch_results, batch_idx)

    def loss_weight(self):
        system = self.get_system()

        if isinstance(self.weight, float):
            return self.weight
        elif self.weight.type == 'exponential_decay':
            weight_num_iters = self.weight.num_epochs \
                * len(system.trainer.datamodule.train_dataset) // system.trainer.datamodule.cur_batch_size
            exponent = self.cur_iter / weight_num_iters
            return self.weight.start * np.power(self.weight.decay, exponent)
        else:
            return self.weight

    def set_iter(self, i):
        self.cur_iter = i - self.wait_iters

    def validation_video(self, batch, batch_idx):
        return {}

    def validation_image(self, batch, batch_idx):
        return {}

    @property
    def render_kwargs(self):
        return {}
