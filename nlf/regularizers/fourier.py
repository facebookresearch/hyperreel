#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch

from .base import BaseRegularizer
from datasets.fourier import fft_rgb


class FourierRegularizer(BaseRegularizer):
    def __init__(
        self,
        system,
        cfg
    ):
        super().__init__(system, cfg)

        self.range = cfg.range
        self.use_absolute = 'complex' not in cfg.fourier_loss.type

    def loss(self, train_batch, batch_results, batch_idx):
        system = self.get_system()
        dataset = self.get_dataset()

        ## Get rays
        all_rgb_fft = dataset.all_rgb_fft.to(
            train_batch['rays'].device
        )

        ## Query
        rays = dataset.get_random_rays(self.cfg.range).type_as(
            train_batch['rays']
        )
        rgb = system(rays)['rgb'].view(
            1, system.img_wh[1], system.img_wh[0], 3
        )

        #### Losses ####

        all_losses = {
            loss: 0.0 for loss in self.loss_fns.keys()
        }

        if self._do_loss('fourier_loss'):
            rgb_fft = fft_rgb(rgb)

            if self.use_absolute:
                rgb_fft = torch.abs(rgb_fft)
                all_rgb_fft = torch.abs(all_rgb_fft)

            all_losses['fourier_loss'] = self._loss_fn(
                'fourier_loss',
                rgb_fft,
                all_rgb_fft
            )

        ## Total loss
        total_loss = 0.0

        for name in all_losses.keys():
            print(name + ':', all_losses[name])
            total_loss += all_losses[name]

        return total_loss
