#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from .base import BaseRegularizer
from losses import loss_dict


class CoarseRegularizer(BaseRegularizer):
    def __init__(
        self,
        system,
        cfg
    ):
        super().__init__(system, cfg)

        self.loss_fn = loss_dict[self.cfg.loss.type]()

    def _loss(self, train_batch, batch_results, batch_idx):
        system = self.get_system()

        if self.cur_iter >= self.cfg.weight.stop_iters:
            return 0.0

        # Get inputs
        rays = train_batch['coords']
        rgb = train_batch['rgb']

        # Loss
        results = system(rays, coarse=True)
        pred_rgb = results['rgb']

        loss = self.loss_fn(
            pred_rgb,
            rgb
        )

        print("Coarse loss:", loss)

        return loss
