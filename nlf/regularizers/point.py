#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch

from .base import BaseRegularizer

from losses import loss_dict

import copy
from omegaconf import OmegaConf # @manual //github/third-party/omry/omegaconf:omegaconf


class PointRegularizer(BaseRegularizer):
    def __init__(
        self,
        system,
        cfg
    ):
        super().__init__(system, cfg)

        # Setup losses
        self.loss_fn = loss_dict[self.cfg.loss.type]()

    def _loss(self, train_batch, batch_results, batch_idx):
        #### Prepare ####
        system = self.get_system()

        ## Batch
        rays = train_batch['coords']

        ## tform constraints
        point_bias = system.render('embed_params', rays, return_bias=True)['params']
        point_bias = point_bias.view(-1, 3)

        loss = self.loss_fn(
            point_bias,
            torch.zeros_like(point_bias)
        )

        return loss
