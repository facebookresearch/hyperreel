#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
import torch.nn as nn

from .base import BaseRegularizer


class TVLoss(nn.Module):
    def __init__(self, TVLoss_weight=1):
        super(TVLoss,self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self,x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]

        count_h = self._tensor_size(x[:,:,1:,:])
        count_w = self._tensor_size(x[:,:,:,1:])
        h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()
        w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()

        return self.TVLoss_weight*2*(h_tv/count_h+w_tv/count_w)/batch_size

    def _tensor_size(self,t):
        return t.size()[1]*t.size()[2]*t.size()[3]


class TensoRF(BaseRegularizer):
    def __init__(
        self,
        system,
        cfg
    ):
        super().__init__(system, cfg)

        # Setup losses
        self.cfg = cfg
        self.tvreg = TVLoss()

        self.update_AlphaMask_list = cfg.update_AlphaMask_list

        self.lr_factor = self.cfg.lr_decay_target_ratio**(1/self.cfg.n_iters)
        self.total_num_tv_iters = self.cfg.total_num_tv_iters if 'total_num_tv_iters' in self.cfg else \
            int(np.round( (np.log(1e-4) / np.log(self.cfg.lr_decay_target_ratio)) * self.cfg.n_iters ))

        self.L1_reg_weight = self.cfg.L1_weight_initial
        self.TV_weight_density = self.cfg.TV_weight_density
        self.TV_weight_app = self.cfg.TV_weight_app

    def _loss(self, train_batch, batch_results, batch_idx):
        #### Prepare ####
        system = self.get_system()

        # Get tensors
        if system.is_subdivided:
            tensorf = system.render_fn.model.ray_model.color_model.net
        else:
            tensorf = system.render_fn.model.color_model.net

        total_loss = 0.0

        # L1 loss
        if self.L1_reg_weight > 0:
            loss_reg_L1 = tensorf.density_L1()
            total_loss += self.L1_reg_weight * loss_reg_L1

        # Return if decayed TV enough
        if self.cur_iter > self.total_num_tv_iters:
            return total_loss

        # TV Loss
        if self.TV_weight_density > 0:
            self.TV_weight_density *= self.lr_factor
            loss_tv = tensorf.TV_loss_density(self.tvreg) * self.cfg.TV_weight_density
            total_loss = total_loss + loss_tv

        if self.TV_weight_app > 0:
            self.TV_weight_app *= self.lr_factor
            loss_tv = loss_tv + tensorf.TV_loss_app(self.tvreg) * self.cfg.TV_weight_app
            total_loss = total_loss + loss_tv

        return total_loss

    def set_iter(self, iteration):
        super().set_iter(iteration)

        if len(self.update_AlphaMask_list) > 0 and self.cur_iter == self.update_AlphaMask_list[0]:
            self.L1_reg_weight = self.cfg.L1_weight_rest
            print("continuing L1_reg_weight", self.L1_reg_weight)
