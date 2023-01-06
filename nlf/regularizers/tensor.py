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
from kornia import create_meshgrid


def roll_1(tensor):
    return torch.cat(
        [
            tensor[:, :1],
            tensor[:, :-1],
        ],
        dim=1
    )


def roll_2(tensor):
    return torch.cat(
        [
            tensor[:, :, :1],
            tensor[:, :, :-1],
        ],
        dim=2
    )


def roll_3(tensor):
    return torch.cat(
        [
            tensor[:, :, :, :1],
            tensor[:, :, :, :-1],
        ],
        dim=3
    )


def tv_loss(tensor, skip_row, skip_col):
    if len(tensor.shape) == 4:
        diff_z = torch.square(torch.roll(tensor, 1, dims=1) - tensor)
        diff_y = torch.square(torch.roll(tensor, 1, dims=2) - tensor)
        diff_x = torch.square(torch.roll(tensor, 1, dims=3) - tensor)
        diff = diff_z + diff_y + diff_x
    elif len(tensor.shape) == 3:
        diff_y = torch.square(torch.roll(tensor, 1, dims=1) - tensor)
        diff_x = torch.square(torch.roll(tensor, 1, dims=2) - tensor)
        diff = diff_y + diff_x

        x = torch.linspace(0, tensor.shape[2] - 1, tensor.shape[2], dtype=torch.int32, device=tensor.device)
        y = torch.linspace(0, tensor.shape[1] - 1, tensor.shape[1], dtype=torch.int32, device=tensor.device)

        rem_x = torch.remainder(x, skip_col)
        rem_y = torch.remainder(y, skip_row)

        skip_x = ((rem_x == 0) | (rem_x == skip_col - 1)) & (skip_col != -1)
        skip_y = ((rem_y == 0) | (rem_y == skip_row - 1)) & (skip_row != -1)

        skip = torch.stack(list(torch.meshgrid([skip_y, skip_x])), dim=-1)
        skip = torch.any(skip, dim=-1, keepdim=False)[None]

        diff = diff * (~skip).float()

    return torch.mean(torch.sqrt(diff + 1e-8))


class TensorTV(BaseRegularizer):
    def __init__(
        self,
        system,
        cfg
    ):
        super().__init__(system, cfg)

        # Setup losses
        self.use_tv = cfg.use_tv
        self.opacity_weight = cfg.opacity_weight
        self.color_weight = cfg.color_weight

        self.skip_row = cfg.skip_row if 'skip_row' in cfg else -1
        self.skip_col = cfg.skip_col if 'skip_col' in cfg else -1

    def _loss(self, train_batch, batch_results, batch_idx):
        #### Prepare ####
        system = self.get_system()

        # Get tensors
        if system.is_subdivided:
            model = system.render_fn.model.ray_model.color_model
        else:
            model = system.render_fn.model.color_model

        loss = 0.0
        M = len(model.net.tensors)

        for tensor_prod in model.net.tensors:
            # Calculate mean TV loss
            num_opacity_basis = tensor_prod.num_opacity_basis
            N = len(tensor_prod.tensors)

            for i in range(N):
                tensor = tensor_prod.tensors[i].tensor

                # Color and appearance
                color_tensor = tensor[:-num_opacity_basis]
                opacity_tensor = tensor[-num_opacity_basis:]

                # TV loss
                if self.use_tv:
                    loss += self.opacity_weight * tv_loss(opacity_tensor, self.skip_row, self.skip_col) / (N * M)
                    loss += self.color_weight * tv_loss(color_tensor, self.skip_row, self.skip_col) / (N * M)
                else:
                    loss += self.opacity_weight * torch.mean(torch.abs(opacity_tensor)) / (N * M)

        # Return
        return loss
