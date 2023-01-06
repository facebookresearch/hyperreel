#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
from torch import nn

from typing import Dict, List
from nlf.activations import get_activation
from ..contract import contract_dict

from utils.ray_utils import (
    get_ray_density
)

from utils.intersect_utils import (
    sort_z,
    sort_with
)


def uniform_weight(cfg):
    def weight_fn(rays, dists):
        return torch.ones_like(dists)

    return weight_fn


def ease_max_weight(cfg):
    weight_start = cfg.weight_start if 'weight_start' in cfg else 1.0
    weight_end = cfg.weight_end if 'weight_end' in cfg else 0.95

    def weight_fn(rays, dists):
        rays_norm = torch.abs(nn.functional.normalize(rays[..., 3:6], p=float("inf"), dim=-1))
        weights = ((rays_norm - weight_end) / (weight_start - weight_end)).clamp(0, 1)
        return weights.unsqueeze(1).repeat(1, dists.shape[1] // 3, 1).view(
            weights.shape[0], -1
        )

    return weight_fn


weight_fn_dict = {
    'uniform': uniform_weight,
    'ease_max': ease_max_weight,
}


class Intersect(nn.Module):
    sort_outputs: List[str]

    def __init__(
        self,
        z_channels,
        cfg,
        **kwargs
    ):
        super().__init__()

        self.cur_iter = 0
        self.cfg = cfg

        # Input/output size
        self.z_channels = z_channels
        self.in_density_field = cfg.in_density_field if 'in_density_field' in cfg else 'sigma'
        self.out_points = cfg.out_points if 'out_points' in cfg else None
        self.out_distance = cfg.out_distance if 'out_distance' in cfg else None

        # Other common parameters
        self.forward_facing = cfg.forward_facing if 'forward_facing' in cfg else False
        self.normalize = cfg.normalize if 'normalize' in cfg else False
        self.residual_z = cfg.residual_z if 'residual_z' in cfg else False
        self.residual_distance = cfg.residual_distance if 'residual_distance' in cfg else False
        self.sort = cfg.sort if 'sort' in cfg else False
        self.clamp = cfg.clamp if 'clamp' in cfg else False

        self.use_dataset_bounds = cfg.use_dataset_bounds if 'use_dataset_bounds' in cfg else False
        self.use_disparity = cfg.use_disparity if 'use_disparity' in cfg else False
        self.use_sigma = cfg.use_sigma if 'use_sigma' in cfg else False

        # Origin
        self.origin = torch.tensor(cfg.origin if 'origin' in cfg else [0.0, 0.0, 0.0], device='cuda')

        # Minimum intersect distance
        if self.use_dataset_bounds:
            self.near = cfg.near if 'near' in cfg else kwargs['system'].dm.train_dataset.near
        else:
            self.near = cfg.near if 'near' in cfg else 0.0

        #self.near = cfg.near if 'near' in cfg else 0.0
        self.far = cfg.far if 'far' in cfg else float("inf")

        # Sorting
        self.weight_fn = weight_fn_dict[cfg.weight_fn.type](cfg.weight_fn) if 'weight_fn' in cfg else None
        self.sort_outputs = list(cfg.sort_outputs) if 'sort_outputs' in cfg else []

        if self.weight_fn is not None:
            self.sort_outputs.append('weights')

        # Mask
        if 'mask' in cfg:
            self.mask_stop_iters = cfg.mask.stop_iters if 'stop_iters' in cfg.mask else float("inf")
        else:
            self.mask_stop_iters = float("inf")

        # Contract function
        if 'contract' in cfg:
            self.contract_fn = contract_dict[cfg.contract.type](
                cfg.contract,
                **kwargs
            )
            self.contract_stop_iters = cfg.contract.stop_iters if 'stop_iters' in cfg.contract else float("inf")
        else:
            self.contract_fn = contract_dict['identity']({})
            self.contract_stop_iters = float("inf")

        # Activation
        self.activation = get_activation(cfg.activation if 'activation' in cfg else 'identity')

        # Dropout params
        self.use_dropout = 'dropout' in cfg
        self.dropout_frequency = cfg.dropout.frequency if 'dropout' in cfg else 2
        self.dropout_stop_iter = cfg.dropout.stop_iter if 'dropout' in cfg else float("inf")

    def process_z_vals(self, z_vals):
        z_vals = z_vals.view(z_vals.shape[0], -1, self.z_scale.shape[-1]) * self.z_scale[None] + self.samples[None]
        z_vals = z_vals.view(z_vals.shape[0], -1)

        if self.contract_fn.contract_samples:
            z_vals = self.contract_fn.inverse_contract_distance(z_vals)
        elif self.use_disparity:
            z_vals = torch.where(
                torch.abs(z_vals) < 1e-8, 1e8 * torch.ones_like(z_vals), z_vals
            )
            z_vals = 1.0 / z_vals

        return z_vals

    def forward(self, rays: torch.Tensor, x: Dict[str, torch.Tensor], render_kwargs: Dict[str, str]):
        rays = torch.cat(
            [
                rays[..., :3] - self.origin[None],
                rays[..., 3:6],
            ],
            dim=-1
        )

        ## Z value processing
        z_vals = x['z_vals'].view(rays.shape[0], -1)

        # Z activation and sigma
        if self.use_sigma and self.in_density_field in x:
            sigma = x[self.in_density_field].view(z_vals.shape[0], -1)
        else:
            sigma = torch.zeros(z_vals.shape[0], z_vals.shape[1], device=z_vals.device)

        z_vals = self.activation(z_vals.view(z_vals.shape[0], sigma.shape[1], -1)) * (1 - sigma.unsqueeze(-1))
        z_vals = z_vals.view(z_vals.shape[0], -1)

        # Apply offset
        if self.use_dropout and ((self.cur_iter % self.dropout_frequency) == 0) and self.cur_iter < self.dropout_stop_iter and self.training:
            z_vals = torch.zeros_like(z_vals)

        # Add samples and contract
        z_vals = self.process_z_vals(z_vals)

        # Residual distances
        if self.residual_z and 'last_z' in x:
            last_z = x['last_z']
            last_z = last_z.view(last_z.shape[0], -1, 1)
            z_vals = z_vals.view(z_vals.shape[0], last_z.shape[1], -1)
            z_vals = (z_vals + last_z).view(z_vals.shape[0], -1)
        else:
            x['last_z'] = z_vals

        # Get distances
        dists = self.intersect(rays, z_vals)

        # Calculate weights
        if self.weight_fn is not None:
            weights = self.weight_fn(rays, dists)
        else:
            weights = torch.ones_like(dists)

        if 'weights' not in x or x['weights'].shape[1] != weights.shape[1]:
            x['weights'] = weights.unsqueeze(-1)
        else:
            x['weights'] = x['weights'] * weights.unsqueeze(-1)

        # Mask
        mask = (dists <= self.near) | (dists >= self.far) | (weights == 0.0)

        if self.cur_iter > self.mask_stop_iters:
            mask = torch.zeros_like(mask)

        dists = torch.where(
            mask,
            torch.zeros_like(dists),
            dists
        )

        # Sort
        if self.sort:
            dists, sort_idx = sort_z(dists, 1, False)

            for output_key in self.sort_outputs:
                x[output_key] = sort_with(sort_idx, x[output_key])

        # Mask again
        dists = dists.unsqueeze(-1)
        mask = (dists == 0.0)

        # Residual distances
        if self.residual_distance and 'last_distance' in x:
            last_dists = x['last_distance']
            last_dists = last_dists.view(last_dists.shape[0], -1, 1, 1)
            dists = dists.view(dists.shape[0], last_dists.shape[1], -1, 1)
            dists = (dists + last_dists).view(dists.shape[0], -1, 1)
        else:
            x['last_distance'] = dists

        # Get points
        points = rays[..., None, :3] + rays[..., None, 3:6] * dists

        # Normalize output
        if self.normalize:
            r = (z_vals[..., None] + 1)
            fac = 1.0 / torch.sqrt(((-r + 1) * (-r + 1) + r * r) + 1e-8)

            points = torch.cat(
                [
                    points[..., :2] * fac,
                    points[..., 2:3]
                ],
                -1
            )

        # Contract
        if not (self.cur_iter > self.contract_stop_iters):
            points, dists = self.contract_fn.contract_points_and_distance(
                rays[..., :3], points, dists
            )
            dists = torch.where(mask, torch.zeros_like(dists), dists)
        
        if self.out_points is not None:
            x[self.out_points] = points

        if self.out_distance is not None:
            x[self.out_distance] = dists

        # Return
        x['points'] = points
        x['distances'] = dists
        x['z_vals'] = z_vals

        return x

    def intersect(self, rays, z_vals):
        pass

    def set_iter(self, i):
        self.cur_iter = i
