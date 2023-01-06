#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch

from utils.intersect_utils import (
    intersect_axis_plane,
)

from .base import Intersect


class IntersectZPlane(Intersect):
    def __init__(
        self,
        z_channels,
        cfg,
        **kwargs
    ):
        super().__init__(z_channels, cfg, **kwargs)

        # Intersect hyper-params
        if self.use_dataset_bounds:
            self.initial = torch.tensor(-kwargs['system'].dm.train_dataset.near)
            self.end = torch.tensor(-kwargs['system'].dm.train_dataset.far)
        else:
            self.initial = torch.tensor(cfg.initial if 'initial' in cfg else 0.0)
            self.end = torch.tensor(cfg.end if 'end' in cfg else 1.0)

        self.num_repeat = cfg.num_repeat if 'num_repeat' in cfg else 1

        # Contract
        if self.contract_fn.contract_samples:
            self.initial = self.contract_fn.contract_distance(self.initial)
            self.end = self.contract_fn.contract_distance(self.end)

        if self.use_disparity:
            self.samples = torch.linspace(
                1.0 / self.end,
                1.0 / self.initial,
                z_channels // self.num_repeat,
                device='cuda'
            )

            self.samples = torch.flip(self.samples, [0])
        else:
            self.samples = torch.linspace(
                self.initial,
                self.end,
                z_channels // self.num_repeat,
                device='cuda'
            )
        
        self.samples = self.samples.repeat(self.num_repeat).view(-1, 1)

        # Calculate z scale
        if z_channels > 1:
            if "z_scale" in cfg:
                self.z_scale = cfg.z_scale
            elif "num_samples_for_scale" in cfg:
                self.z_scale = torch.abs(self.samples[1] - self.samples[0]) \
                    * (z_channels / float(cfg.num_samples_for_scale))
            else:
                self.z_scale = torch.abs(self.samples[1] - self.samples[0])
        else:
            self.z_scale = cfg.z_scale if "z_scale" in cfg else 1.0
        
        self.z_scale = torch.tensor(self.z_scale).view(-1, 1).cuda()

        # Local prediction
        self.use_local_prediction = cfg.use_local_prediction if 'use_local_prediction' in cfg else False
        self.voxel_size = cfg.voxel_size if 'voxel_size' in cfg else 1.0

    def intersect(self, rays, z_vals):
        z_vals = z_vals.view(z_vals.shape[0], -1)

        if self.clamp:
            z_vals = z_vals.clamp(self.initial, self.end)

        # Local prediction
        if self.use_local_prediction:
            rays_z = rays[..., 2:3]
            origin = torch.round(rays_z / self.voxel_size) * self.voxel_size
            z_vals = z_vals + origin

        # Calculate intersection
        dists = intersect_axis_plane(
            rays[..., None, :],
            z_vals,
            2,
            exclude=False
        )

        return dists


z_intersect_dict = {
    'z_plane': IntersectZPlane,
}
