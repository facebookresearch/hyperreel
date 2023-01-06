#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch

from utils.intersect_utils import (
    dot,
    intersect_axis_plane,
    intersect_voxel_grid,
    intersect_plane,
)

from .base import Intersect


class IntersectVoxelGrid(Intersect):
    def __init__(self, z_channels, cfg, **kwargs):
        super().__init__(z_channels, cfg, **kwargs)

        # Intersect params
        self.outward_facing = cfg.outward_facing if "outward_facing" in cfg else False
        self.fac = cfg.fac if "fac" in cfg else 1.0

        if self.use_dataset_bounds:
            self.initial = torch.tensor(list(cfg.initial) if 'initial' in cfg else kwargs['system'].dm.train_dataset.bbox_min * self.fac)
            self.end = torch.tensor(list(cfg.end) if 'end' in cfg else kwargs['system'].dm.train_dataset.bbox_max * self.fac)
        else:
            self.initial = torch.tensor(
                list(cfg.initial) if "initial" in cfg else [0.0, 0.0, 0.0]
            )
            self.end = torch.tensor(list(cfg.end) if "end" in cfg else [1.0, 1.0, 1.0])

        # Contract
        if self.contract_fn.contract_samples:
            self.initial = self.contract_fn.contract_distance(self.initial)
            self.end = self.contract_fn.contract_distance(self.end)

        # Max axis
        self.max_axis = cfg.max_axis if "max_axis" in cfg else False

        # Calculate samples
        z_channels = z_channels // 3
        self.samples = []

        for dim in range(3):
            samples = torch.linspace(
                self.initial[dim], self.end[dim], z_channels, device="cuda"
            )

            self.samples.append(samples)

        self.samples = torch.stack(self.samples, -1)

        if "z_scale" in cfg:
            self.z_scale = torch.tensor(list(cfg.z_scale)).cuda()
        elif z_channels > 1:
            self.z_scale = torch.abs(self.samples[1] - self.samples[0])
        else:
            self.z_scale = torch.tensor([1.0, 1.0, 1.0]).cuda()

        self.z_scale[self.z_scale == 0.0] = 1.0

        #print("Initial, end in contracted space:", self.initial, self.end)
        #print("Samples in contracted space:", self.samples)

        # Local prediction
        self.use_local_prediction = (
            cfg.use_local_prediction if "use_local_prediction" in cfg else False
        )
        self.voxel_size = torch.tensor(
            cfg.voxel_size if "voxel_size" in cfg else [1.0, 1.0, 1.0]
        ).cuda()

    def intersect(self, rays, z_vals):
        z_vals = z_vals.reshape(z_vals.shape[0], self.z_channels // 3, 3)

        # Outward facing
        if self.outward_facing:
            dir_sign = torch.sign(rays[..., 3:6])
            z_vals = z_vals * dir_sign[..., None, :]

        # Local prediction
        if self.use_local_prediction:
            rays_o = rays[..., 0:3]
            origin = torch.round(
                rays_o / self.voxel_size.unsqueeze(0)
            ) * self.voxel_size.unsqueeze(0)
            z_vals = z_vals + origin.unsqueeze(1)

        # Calculate intersection distance
        dists = intersect_voxel_grid(
            rays[..., None, :], torch.zeros_like(self.origin[None, None]), z_vals
        )

        # Max axis
        if self.max_axis:
            max_mask = torch.abs(rays[..., 3:6]) < (
                torch.max(torch.abs(rays[..., 3:6]), dim=-1, keepdim=True)[0] - 1e-8
            )
            max_mask = max_mask[..., None, :].repeat(1, self.z_channels // 3, 1)

            dists = dists.view(dists.shape[0], self.z_channels // 3, -1)
            dists = torch.where(
                max_mask,
                torch.zeros_like(dists),
                dists,
            ).view(dists.shape[0], self.z_channels)

        return dists


class IntersectDeformableVoxelGrid(Intersect):
    def __init__(self, z_channels, cfg, **kwargs):
        super().__init__(z_channels, cfg, **kwargs)

        # Starting normals
        self.num_axes = len(list(cfg.start_normal)) if 'start_normal' in cfg else 3
        self.start_normal = torch.tensor(
            list(cfg.start_normal) if 'start_normal' in cfg else \
            [
                [ 1.0, 0.0, 0.0, ],
                [ 0.0, 1.0, 0.0, ],
                [ 0.0, 0.0, 1.0, ],
            ]
        )
        self.normal_scale_factor = cfg.normal_scale_factor if 'normal_scale_factor' in cfg else 0.1

        # Intersect params
        self.outward_facing = cfg.outward_facing if "outward_facing" in cfg else False
        self.fac = cfg.fac if "fac" in cfg else 1.0

        if self.use_dataset_bounds:
            points = kwargs['system'].dm.train_dataset.all_points
            mask = kwargs['system'].dm.train_dataset.all_depth != 0.0

            valid_points = points[mask.repeat(1, 3)].reshape(-1, 3)
            self.initial = dot(self.start_normal.unsqueeze(0), valid_points.unsqueeze(1)).min(0)[0].cuda()
            self.end = dot(self.start_normal.unsqueeze(0), valid_points.unsqueeze(1)).max(0)[0].cuda()

            #print(self.initial, self.end)
            #exit()
        else:
            self.initial = torch.tensor(
                list(cfg.initial) if "initial" in cfg else [0.0, 0.0, 0.0]
            )
            self.end = torch.tensor(list(cfg.end) if "end" in cfg else [1.0, 1.0, 1.0])

        self.start_normal = self.start_normal.cuda()

        # Contract
        if self.contract_fn.contract_samples:
            self.initial = self.contract_fn.contract_distance(self.initial)
            self.end = self.contract_fn.contract_distance(self.end)

        # Max axis
        self.max_axis = cfg.max_axis if "max_axis" in cfg else False

        # Calculate samples
        z_channels = z_channels // self.num_axes
        self.samples = []

        for dim in range(self.num_axes):
            samples = torch.linspace(
                self.initial[dim], self.end[dim], z_channels, device="cuda"
            )

            self.samples.append(samples)

        self.samples = torch.stack(self.samples, -1).view(-1, 1)

        # Calculate z scale
        if "z_scale" in cfg:
            self.z_scale = torch.tensor(list(cfg.z_scale)).cuda()
        elif z_channels > 1:
            self.z_scale = torch.abs(self.samples[1] - self.samples[0])
        else:
            self.z_scale = torch.tensor([1.0 for i in range(self.num_axes)]).cuda()

        self.z_scale[self.z_scale == 0.0] = 1.0

    def process_z_vals(self, z_vals):
        z_vals = z_vals.view(z_vals.shape[0], -1, 4)
        d = super().process_z_vals(z_vals[..., -1])
        return torch.cat([z_vals[..., :3], d[..., None]], -1).view(z_vals.shape[0], -1)

    def intersect(self, rays, z_vals):
        z_vals = z_vals.reshape(
            z_vals.shape[0],
            -1,
            4,
        )

        normal = z_vals[..., :3]
        distance = z_vals[..., -1]

        # Correct normals
        normal = normal.view(
            -1,
            self.num_axes,
            3
        ) * self.normal_scale_factor + self.start_normal.unsqueeze(0)
        normal = normal.view(z_vals.shape[0], -1, 3)
        normal = torch.nn.functional.normalize(normal, dim=-1)

        # Calculate intersection distance
        dists = intersect_plane(
            rays[..., None, :],
            normal,
            distance
        )

        return dists


voxel_intersect_dict = {
    "voxel_grid": IntersectVoxelGrid,
    "deformable_voxel_grid": IntersectDeformableVoxelGrid,
}
