#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
import torch.nn.functional as F

from nlf.activations import get_activation
from nlf.param import pluecker_pos, pluecker_pos_cylinder

from utils.intersect_utils import (
    dot,
    min_sphere_radius,
    min_cylinder_radius,
    intersect_cylinder,
    intersect_plane,
    intersect_sphere,
)

from .base import Intersect


class IntersectPlane(Intersect):
    def __init__(self, z_channels, cfg, **kwargs):
        super().__init__(z_channels, cfg, **kwargs)

        self.global_near = cfg.global_near if "global_near" in cfg else -1.0

        self.initial = torch.tensor(cfg.initial if "initial" in cfg else -1.0)
        self.end = torch.tensor(cfg.end if "end" in cfg else 1.0)

        if self.contract_fn.contract_samples:
            self.initial = self.contract_fn.contract_distance(self.initial)
            self.end = self.contract_fn.contract_distance(self.end)

        if self.use_disparity:
            disparities = torch.linspace(
                1.0 / self.end, 1.0 / self.initial, z_channels, device="cuda"
            )
            self.samples = torch.flip(disparities, [0])
        else:
            self.samples = torch.linspace(
                self.initial, self.end, z_channels, device="cuda"
            )

        if z_channels > 1:
            self.z_scale = (
                cfg.z_scale
                if "z_scale" in cfg
                else torch.abs(self.samples[1] - self.samples[0])
            )
        else:
            self.z_scale = cfg.z_scale if "z_scale" in cfg else 1.0

        if self.z_scale == 0.0:
            self.z_scale = 1.0

    def process_z_vals(self, normal):
        normal = normal.view(normal.shape[0], -1, 4)
        d = super().process_z_vals(normal[..., -1])
        return torch.cat([normal[..., :3], d[..., None]], -1).view(normal.shape[0], -1)

    def intersect(self, rays, normal):
        normal = normal.view(normal.shape[0], -1, 4)

        return intersect_plane(rays[..., None, :], normal[..., :3], normal[..., -1])


class IntersectEuclideanDistance(Intersect):
    def __init__(self, z_channels, cfg, **kwargs):
        super().__init__(z_channels, cfg, **kwargs)

        # Intersect params
        self.global_near = cfg.global_near if "global_near" in cfg else -1.0
        self.min_radius = cfg.min_radius if "min_radius" in cfg else 0.0

        self.initial = torch.tensor(cfg.initial if "initial" in cfg else 0.0)
        self.end = torch.tensor(cfg.end if "end" in cfg else 2.0)

        # Contract
        if self.contract_fn.contract_samples:
            self.initial = self.contract_fn.contract_distance(self.initial)
            self.end = self.contract_fn.contract_distance(self.end)

        if self.use_disparity:
            disparities = torch.linspace(
                1.0 / self.end, 1.0 / self.initial, z_channels, device="cuda"
            )
            self.samples = torch.flip(disparities, [0])
        else:
            self.samples = torch.linspace(
                self.initial, self.end, z_channels, device="cuda"
            )

        if z_channels > 1:
            self.z_scale = (
                cfg.z_scale
                if "z_scale" in cfg
                else torch.abs(self.samples[1] - self.samples[0])
            )
        else:
            self.z_scale = cfg.z_scale if "z_scale" in cfg else 1.0

        if self.z_scale == 0.0:
            self.z_scale = 1.0

    def intersect(self, rays, distance):
        distance = distance.view(distance.shape[0], -1)

        # Min radius
        if self.min_radius > 0:
            d_from_o = torch.linalg.norm(rays[..., :3], dim=-1)
            distance_offset = distance - d_from_o.unsqueeze(1)
        else:
            distance_offset = 0

        # Add distance offset
        distance = distance + distance_offset

        return distance


class IntersectEuclideanDistanceUnified(Intersect):
    def __init__(self, z_channels, cfg, **kwargs):
        super().__init__(z_channels, cfg, **kwargs)

        # Intersect params
        self.global_near = cfg.global_near if "global_near" in cfg else -1.0
        self.min_radius = cfg.min_radius if "min_radius" in cfg else 0.0

        if self.use_dataset_bounds:
            self.initial = torch.tensor(cfg.initial if 'initial' in cfg else -kwargs['system'].dm.train_dataset.far)
            self.end = torch.tensor(cfg.end if 'end' in cfg else kwargs['system'].dm.train_dataset.far)
        else:
            self.initial = torch.tensor(cfg.initial if 'initial' in cfg else 0.0)
            self.end = torch.tensor(cfg.end if 'end' in cfg else 1.0)

        # Contract
        if self.contract_fn.contract_samples:
            self.initial = self.contract_fn.contract_distance(self.initial)
            self.end = self.contract_fn.contract_distance(self.end)

        self.samples = torch.linspace(
            self.initial, self.end, z_channels, device="cuda"
        ).view(-1, 1)

        # Calculate z scale
        if z_channels > 1:
            self.z_scale = (
                cfg.z_scale
                if "z_scale" in cfg
                else torch.abs(self.samples[1] - self.samples[0])
            )
        else:
            self.z_scale = cfg.z_scale if "z_scale" in cfg else 1.0

        self.z_scale = torch.tensor(self.z_scale).view(-1, 1).cuda()

        # Unify positions
        self.unify_fn = pluecker_pos(None)

    def intersect(self, rays, distance):
        distance = distance.view(distance.shape[0], -1)

        # Base distance
        rays_o, rays_d = rays[..., :3], rays[..., 3:6]
        base_pos = self.unify_fn(rays)
        diff = base_pos - rays_o

        # Add distance offset
        distance = distance + (
            torch.sign(dot(rays_d, diff)) * torch.norm(diff, dim=-1)
        ).unsqueeze(1)

        return distance


class IntersectCylinderOld(Intersect):
    def __init__(self, z_channels, cfg, **kwargs):
        super().__init__(z_channels, cfg, **kwargs)

        # Intersect hyper-params
        if self.use_dataset_bounds:
            self.initial = torch.tensor(cfg.initial if 'initial' in cfg else kwargs['system'].dm.train_dataset.near * 1.5)
            self.end = torch.tensor(cfg.end if 'end' in cfg else kwargs['system'].dm.train_dataset.far * 1.5)
        else:
            self.initial = torch.tensor(cfg.initial if 'initial' in cfg else 0.0)
            self.end = torch.tensor(cfg.end if 'end' in cfg else 1.0)

        # Contract
        if self.contract_fn.contract_samples:
            self.initial = self.contract_fn.contract_distance(self.initial)
            self.end = self.contract_fn.contract_distance(self.end)

        # Origin scale
        self.origin_scale_factor = cfg.origin_scale_factor if 'origin_scale_factor' in cfg else 0.0
        self.origin_initial = torch.tensor(cfg.origin_initial if 'origin_initial' in cfg else [1.0, 1.0, 1.0]).cuda()

        # Flip axes
        self.flip_axes = cfg.flip_axes if 'flip_axes' in cfg else False

        # Calculate samples
        self.samples = torch.linspace(
            self.initial, self.end, z_channels, device="cuda"
        ).view(-1, 1)

        # Calculate z scale
        if z_channels > 1:
            self.z_scale = (
                cfg.z_scale
                if "z_scale" in cfg
                else torch.abs(self.samples[1] - self.samples[0])
            )
        else:
            self.z_scale = cfg.z_scale if "z_scale" in cfg else 1.0

        self.z_scale = torch.tensor(self.z_scale).view(-1, 1).cuda()

        # Unify positions
        self.unify_fn = pluecker_pos(None)

    def process_origins(self, origins):
        origins = origins * self.origin_scale_factor + self.origin_initial.view(1, 1, 3)
        return origins

    def process_z_vals(self, z_vals):
        z_vals = z_vals.view(z_vals.shape[0], -1, 4)
        origins = self.process_origins(z_vals[..., :3])
        radii = super().process_z_vals(z_vals[..., -1])
        return torch.cat([origins, radii[..., None]], -1).view(z_vals.shape[0], -1)

    def intersect(self, rays, z_vals):
        z_vals = z_vals.view(z_vals.shape[0], self.z_channels, 4)
        origins = z_vals[..., :3]
        radii = z_vals[..., -1]

        rays = torch.cat(
            [
                rays[..., None, 0:3] * origins,
                rays[..., None, 3:6] * origins,
            ],
            -1
        )

        # Calculate intersection
        return intersect_cylinder(
            rays,
            torch.zeros_like(origins),
            radii,
        )


class IntersectCylinderNew(Intersect):
    def __init__(self, z_channels, cfg, **kwargs):
        super().__init__(z_channels, cfg, **kwargs)

        # Intersect hyper-params
        if self.use_dataset_bounds:
            if cfg.outward_facing:
                self.initial = torch.tensor(cfg.initial if 'initial' in cfg else kwargs['system'].dm.train_dataset.near * 1.5)
            else:
                self.initial = torch.tensor(cfg.initial if 'initial' in cfg else -kwargs['system'].dm.train_dataset.far * 1.5)

            self.end = torch.tensor(cfg.end if 'end' in cfg else kwargs['system'].dm.train_dataset.far * 1.5)
        else:
            self.initial = torch.tensor(cfg.initial if 'initial' in cfg else 0.0)
            self.end = torch.tensor(cfg.end if 'end' in cfg else 1.0)

        # Contract
        if self.contract_fn.contract_samples:
            self.initial = self.contract_fn.contract_distance(self.initial)
            self.end = self.contract_fn.contract_distance(self.end)

        # Origin scale
        self.resize_scale_factor = cfg.resize_scale_factor if 'resize_scale_factor' in cfg else 0.0
        self.resize_initial = torch.tensor(cfg.resize_initial if 'resize_initial' in cfg else [1.0, 1.0, 1.0]).cuda()
        self.origin_scale_factor = cfg.origin_scale_factor if 'origin_scale_factor' in cfg else 0.0

        # Flip axes
        self.flip_axes = cfg.flip_axes if 'flip_axes' in cfg else False

        # Calculate samples
        self.samples = torch.linspace(
            self.initial, self.end, z_channels, device="cuda"
        ).view(-1, 1)

        # Calculate z scale
        if z_channels > 1:
            self.z_scale = (
                cfg.z_scale
                if "z_scale" in cfg
                else torch.abs(self.samples[1] - self.samples[0])
            )
        else:
            self.z_scale = cfg.z_scale if "z_scale" in cfg else 1.0

        self.z_scale = torch.tensor(self.z_scale).view(-1, 1).cuda()

        # Unify positions
        self.unify_fn = pluecker_pos_cylinder(None)

    def process_origins(self, origins):
        origins = origins * self.origin_scale_factor
        return origins

    def process_resize(self, resize):
        resize = resize * self.resize_scale_factor + self.resize_initial.view(1, 1, 3)
        return resize

    def process_z_vals(self, z_vals):
        z_vals = z_vals.view(z_vals.shape[0], -1, 8)
        origins = self.process_origins(z_vals[..., :3])
        resize = self.process_resize(z_vals[..., 3:6])
        raw_offsets = super().process_z_vals(z_vals[..., -2])
        radii = super().process_z_vals(z_vals[..., -1])
        return torch.cat([origins, resize, raw_offsets[..., None], radii[..., None]], -1).view(z_vals.shape[0], -1)

    def intersect(self, rays, z_vals):
        z_vals = z_vals.view(z_vals.shape[0], self.z_channels, 8)
        origins = z_vals[..., :3]
        resize = z_vals[..., 3:6]
        raw_offsets = z_vals[..., -2]
        radii = z_vals[..., -1]

        # Transform the space
        rays_o = (rays[..., None, 0:3] - origins) * resize
        rays_d = rays[..., None, 3:6] * resize
        rays = torch.cat(
            [
                rays_o,
                torch.nn.functional.normalize(rays_d, p=2.0, dim=-1),
            ],
            -1
        )

        # Calculate intersection distances (in transformed space)
        t = intersect_cylinder(
            rays,
            torch.zeros_like(origins),
            radii,
        )

        # Recycle samples for not-hit cylinders
        min_radius = min_cylinder_radius(rays, torch.zeros_like(rays[..., :3]))

        base_pos = self.unify_fn(rays)
        rays_o_cyl = torch.cat([rays[..., 0:1], torch.zeros_like(rays[..., 1:2]), rays[..., 2:3]], -1)
        rays_d_cyl = torch.cat([rays[..., 3:4], torch.zeros_like(rays[..., 4:5]), rays[..., 5:6]], -1)
        diff = (base_pos - rays_o_cyl)

        base_distance = torch.sign(dot(rays_d_cyl, diff)) * torch.norm(diff, dim=-1) / torch.norm(rays_d_cyl, dim=-1)

        t = torch.where(
            torch.abs(radii) < min_radius + 4 * self.z_scale,
            raw_offsets + base_distance,
            t
        )

        # Transform distances
        return t / (torch.norm(rays_d, dim=-1) + 1e-5)


class IntersectSphereOld(Intersect):
    def __init__(self, z_channels, cfg, **kwargs):
        super().__init__(z_channels, cfg, **kwargs)

        # Intersect hyper-params
        if self.use_dataset_bounds:
            self.initial = torch.tensor(cfg.initial if 'initial' in cfg else kwargs['system'].dm.train_dataset.near * 1.5)
            self.end = torch.tensor(cfg.end if 'end' in cfg else kwargs['system'].dm.train_dataset.far * 1.5)
        else:
            self.initial = torch.tensor(cfg.initial if 'initial' in cfg else 0.0)
            self.end = torch.tensor(cfg.end if 'end' in cfg else 1.0)

        # Contract
        if self.contract_fn.contract_samples:
            self.initial = self.contract_fn.contract_distance(self.initial)
            self.end = self.contract_fn.contract_distance(self.end)

        # Origin scale
        self.origin_scale_factor = cfg.origin_scale_factor if 'origin_scale_factor' in cfg else 0.0
        self.origin_initial = torch.tensor(cfg.origin_initial if 'origin_initial' in cfg else [1.0, 1.0, 1.0]).cuda()

        # Flip axes
        self.flip_axes = cfg.flip_axes if 'flip_axes' in cfg else False

        # Calculate samples
        self.samples = torch.linspace(
            self.initial, self.end, z_channels, device="cuda"
        ).view(-1, 1)

        # Calculate z scale
        if z_channels > 1:
            self.z_scale = (
                cfg.z_scale
                if "z_scale" in cfg
                else torch.abs(self.samples[1] - self.samples[0])
            )
        else:
            self.z_scale = cfg.z_scale if "z_scale" in cfg else 1.0

        self.z_scale = torch.tensor(self.z_scale).view(-1, 1).cuda()

        # Unify positions
        self.unify_fn = pluecker_pos(None)

    def process_origins(self, origins):
        origins = origins * self.origin_scale_factor + self.origin_initial.view(1, 1, 3)
        return origins

    def process_z_vals(self, z_vals):
        z_vals = z_vals.view(z_vals.shape[0], -1, 4)
        origins = self.process_origins(z_vals[..., :3])
        radii = super().process_z_vals(z_vals[..., -1])
        return torch.cat([origins, radii[..., None]], -1).view(z_vals.shape[0], -1)

    def intersect(self, rays, z_vals):
        z_vals = z_vals.view(z_vals.shape[0], self.z_channels, 4)
        origins = z_vals[..., :3]
        radii = z_vals[..., -1]

        rays = torch.cat(
            [
                rays[..., None, 0:3] * origins,
                rays[..., None, 3:6] * origins,
            ],
            -1
        )

        # Calculate intersection
        return intersect_sphere(
            rays,
            torch.zeros_like(origins),
            radii,
        )


class IntersectSphereNew(Intersect):
    def __init__(self, z_channels, cfg, **kwargs):
        super().__init__(z_channels, cfg, **kwargs)

        # Intersect hyper-params
        if self.use_dataset_bounds:
            if cfg.outward_facing:
                self.initial = torch.tensor(cfg.initial if 'initial' in cfg else kwargs['system'].dm.train_dataset.near * 1.5)
            else:
                self.initial = torch.tensor(cfg.initial if 'initial' in cfg else -kwargs['system'].dm.train_dataset.far * 1.5)

            self.end = torch.tensor(cfg.end if 'end' in cfg else kwargs['system'].dm.train_dataset.far * 1.5)
        else:
            self.initial = torch.tensor(cfg.initial if 'initial' in cfg else 0.0)
            self.end = torch.tensor(cfg.end if 'end' in cfg else 1.0)

        # Contract
        if self.contract_fn.contract_samples:
            self.initial = self.contract_fn.contract_distance(self.initial)
            self.end = self.contract_fn.contract_distance(self.end)

        # Origin scale
        self.resize_scale_factor = cfg.resize_scale_factor if 'resize_scale_factor' in cfg else 0.0
        self.resize_initial = torch.tensor(cfg.resize_initial if 'resize_initial' in cfg else [1.0, 1.0, 1.0]).cuda()
        self.origin_scale_factor = cfg.origin_scale_factor if 'origin_scale_factor' in cfg else 0.0

        # Flip axes
        self.flip_axes = cfg.flip_axes if 'flip_axes' in cfg else False

        # Calculate samples
        self.samples = torch.linspace(
            self.initial, self.end, z_channels, device="cuda"
        ).view(-1, 1)

        # Calculate z scale
        if z_channels > 1:
            self.z_scale = (
                cfg.z_scale
                if "z_scale" in cfg
                else torch.abs(self.samples[1] - self.samples[0])
            )
        else:
            self.z_scale = cfg.z_scale if "z_scale" in cfg else 1.0

        self.z_scale = torch.tensor(self.z_scale).view(-1, 1).cuda()

        # Unify positions
        self.unify_fn = pluecker_pos(None)

    def process_origins(self, origins):
        origins = origins * self.origin_scale_factor
        return origins

    def process_resize(self, resize):
        resize = resize * self.resize_scale_factor + self.resize_initial.view(1, 1, 3)
        return resize

    def process_z_vals(self, z_vals):
        z_vals = z_vals.view(z_vals.shape[0], -1, 8)
        origins = self.process_origins(z_vals[..., :3])
        resize = self.process_resize(z_vals[..., 3:6])
        raw_offsets = super().process_z_vals(z_vals[..., -2])
        radii = super().process_z_vals(z_vals[..., -1])
        return torch.cat([origins, resize, raw_offsets[..., None], radii[..., None]], -1).view(z_vals.shape[0], -1)

    def intersect(self, rays, z_vals):
        z_vals = z_vals.view(z_vals.shape[0], self.z_channels, 8)
        origins = z_vals[..., :3]
        resize = z_vals[..., 3:6]
        raw_offsets = z_vals[..., -2]
        radii = z_vals[..., -1]

        # Transform the space
        rays_o = (rays[..., None, 0:3] - origins) * resize
        rays_d = rays[..., None, 3:6] * resize
        rays = torch.cat(
            [
                rays_o,
                torch.nn.functional.normalize(rays_d, p=2.0, dim=-1),
            ],
            -1
        )

        # Calculate intersection distances (in transformed space)
        t = intersect_sphere(
            rays,
            torch.zeros_like(origins),
            radii,
        )

        # Recycle samples for not-hit spheres
        min_radius = min_sphere_radius(rays, torch.zeros_like(rays[..., :3]))

        base_pos = self.unify_fn(rays)
        diff = (base_pos - rays[..., :3])
        base_distance = torch.sign(dot(rays[..., 3:6], diff)) * torch.norm(diff, dim=-1)

        t = torch.where(
            torch.abs(radii) < min_radius + 4 * self.z_scale,
            raw_offsets + base_distance,
            t
        )

        # Transform distances
        return t / (torch.norm(rays_d, dim=-1) + 1e-5)


primitive_intersect_dict = {
    "euclidean_distance": IntersectEuclideanDistance,
    "euclidean_distance_unified": IntersectEuclideanDistanceUnified,
    "plane": IntersectPlane,
    "cylinder": IntersectCylinderOld,
    "cylinder_new": IntersectCylinderNew,
    "sphere": IntersectSphereOld,
    "sphere_new": IntersectSphereNew,
}
