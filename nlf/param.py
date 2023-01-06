#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch

from torch import nn
from utils.intersect_utils import intersect_cylinder

from utils.intersect_utils import (
    intersect_axis_plane,
    intersect_sphere,
)

from .contract import contract_dict

def identity(cfg, **kwargs):
    def param(x):
        return x

    return param


def take(cfg, **kwargs):
    input_channels = cfg.input_channels

    def param(x):
        return x[..., input_channels].view(*(x.shape[:-1] + (len(input_channels),)))

    return param


def position(cfg, **kwargs):
    def pos(rays):
        return rays[..., :3]

    return pos


def xy_param(cfg, *kwargs):
    def param(rays):
        rays = rays.view(rays.shape[0], -1, 6)
        rays = torch.cat([rays[..., :2], rays[..., 3:5]], dim=-1)
        return rays.view(rays.shape[0], -1)

    return param


def rays_param(cfg, **kwargs):
    def param(rays):
        rays = rays.view(rays.shape[0], -1, 6)
        rays_o = rays[..., :3]
        rays_d = torch.nn.functional.normalize(rays[..., 3:6] - rays_o, dim=-1)
        rays = torch.cat([rays_o, rays_d], dim=-1)
        return rays.view(rays.shape[0], -1)

    return param


class TwoPlaneParam(nn.Module):
    def __init__(
        self,
        cfg,
        **kwargs
    ):

        super().__init__()

        self.in_channels = cfg.in_channels if 'in_channels' in cfg else 6
        self.out_channels = cfg.n_dims if 'n_dims' in cfg else 4

        self.st_multiplier = cfg.st_multiplier if 'st_multiplier' in cfg else 1.0
        self.uv_multiplier = cfg.uv_multiplier if 'uv_multiplier' in cfg else 1.0

        self.near = cfg.near if 'near' in cfg else -1.0
        self.far = cfg.far if 'far' in cfg else 0.0

        self.origin = torch.tensor(cfg.origin if 'origin' in cfg else [0.0, 0.0, 0.0], device='cuda')

        # Local param
        self.use_local_param = cfg.use_local_param if 'use_local_param' in cfg else False
        self.voxel_size = cfg.voxel_size if 'voxel_size' in cfg else 1.0

    def forward(self, rays):
        # Offset z
        rays_o, rays_d = rays[..., :3] - self.origin.unsqueeze(0), rays[..., 3:6]

        if self.use_local_param:
            z_offset = torch.round(rays_o[..., 2:3] / self.voxel_size) * self.voxel_size
            rays_o = rays_o - z_offset

        rays = torch.cat([rays_o, rays_d], -1)

        # Intersection distances
        t1 = intersect_axis_plane(
            rays, self.near, 2
        )

        t2 = intersect_axis_plane(
            rays, self.far, 2
        )

        # Param rays
        param_rays = torch.cat(
            [
                rays[..., :2] + rays[..., 3:5] * t1.unsqueeze(-1),
                rays[..., :2] + rays[..., 3:5] * t2.unsqueeze(-1),
            ],
            dim=-1
        )

        return param_rays

    def set_iter(self, i):
        self.cur_iter = i


def multi_plane_param(cfg, **kwargs):
    # Intersect hyper-params
    initial_z = cfg.initial_z if 'initial_z' in cfg else -1.0
    end_z = cfg.end_z if 'end_z' in cfg else 1.0
    z_channels = cfg.z_channels if 'z_channels' in cfg else 8
    voxel_size = cfg.voxel_size if 'voxel_size' in cfg else 1.0

    depth_samples = torch.linspace(
        initial_z,
        end_z,
        z_channels,
        device='cuda'
    ) * voxel_size

    def param(rays):
        t = intersect_axis_plane(
            rays[..., None, :], depth_samples[None], 2,
            exclude=False
        )

        param_rays = rays[..., None, :3] + rays[..., None, 3:6] * t.unsqueeze(-1)

        return param_rays.view(rays.shape[0], -1)

    return param


def calc_scale(r):
    return 1.0 / torch.sqrt(((-r + 1) * (-r + 1) + r * r) + 1e-8)


def two_plane_matrix(cfg, **kwargs):
    global_near = cfg.global_near if 'global_near' in cfg else -1.0
    near = (cfg.near if 'near' in cfg else 0.0) * cfg.voxel_size
    far = (cfg.far if 'far' in cfg else 1.0) * cfg.voxel_size

    def param(rays):
        # Get near, far zs
        start_z = rays[..., 2]
        near_z = near + start_z
        far_z = far + start_z

        # Intersect
        isect_pts_1, _ = intersect_axis_plane(
            rays, near_z, 2, exclude=False
        )

        isect_pts_2, _ = intersect_axis_plane(
            rays, far_z, 2, exclude=False
        )

        # Scale factors
        near_scale = calc_scale((near_z - global_near))
        far_scale = calc_scale((far_z - global_near))

        param_rays = torch.cat(
            [
                isect_pts_1[..., :2] * near_scale[..., None],
                isect_pts_1[..., -1:],
                isect_pts_2[..., :2] * far_scale[..., None],
                isect_pts_2[..., -1:],
            ],
            dim=-1
        )

        return param_rays

    return param


def two_plane_pos(cfg, **kwargs):
    near = cfg.near if 'near' in cfg else -1.0
    far = cfg.far if 'far' in cfg else 0.0

    pre_mult = 1.0
    post_mult = 1.0

    if 'voxel_size' in cfg:
        near = cfg.near if 'near' in cfg else -0.5
        far = cfg.far if 'far' in cfg else 0.5 # noqa

        pre_mult = 1.0 / cfg.voxel_size
        post_mult = cfg.voxel_size

    if 'multiplier' in cfg:
        near = cfg.near if 'near' in cfg else 0.0
        far = cfg.far if 'far' in cfg else 1.0 # noqa

        post_mult = cfg.multiplier

    def pos(rays):
        rays = rays * pre_mult

        isect_pts, _ = intersect_axis_plane(
            rays, near, 2, exclude=False
        )

        return isect_pts * post_mult

    return pos


class PlueckerParam(nn.Module):
    def __init__(
        self,
        cfg,
        **kwargs
    ):

        super().__init__()

        self.in_channels = cfg.in_channels if 'in_channels' in cfg else 6
        self.out_channels = cfg.n_dims if 'n_dims' in cfg else 6

        self.direction_multiplier = cfg.direction_multiplier if 'direction_multiplier' in cfg else 1.0
        self.moment_multiplier = cfg.moment_multiplier if 'moment_multiplier' in cfg else 1.0

        self.origin = torch.tensor(cfg.origin if 'origin' in cfg else [0.0, 0.0, 0.0], device='cuda')

        # Local param
        self.use_local_param = cfg.use_local_param if 'use_local_param' in cfg else False
        self.voxel_size = torch.tensor(cfg.voxel_size if 'voxel_size' in cfg else [1.0, 1.0, 1.0]).cuda()

    def forward(self, rays):
        rays_o, rays_d = rays[..., :3] - self.origin.unsqueeze(0), rays[..., 3:6]
        rays_d = torch.nn.functional.normalize(rays_d, p=2.0, dim=-1)

        if self.use_local_param:
            origin = torch.round(rays_o / self.voxel_size.unsqueeze(0)) * self.voxel_size.unsqueeze(0)
            rays_o = rays_o - origin

        m = torch.cross(rays_o, rays_d, dim=-1)
        return torch.cat([rays_d * self.direction_multiplier, m * self.moment_multiplier], dim=-1)

    def set_iter(self, i):
        self.cur_iter = i


class ContractPointsParam(nn.Module):
    def __init__(
        self,
        cfg,
        **kwargs
    ):

        super().__init__()

        self.param = ray_param_dict[cfg.param.fn](cfg.param)

        self.in_channels = self.param.in_channels
        self.out_channels = self.param.out_channels

        self.contract_fn = contract_dict[cfg.contract.type](
            cfg.contract,
            **kwargs
        )
        self.contract_start_channel = cfg.contract_start_channel
        self.contract_end_channel = cfg.contract_end_channel

    def forward(self, rays):
        param_rays = self.param(rays)

        return torch.cat(
            [
                param_rays[..., :self.contract_start_channel],
                self.contract_fn.contract_points(param_rays[..., self.contract_start_channel:self.contract_end_channel]),
                param_rays[..., self.contract_end_channel:],
            ],
            -1
        )

    def set_iter(self, i):
        self.cur_iter = i
        self.param.set_iter(i)


def pluecker_pos(cfg):
    def pos(rays):
        rays_o, rays_d = rays[..., :3], rays[..., 3:6]
        rays_d = torch.nn.functional.normalize(rays_d, p=2.0, dim=-1)

        m = torch.cross(rays_o, rays_d, dim=-1)
        rays_o = torch.cross(rays_d, m, dim=-1)

        return rays_o

    return pos


def pluecker_pos_cylinder(cfg):
    def pos(rays):
        rays_o, rays_d = rays[..., :3], rays[..., 3:6]
        rays_o = torch.cat([rays_o[..., 0:1], torch.zeros_like(rays[..., 1:2]), rays_o[..., 2:3]], -1)
        rays_d = torch.cat([rays_d[..., 0:1], torch.zeros_like(rays[..., 1:2]), rays_d[..., 2:3]], -1)
        rays_d = torch.nn.functional.normalize(rays_d, p=2.0, dim=-1)

        m = torch.cross(rays_o, rays_d, dim=-1)
        rays_o = torch.cross(rays_d, m, dim=-1)

        return rays_o

    return pos


def spherical_param(cfg, **kwargs):
    def param(rays):
        isect_pts = intersect_sphere(
            rays,
            cfg.radius
        ) / cfg.radius

        return torch.cat([isect_pts, rays[..., 3:6]], dim=-1)

    return param


def two_cylinder_param(cfg, **kwargs):
    origin = torch.tensor(cfg.origin if 'origin' in cfg else [0.0, 0.0, 0.0], device='cuda')
    near = (cfg.near if 'near' in cfg else 1.0)
    far = (cfg.far if 'far' in cfg else 2.0)

    def param(rays):
        isect_pts_1, _ = intersect_cylinder(
            rays,
            origin[None],
            near,
            sort=False
        )
        isect_pts_2, _ = intersect_cylinder(
            rays,
            origin[None],
            far,
            sort=False
        )
        param_rays = torch.cat([isect_pts_1, isect_pts_2], dim=-1)

        return param_rays

    return param


time_param_dict = {
    'identity': identity
}


class RayPlusTime(nn.Module):
    def __init__(
        self,
        cfg,
        **kwargs
    ):

        super().__init__()

        self.group = cfg.group if 'group' in cfg else (kwargs['group'] if 'group' in kwargs else 'embedding')
        self.ray_param_fn = ray_param_dict[cfg.ray_param.fn](cfg.ray_param)
        self.time_param_fn = time_param_dict[cfg.time_param.fn](cfg.time_param)

        self.in_channels = cfg.in_channels if 'in_channels' in cfg else 7
        self.out_channels = cfg.n_dims if 'n_dims' in cfg else self.in_channels

        self.dummy_layer = nn.Linear(1, 1)

    def forward(self, x):
        param_rays = self.ray_param_fn(x[..., :-1])
        param_times = self.time_param_fn(x[..., -1:])
        return torch.cat([param_rays, param_times], dim=-1)

    def set_iter(self, i):
        self.cur_iter = i


class VoxelCenterParam(nn.Module):
    def __init__(
        self,
        cfg,
        **kwargs
    ):

        super().__init__()

        self.in_channels = cfg.in_channels if 'in_channels' in cfg else 3
        self.out_channels = cfg.n_dims if 'n_dims' in cfg else 3
        self.origin = torch.tensor(cfg.origin if 'origin' in cfg else [0.0, 0.0, 0.0], device='cuda')
        self.voxel_size = torch.tensor(cfg.voxel_size if 'voxel_size' in cfg else [1.0, 1.0, 1.0]).cuda()

    def forward(self, x):
        x = x - self.origin.unsqueeze(0)
        origin = torch.round(x / self.voxel_size.unsqueeze(0)) * self.voxel_size.unsqueeze(0)
        return origin

    def set_iter(self, i):
        self.cur_iter = i


class ZSliceParam(nn.Module):
    def __init__(
        self,
        cfg,
        **kwargs
    ):

        super().__init__()

        self.in_channels = cfg.in_channels if 'in_channels' in cfg else 1
        self.out_channels = cfg.n_dims if 'n_dims' in cfg else 1
        self.voxel_size = cfg.voxel_size if 'voxel_size' in cfg else 1.0

    def forward(self, x):
        return torch.round(x / self.voxel_size) * self.voxel_size

    def set_iter(self, i):
        self.cur_iter = i



ray_param_dict = {
    'identity': identity,
    'take': take,
    'pluecker': PlueckerParam,
    'position': position,
    'spherical': spherical_param,
    'xy': xy_param,
    'rays': rays_param,
    'two_plane': TwoPlaneParam,
    'multi_plane': multi_plane_param,
    'two_plane_matrix': two_plane_matrix,
    'two_cylinder': two_cylinder_param,
    'ray_plus_time': RayPlusTime,
    'voxel_center': VoxelCenterParam,
    'z_slice': ZSliceParam,
    'contract_points': ContractPointsParam,
}


ray_param_pos_dict = {
    'pluecker': pluecker_pos,
    'two_plane': two_plane_pos,
}


class RayParam(nn.Module):
    def __init__(
        self,
        cfg,
        **kwargs
    ):

        super().__init__()

        self.group = cfg.group if 'group' in cfg else (kwargs['group'] if 'group' in kwargs else 'embedding')

        self.ray_param_fn = ray_param_dict[cfg.fn](cfg, **kwargs)
        self.in_channels = cfg.in_channels if 'in_channels' in cfg else 6
        self.out_channels = cfg.n_dims if 'n_dims' in cfg else self.in_channels

        self.dummy_layer = nn.Linear(1, 1)

    def forward(self, x):
        return self.ray_param_fn(x)

    def set_iter(self, i):
        self.cur_iter = i
