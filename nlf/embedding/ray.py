#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch

from typing import Dict
from torch import nn
from nlf.activations import get_activation
from nlf.nets import net_dict
from nlf.pe import IdentityPE, pe_dict

from nlf.param import (
    RayParam
)

from nlf.intersect import (
    intersect_dict,
)

from utils.intersect_utils import intersect_axis_plane

import pytorch3d.transforms as transforms


class CalibratePlanarEmbedding(nn.Module):
    def __init__(
        self,
        in_channels,
        cfg,
        **kwargs
    ):
        super().__init__()

        self.cfg = cfg
        self.opt_group = cfg.group if 'group' in cfg else (kwargs['group'] if 'group' in kwargs else 'calibration')

        # Rays
        self.rays_name = cfg.rays_name if 'rays_name' in cfg else 'rays'

        # Offset
        self.offset = torch.nn.Parameter(
            torch.tensor([[0.0, 0.0]]).float().cuda(), requires_grad=True
        )

        self.activation = get_activation(
            cfg.activation if 'activation' in cfg else 'identity'
        )


    def forward(self, x: Dict[str, torch.Tensor], render_kwargs: Dict[str, str]):
        # Get rays
        rays = x[self.rays_name]
        rays_o = rays[..., 0:3]
        rays_d = rays[..., 3:6]

        # Intersect
        t = intersect_axis_plane(
            rays,
            0.0,
            -1
        )
        rays_o2 = rays_o + t.unsqueeze(-1) * rays_d

        # Add offset
        offset = self.activation(self.offset)

        rays_o = torch.cat(
            [
                rays_o[..., :2] + offset,
                rays_o[..., 2:],
            ],
            dim=-1
        )

        rays_d = torch.nn.functional.normalize(rays_o2 - rays_o, dim=-1)

        rays = torch.cat([rays_o, rays_d], dim=-1)
        print("Offset:", offset)

        # Return
        x[self.rays_name] = rays
        return x

    def set_iter(self, i):
        self.cur_iter = i


class CalibrateEmbedding(nn.Module):
    def __init__(
        self,
        in_channels,
        cfg,
        **kwargs
    ):
        super().__init__()

        self.cfg = cfg
        self.opt_group = cfg.group if 'group' in cfg else (kwargs['group'] if 'group' in kwargs else 'calibration')

        # Rays
        self.rays_name = cfg.rays_name if 'rays_name' in cfg else 'rays'

        # Pose parameters
        self.use_pose = cfg.use_pose if 'use_pose' in cfg else False

        if self.use_pose:
            self.num_views = kwargs['system'].dm.train_dataset.total_num_views
            self.datasets = [kwargs['system'].dm.val_dataset] 
            self.constant_id = cfg.constant_id if 'constant_id' in cfg else 0

            self.base_quaternions = torch.zeros((self.num_views, 4), dtype=torch.float32, device='cuda')
            self.base_quaternions[:, 0] = 1.0

            self.quaternions = torch.nn.Parameter(
                torch.zeros((self.num_views, 4), dtype=torch.float32, device='cuda'), requires_grad=True
            )
            self.translations = torch.nn.Parameter(
                torch.zeros((self.num_views, 3), dtype=torch.float32, device='cuda'), requires_grad=True
            )

            self.quaternion_activation = get_activation(
                cfg.quaternion_activation if 'quaternion_activation' in cfg else 'identity'
            )
            self.translation_activation = get_activation(
                cfg.translation_activation if 'translation_activation' in cfg else 'identity'
            )

        # Time
        self.use_time = cfg.use_time if 'use_time' in cfg else False

        if self.use_time:
            self.num_frames = kwargs['system'].dm.train_dataset.num_frames

            self.time_offsets = torch.nn.Parameter(
                torch.zeros((self.num_views, 1), dtype=torch.float32, device='cuda'), requires_grad=True
            )
            self.time_activation = get_activation(
                cfg.time_activation if 'time_activation' in cfg else 'identity'
            )

        # NDC
        self.use_ndc = cfg.use_ndc if 'use_ndc' in cfg else False


    def forward(self, x: Dict[str, torch.Tensor], render_kwargs: Dict[str, str]):
        # Get rays
        rays = x[self.rays_name]
        rays_o = rays[..., 0:3]
        rays_d = rays[..., 3:6]

        # Get camera IDs
        if rays.shape[-1] > 7:
            camera_ids = torch.round(rays[..., -2]).long()
        else:
            camera_ids = torch.round(rays[..., -1]).long()

        if self.use_pose:
            quaternion_offsets = self.quaternion_activation(self.quaternions)
            quaternion_offsets[self.constant_id] = 0
            quaternions = self.base_quaternions + quaternion_offsets
            #quaternions = torch.nn.functional.normalize(quaternions[camera_ids].view(-1, 4), -1)
            quaternions = quaternions[camera_ids].view(-1, 4)

            translation_offsets = self.translation_activation(self.translations)
            translation_offsets[self.constant_id] = 0
            translations = translation_offsets[camera_ids].view(-1, 3)

            rays_d = transforms.quaternion_apply(quaternions, rays_d)
            rays_o = translations + rays_o

        if self.use_time:
            time_offsets = self.time_activation(self.time_offsets)
            time_offsets[self.constant_id] = 0.0
            time_offsets = time_offsets[camera_ids].view(-1, 1)

            rays_t = rays[..., -1:]
            print(self.time_offsets[12], self.time_offsets[1])
            rays_t = rays_t + time_offsets

        # Update rays
        if self.use_pose:
            updated_rays = torch.cat([rays_o, rays_d], dim=-1)
        else:
            updated_rays = rays[..., :6]

        # Apply NDC
        if self.use_ndc:
            updated_rays = self.datasets[0].to_ndc(updated_rays)

        # Update times
        if self.use_time:
            rays = torch.cat([updated_rays, rays[..., 6:-1], rays_t], dim=-1)
        else:
            rays = torch.cat([updated_rays, rays[..., 6:]], dim=-1)

        x[self.rays_name] = rays
        return x

    def set_iter(self, i):
        self.cur_iter = i

        if self.use_pose:
            self.quaternion_activation.set_iter(i)
            self.translation_activation.set_iter(i)
        
        if self.use_time:
            self.time_activation.set_iter(i)


class RayPredictionEmbedding(nn.Module):
    def __init__(
        self,
        in_channels,
        cfg,
        **kwargs
    ):
        super().__init__()

        self.group = cfg.group if 'group' in cfg else (kwargs['group'] if 'group' in kwargs else 'embedding')
        self.cfg = cfg

        # Rays
        self.rays_name = cfg.rays_name if 'rays_name' in cfg else 'rays'

        # Ray parameterization and positional encoding
        self.param_names = list(cfg.params.keys())
        self.params = nn.ModuleList()
        self.pes = nn.ModuleList()
        self.in_channels = 0
        self.param_channels = []

        for param_key in cfg.params.keys():
            param_cfg = cfg.params[param_key]

            # Start, end channels
            self.param_channels.append(
                (param_cfg.start, param_cfg.end)
            )
            in_channels = param_cfg.end - param_cfg.start

            # Create param
            if 'in_channels' not in param_cfg.param:
                param_cfg.param.in_channels = in_channels

            param = RayParam(param_cfg.param, system=kwargs['system'])
            self.params.append(param)

            # Create PE
            if 'pe' in param_cfg:
                pe = pe_dict[param_cfg.pe.type](
                    param.out_channels,
                    param_cfg.pe
                )
            else:
                pe = IdentityPE(param.out_channels)

            self.pes.append(pe)

            # Update in channels
            self.in_channels += pe.out_channels

        # Intersect
        self.z_channels = cfg.z_channels

        # Outputs
        self.outputs = cfg.outputs
        self.output_names = list(self.outputs.keys())
        self.output_shapes = [self.outputs[k].channels for k in self.outputs.keys()]
        self.preds_per_z = sum(self.output_shapes)

        self.ray_outputs = cfg.ray_outputs if 'ray_outputs' in cfg else {}
        self.ray_output_names = list(self.ray_outputs.keys())
        self.ray_output_shapes = [self.ray_outputs[k].channels for k in self.ray_outputs.keys()]

        self.total_ray_out_channels = sum(self.ray_output_shapes)
        self.total_point_out_channels = self.z_channels * self.preds_per_z
        self.total_out_channels = self.total_point_out_channels + self.total_ray_out_channels

        # Net
        if 'depth' in cfg.net:
            cfg.net['depth'] -= 2
            cfg.net['linear_last'] = False

        self.net = net_dict[cfg.net.type](
            self.in_channels,
            self.total_out_channels,
            cfg.net,
            group=self.group
        )

        # Activations
        self.activations = nn.ModuleList()

        for output_key in self.outputs.keys():
            output_cfg = self.outputs[output_key]

            if 'activation' in output_cfg:
                self.activations.append(get_activation(output_cfg.activation))
            else:
                self.activations.append(get_activation('identity'))

        # Ray activations
        self.ray_activations = nn.ModuleList()

        for output_key in self.ray_outputs.keys():
            output_cfg = self.ray_outputs[output_key]

            if 'activation' in output_cfg:
                self.ray_activations.append(get_activation(output_cfg.activation))
            else:
                self.ray_activations.append(get_activation('identity'))

    def forward(self, x: Dict[str, torch.Tensor], render_kwargs: Dict[str, str]):
        rays = x[self.rays_name]

        # Apply parameterization
        param_x = []

        for idx, (param, pe) in enumerate(zip(self.params, self.pes)):
            cur_x = rays[:, self.param_channels[idx][0]:self.param_channels[idx][1]]
            param_x.append(pe(param(cur_x)))

        param_x = torch.cat(param_x, -1)

        # Get outputs
        outputs_flat = self.net(param_x)

        # Get point outputs
        if self.total_point_out_channels > 0:
            point_outputs = outputs_flat[..., :self.total_point_out_channels].reshape(rays.shape[0], self.z_channels, -1)
            point_outputs = torch.split(point_outputs, self.output_shapes, -1)

            for idx, activation in enumerate(self.activations):
                x[self.output_names[idx]] = activation(point_outputs[idx])
        
        # Get ray outputs
        if self.total_ray_out_channels > 0:
            ray_outputs = outputs_flat[..., self.total_point_out_channels:]
            ray_outputs = torch.split(ray_outputs, self.ray_output_shapes, -1)

            for idx, activation in enumerate(self.ray_activations):
                x[self.ray_output_names[idx]] = activation(ray_outputs[idx])

        return x

    def set_iter(self, i):
        self.cur_iter = i
        self.net.set_iter(i)

        for act in self.activations:
            if getattr(act, "set_iter", None) is not None:
                act.set_iter(i)

        for act in self.ray_activations:
            if getattr(act, "set_iter", None) is not None:
                act.set_iter(i)

        for pe in self.pes:
            if getattr(pe, "set_iter", None) is not None:
                pe.set_iter(i)


class RayIntersectEmbedding(nn.Module):
    def __init__(
        self,
        in_channels,
        cfg,
        **kwargs
    ):
        super().__init__()

        self.group = cfg.group if 'group' in cfg else (kwargs['group'] if 'group' in kwargs else 'embedding')
        self.cfg = cfg

        # Rays
        self.rays_name = cfg.rays_name if 'rays_name' in cfg else 'rays'

        # Intersect
        self.z_channels = cfg.z_channels
        self.intersect_fn = intersect_dict[cfg.intersect.type](
            self.z_channels, cfg.intersect, **kwargs
        )


    def forward(self, x: Dict[str, torch.Tensor], render_kwargs: Dict[str, str]):
        rays = x[self.rays_name]
        return self.intersect_fn(rays, x, render_kwargs)

    def set_iter(self, i):
        self.cur_iter = i
        self.intersect_fn.set_iter(i)


class CreateRaysEmbedding(nn.Module):
    def __init__(
        self,
        in_channels,
        cfg,
        **kwargs
    ):
        super().__init__()

        self.group = cfg.group if 'group' in cfg else (kwargs['group'] if 'group' in kwargs else 'embedding')
        self.cfg = cfg

        # Rays
        self.in_rays_name = cfg.in_rays_name if 'in_rays_name' in cfg else 'rays'
        self.in_points_name = cfg.in_points_name if 'in_points_name' in cfg else 'points'
        self.out_rays_name = cfg.out_rays_name if 'out_rays_name' in cfg else 'rays'

        # Extra outputs
        self.extra_outputs = cfg.extra_outputs

    def forward(self, x: Dict[str, torch.Tensor], render_kwargs: Dict[str, str]):
        rays = x[self.in_rays_name]
        points = x[self.in_points_name]
        x[self.out_rays_name] = torch.cat(
            [
                points,
                rays[..., None, 3:6].repeat(1, points.shape[1], 1),
            ],
            dim=-1
        )
        return x

    def set_iter(self, i):
        self.cur_iter = i


ray_embedding_dict = {
    'calibrate_planar': CalibratePlanarEmbedding,
    'calibrate': CalibrateEmbedding,
    'ray_prediction': RayPredictionEmbedding,
    'ray_intersect': RayIntersectEmbedding,
    'create_rays': CreateRaysEmbedding,
}
