#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch

from typing import Dict, List
from torch import nn
from nlf.nets import net_dict
from nlf.pe import IdentityPE, pe_dict

from nlf.param import (
    RayParam
)

from nlf.contract import contract_dict

from nlf.activations import get_activation
from utils.intersect_utils import (
    sort_z
)
from utils.flow_utils import (
    get_base_time
)
from utils.ray_utils import (
    dot,
    from_ndc,
    reflect,
    get_ray_density
)
from utils.rotation_conversions import (
    axis_angle_to_matrix
)


class PointPredictionEmbedding(nn.Module):
    def __init__(
        self,
        in_channels,
        cfg,
        **kwargs
    ):
        super().__init__()

        self.group = cfg.group if 'group' in cfg else (kwargs['group'] if 'group' in kwargs else 'embedding')
        self.cfg = cfg

        # Filtering
        self.filter = cfg.filter if 'filter' in cfg else False

        # Rays & points
        self.rays_name = cfg.rays_name if 'rays_name' in cfg else 'rays'
        self.points_name = cfg.points_name if 'points_name' in cfg else 'points'

        # Inputs
        self.in_z_channels = cfg.in_z_channels if 'in_z_channels' in cfg else 1
        self.inputs = cfg.inputs
        self.input_names = list(self.inputs.keys())
        self.input_shapes = [self.inputs[k] for k in self.inputs.keys()]

        # Ray parameterization and positional encoding
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

            param = RayParam(param_cfg.param)
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

        #self.total_in_channels = self.in_channels * self.in_z_channels
        self.total_in_channels = self.in_channels

        # Outputs
        self.out_z_channels = cfg.out_z_channels if 'out_z_channels' in cfg else 1
        self.outputs = cfg.outputs
        self.output_names = list(self.outputs.keys())
        self.output_shapes = [self.outputs[k].channels for k in self.outputs.keys()]
        self.output_residual = [
            self.outputs[k].residual if 'residual' in self.outputs[k] else False \
            for k in self.outputs.keys()
        ]
        self.out_channels = sum(self.output_shapes)
        self.total_out_channels = self.out_channels * self.out_z_channels
        self.out_z_per_in_z = self.out_z_channels // self.in_z_channels

        # Net
        if 'depth' in cfg.net:
            cfg.net['depth'] -= 2
            cfg.net['linear_last'] = False

        self.net = net_dict[cfg.net.type](
            self.total_in_channels,
            #self.in_channels,
            #self.total_out_channels,
            self.out_channels * self.out_z_per_in_z,
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

    def forward(self, x: Dict[str, torch.Tensor], render_kwargs: Dict[str, str]):
        rays = x[self.rays_name]
        points = x[self.points_name]

        # Get inputs
        inputs = []

        for inp_idx, inp_name in enumerate(self.input_names):
            if inp_name == 'viewdirs':
                inputs.append(rays[..., None, 3:6].repeat(1, points.shape[1], 1))
            elif inp_name == 'origins':
                inputs.append(rays[..., None, 0:3].repeat(1, points.shape[1], 1))
            elif inp_name == 'times':
                inputs.append(rays[..., None, -1:].repeat(1, points.shape[1], 1))
            else:
                inputs.append(x[inp_name][..., :self.input_shapes[inp_idx]])

        inputs = torch.cat(inputs, -1)

        # Apply parameterization
        param_inputs = []

        for idx in range(len(self.params)):
            cur_input = inputs[..., self.param_channels[idx][0]:self.param_channels[idx][1]]
            param_inputs.append(
                self.pes[idx](
                    self.params[idx](cur_input)
                )
            )

        inputs = torch.cat(param_inputs, -1).view(-1, self.total_in_channels)

        if self.filter:
            # Run on valid
            valid_mask = x['distances'].view(-1) > 0.0
            outputs_flat = inputs.new_zeros(inputs.shape[0], self.total_out_channels)
            outputs_flat_valid = self.net(inputs[valid_mask])
            outputs_flat[valid_mask] = outputs_flat_valid
        else:
            # Run on all
            outputs_flat = self.net(inputs)

        # Get outputs
        outputs_flat = outputs_flat.view(points.shape[0], -1, self.out_channels)
        outputs_flat = torch.split(outputs_flat, self.output_shapes, -1)

        for i in range(len(self.output_shapes)):
            cur_output = self.activations[i](outputs_flat[i])

            if self.output_residual[i]:
                last_output = x[self.output_names[i]].view(
                    cur_output.shape[0], -1, 1, cur_output.shape[-1]
                )
                cur_output_shape = cur_output.shape
                cur_output = cur_output.view(
                    cur_output.shape[0],
                    last_output.shape[1],
                    -1,
                    cur_output.shape[-1]
                ) + last_output
                #) + torch.mean(last_output, dim=-2, keepdim=True)
                cur_output = cur_output.view(*cur_output_shape)

            x[self.output_names[i]] = cur_output

        return x

    def set_iter(self, i):
        self.cur_iter = i
        self.net.set_iter(i)

        for act in self.activations:
            if getattr(act, "set_iter", None) is not None:
                act.set_iter(i)

        for pe in self.pes:
            if getattr(pe, "set_iter", None) is not None:
                pe.set_iter(i)


class ExtractFieldsEmbedding(nn.Module):
    fields: List[str]

    def __init__(
        self,
        in_channels,
        cfg,
        **kwargs
    ):
        super().__init__()

        self.group = cfg.group if 'group' in cfg else (kwargs['group'] if 'group' in kwargs else 'embedding')
        self.cfg = cfg
        self.fields = list(cfg.fields)

    def forward(self, x: Dict[str, torch.Tensor], render_kwargs: Dict[str, str]):
        fields = self.fields + list(render_kwargs.get('fields', []))
        outputs = {}

        for field in fields:
            if field in x:
                outputs[field] = x[field]

        return outputs

    def set_iter(self, i):
        self.cur_iter = i


class CreatePointsEmbedding(nn.Module):
    def __init__(
        self,
        in_channels,
        cfg,
        **kwargs
    ):
        super().__init__()

        self.group = cfg.group if 'group' in cfg else (kwargs['group'] if 'group' in kwargs else 'embedding')
        self.cfg = cfg

        # Rays & points
        self.rays_name = cfg.rays_name if 'rays_name' in cfg else 'rays'
        self.out_points_field = cfg.out_points_field if 'out_points_field' in cfg else 'points'

        # Activation
        self.activation = get_activation(cfg.activation if 'activation' in cfg else 'sigmoid')

    def forward(self, x: Dict[str, torch.Tensor], render_kwargs: Dict[str, str]):
        rays = x[self.rays_name]
        dists = x['distances']

        points = rays[..., None, 0:3] + rays[..., None, 3:6] * dists.unsqueeze(-1)
        x[self.out_points_field] = points

        return x

    def set_iter(self, i):
        self.cur_iter = i


class PointDensityEmbedding(nn.Module):
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

        # In and out field
        self.in_field = cfg.in_field if 'in_field' in cfg else 'sigma'
        self.out_field = cfg.out_field if 'out_field' in cfg else 'sigma'

        # Activation
        self.activation = get_activation(cfg.activation if 'activation' in cfg else 'sigmoid')

        self.shift = cfg.shift if 'shift' in cfg else 0
        self.shift_range = cfg.shift_range if 'shift_range' in cfg else 0

        self.window_start_iters = cfg.window_start_iters if 'window_start_iters' in cfg else 0
        self.window_iters = cfg.window_iters if 'window_iters' in cfg else 0

    def window(self):
        cur_iter = self.cur_iter - self.window_start_iters

        if cur_iter < 0:
            return 0.0
        elif cur_iter >= self.window_iters:
            return 1.0
        else:
            return cur_iter / self.window_iters

    def get_sigma(self, z_vals, rays):
        sigma = self.activation(
            z_vals[..., -1:] + self.shift
        )
        return sigma

    def forward(self, x: Dict[str, torch.Tensor], render_kwargs: Dict[str, str]):
        rays = x[self.rays_name]
        window_fac = self.window()
        x[self.out_field] = self.get_sigma(
            x[self.in_field], rays
        ) * window_fac + (1 - window_fac)
        return x

    def set_iter(self, i):
        self.cur_iter = i


class PointOffsetEmbedding(nn.Module):
    def __init__(
        self,
        in_channels,
        cfg,
        **kwargs
    ):
        super().__init__()

        self.group = cfg.group if 'group' in cfg else (kwargs['group'] if 'group' in kwargs else 'embedding')
        self.cfg = cfg

        # In & out fields
        self.in_density_field = cfg.in_density_field if 'in_density_field' in cfg else 'sigma'

        self.in_offset_field = cfg.in_offset_field if 'in_offset_field' in cfg else 'point_offset'
        self.out_offset_field = cfg.out_offset_field if 'out_offset_field' in cfg else 'offset'

        self.in_points_field = cfg.in_points_field if 'in_points_field' in cfg else 'points'
        self.out_points_field = cfg.out_points_field if 'out_points_field' in cfg else 'points'
        self.save_points_field = cfg.save_points_field if 'save_points_field' in cfg else None

        # Point offset
        self.use_sigma = cfg.use_sigma if 'use_sigma' in cfg else True
        self.activation = get_activation(
            cfg.activation if 'activation' in cfg else 'identity'
        )

        # Dropout params
        self.use_dropout = 'dropout' in cfg
        self.dropout_frequency = cfg.dropout.frequency if 'dropout' in cfg else 2
        self.dropout_stop_iter = cfg.dropout.stop_iter if 'dropout' in cfg else float("inf")

    def forward(self, x: Dict[str, torch.Tensor], render_kwargs: Dict[str, str]):
        in_points = x[self.in_points_field]

        if self.save_points_field is not None:
            x[self.save_points_field] = in_points

        # Get point offset
        if self.use_sigma and self.in_density_field in x:
            sigma = x[self.in_density_field]
        else:
            sigma = torch.zeros(in_points.shape[0], in_points.shape[1], 1, device=in_points.device)

        point_offset = self.activation(x[self.in_offset_field]) * (1 - sigma)

        # Dropout
        if self.use_dropout and ((self.cur_iter % self.dropout_frequency) == 0) and self.cur_iter < self.dropout_stop_iter and self.training:
            point_offset = torch.zeros_like(point_offset)

        # Apply offset
        x[self.in_offset_field] = point_offset
        x[self.out_points_field] = x[self.in_points_field] + point_offset

        if self.out_offset_field is not None:
            x[self.out_offset_field] = point_offset

        return x

    def set_iter(self, i):
        self.cur_iter = i


class GenerateNumSamplesEmbedding(nn.Module):
    def __init__(
        self,
        in_channels,
        cfg,
        **kwargs
    ):
        super().__init__()

        self.group = cfg.group if 'group' in cfg else (kwargs['group'] if 'group' in kwargs else 'embedding')
        self.cfg = cfg

        self.sample_range = cfg.sample_range
        self.inference_samples = cfg.inference_samples
        self.total_samples = cfg.total_samples

        self.num_samples_field = cfg.num_samples_field if 'num_samples_field' in cfg else 'num_samples'
        self.total_samples_field = cfg.total_samples_field if 'total_samples_field' in cfg else 'total_samples'

        self.rays_name = cfg.rays_name if 'rays_name' in cfg else 'rays'

    def forward(self, x: Dict[str, torch.Tensor], render_kwargs: Dict[str, str]):
        # Get num samples
        if self.training:
            num_samples = np.random.rand() * (self.sample_range[1] - self.sample_range[0]) \
                + self.sample_range[0]
            num_samples = int(np.round(num_samples))
        else:
            num_samples = self.inference_samples
        
        x[self.num_samples_field] = num_samples
        x[self.total_samples_field] = self.total_samples

        # Add num samples to rays
        rays = x[self.rays_name]

        x[self.rays_name] = torch.cat(
            [
                rays,
                torch.ones_like(rays[..., :1]) * num_samples
            ],
            dim=-1
        )

        return x

    def set_iter(self, i):
        self.cur_iter = i


class SelectPointsEmbedding(nn.Module):
    def __init__(
        self,
        in_channels,
        cfg,
        **kwargs
    ):
        super().__init__()

        self.group = cfg.group if 'group' in cfg else (kwargs['group'] if 'group' in kwargs else 'embedding')
        self.cfg = cfg
        self.fields = cfg.fields

    def forward(self, x: Dict[str, torch.Tensor], render_kwargs: Dict[str, str]):
        total_num_samples = x["total_samples"]
        num_samples = x["num_samples"]

        # Get samples
        samples = np.arange(0, total_num_samples, int(total_num_samples / num_samples))

        # Select samples
        for key in x.keys():
            if key in self.fields:
                x[key] = x[key][:, samples].contiguous()

        return x

    def set_iter(self, i):
        self.cur_iter = i


class RandomOffsetEmbedding(nn.Module):
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
        self.in_points_field = cfg.in_points_field if 'in_points_field' in cfg else 'points'
        self.in_distances_field = cfg.in_distances_field if 'in_distances_field' in cfg else 'distances'

        self.out_points_field = cfg.out_points_field if 'out_points_field' in cfg else 'points'
        self.out_distances_field = cfg.out_distances_field if 'out_distances_field' in cfg else 'distances'

        # Random config
        self.random_per_sample = cfg.random_per_sample if 'random_per_sample' in cfg else 1
        self.frequency = cfg.frequency if 'frequency' in cfg else 2
        self.stop_iter = cfg.stop_iter if 'stop_iter' in cfg else float("inf")

        # NOTE: If used with contract, should use contract point embedding

    def forward(self, x: Dict[str, torch.Tensor], render_kwargs: Dict[str, str]):
        if not self.training or not ((self.cur_iter % self.frequency) == 0) or self.cur_iter >= self.stop_iter:
            return x

        rays = x[self.rays_name]

        points = x[self.in_points_field]
        points = points.view(rays.shape[0], -1, points.shape[-1])

        dists = x[self.in_distances_field]
        dists = dists.view(rays.shape[0], -1, dists.shape[-1])

        # Get offset
        diffs = points[..., 1:, :] - points[..., :-1, :]

        offset = diffs.new_zeros(diffs.shape[0], diffs.shape[1], self.random_per_sample) \
            + torch.linspace(
                0.0, 1.0 - 1.0 / self.random_per_sample, self.random_per_sample, device=diffs.device
            ).view(1, 1, -1)
        offset = offset + torch.rand_like(offset) / self.random_per_sample

        # Add offset
        points = torch.cat(
            [
                points[:, :-1, None, :] + offset.unsqueeze(-1) * diffs.unsqueeze(-2),
                points[:, -1:, None, :].repeat(1, 1, self.random_per_sample, 1)
            ],
            dim=1
        ).view(points.shape[0], -1, 3)
        dists = torch.linalg.norm(points - rays[..., None, :3], dim=-1)

        x[self.out_points_field] = points
        x[self.out_distances_field] = dists

        # Update outputs
        for key in x.keys():
            if key not in ['points', 'distances', 'rays']:
                x[key] = x[key].view(
                    points.shape[0], -1, 1, x[key].shape[-1]
                ).repeat(1, 1, self.random_per_sample, 1)
                x[key] = x[key].view(points.shape[0], -1, x[key].shape[-1])

        return x

    def set_iter(self, i):
        self.cur_iter = i


class ColorTransformEmbedding(nn.Module):
    def __init__(
        self,
        in_channels,
        cfg,
        **kwargs
    ):
        super().__init__()

        self.opt_group = cfg.group if 'group' in cfg else (kwargs['group'] if 'group' in kwargs else 'embedding')
        self.cfg = cfg

        # Out fields
        self.out_transform_field = cfg.out_transform_field if 'out_transform_field' in cfg else 'color_transform_global'
        self.out_shift_field = cfg.out_shift_field if 'out_shift_field' in cfg else 'color_shift_global'

        # Transform
        self.num_views = kwargs['system'].dm.train_dataset.total_images_per_frame
        self.val_all = kwargs['system'].dm.train_dataset.val_all
        self.color_embedding = nn.Parameter(
            torch.zeros((self.num_views, 12), device='cuda'), requires_grad=True
        )
        self.transform_activation = get_activation(
            cfg.transform_activation if 'transform_activation' in cfg else 'identity'
        )
        self.shift_activation = get_activation(
            cfg.shift_activation if 'shift_activation' in cfg else 'identity'
        )

    def forward(self, x: Dict[str, torch.Tensor], render_kwargs: Dict[str, str]):
        if not self.val_all:
            return x

        camera_ids = torch.round(x['rays'][..., -2]).long()
        color_transforms = self.color_embedding[camera_ids]

        x[self.out_transform_field] = self.transform_activation(color_transforms[..., :9])
        x[self.out_shift_field] = self.shift_activation(color_transforms[..., -3:])

        return x

    def set_iter(self, i):
        self.cur_iter = i
        self.transform_activation.set_iter(i)
        self.shift_activation.set_iter(i)


class ContractEmbedding(nn.Module):
    def __init__(
        self,
        in_channels,
        cfg,
        **kwargs
    ):
        super().__init__()

        self.group = cfg.group if 'group' in cfg else (kwargs['group'] if 'group' in kwargs else 'embedding')
        self.cfg = cfg

        # Origin
        self.origin = torch.tensor(cfg.origin if 'origin' in cfg else [0.0, 0.0, 0.0], device='cuda')

        # Contract function
        self.contract_fn = contract_dict[cfg.contract.type](
            cfg.contract, system=kwargs['system']
        )

        # In & out fields
        self.in_points_field = cfg.in_points_field if 'in_points_field' in cfg else 'points'
        self.in_distance_field = cfg.in_distance_field if 'in_distance_field' in cfg else 'distance'
        self.in_direction_field = cfg.in_direction_field if 'in_direction_field' in cfg else 'viewdirs'

        self.out_points_field = cfg.out_points_field if 'out_points_field' in cfg else 'points'
        self.out_direction_field = cfg.out_direction_field if 'out_direction_field' in cfg else 'viewdirs'
        self.out_distance_field = cfg.out_distance_field if 'out_distance_field' in cfg else 'distances'

    def forward(self, x: Dict[str, torch.Tensor], render_kwargs: Dict[str, str]):
        # Get rays
        rays = x['rays']
        rays = torch.cat(
            [
                rays[..., :3] - self.origin[None],
                rays[..., 3:6],
            ],
            dim=-1
        )

        # Get points
        points = x[self.in_points_field]
        dists = x[self.in_distance_field]

        points, dists = self.contract_fn.contract_points_and_distance(
            rays[..., :3], points, dists
        )

        # Get viewing directions
        viewdirs = torch.cat(
            [
                points[..., 1:, :] - points[..., :-1, :],
                torch.ones_like(points[..., :1, :])
            ],
            dim=1
        )

        # Output
        x[self.out_points_field] = points
        x[self.out_direction_field] = viewdirs
        x[self.out_distance_field] = dists

        return x

    def set_iter(self, i):
        self.cur_iter = i


class ReflectEmbedding(nn.Module):
    def __init__(
        self,
        in_channels,
        cfg,
        **kwargs
    ):
        super().__init__()

        self.group = cfg.group if 'group' in cfg else (kwargs['group'] if 'group' in kwargs else 'embedding')
        self.cfg = cfg

        # In & out fields
        self.in_points_field = cfg.in_points_field if 'in_points_field' in cfg else 'points'
        self.in_direction_field = cfg.in_direction_field if 'in_direction_field' in cfg else 'viewdirs'
        self.in_normal_field = cfg.in_normal_field if 'in_normal_field' in cfg else 'normal'
        self.in_distance_field = cfg.in_distance_field if 'in_distance_field' in cfg else 'ref_distance'

        self.direction_offset_field = cfg.direction_offset_field if 'direction_offset_field' in cfg else 'ref_viewdirs_offset'

        self.out_points_field = cfg.out_points_field if 'out_points_field' in cfg else 'ref_points'
        self.out_direction_field = cfg.out_direction_field if 'out_direction_field' in cfg else 'ref_viewdirs'
        self.out_normal_field = cfg.out_normal_field if 'out_normal_field' in cfg else 'normal'

        # Forward facing
        self.forward_facing = cfg.forward_facing if 'forward_facing' in cfg else False
        self.direction_init = cfg.direction_init if 'direction_init' in cfg else False

    def forward(self, x: Dict[str, torch.Tensor], render_kwargs: Dict[str, str]):
        rays = x['rays']

        # Get points, viewdirs & normal
        points = x[self.in_points_field]

        if self.in_direction_field not in x:
            dirs = rays[..., None, 3:6].repeat(1, points.shape[1], 1)
        else:
            dirs = x[self.in_direction_field]

        normal = x[self.in_normal_field]

        if self.forward_facing:
            normal[..., -1] = normal[..., -1] - 1
        elif self.direction_init:
            normal = normal - dirs

        normal = nn.functional.normalize(normal, dim=-1)
        x[self.out_normal_field] = normal

        # Get reflected directions & points
        ref_dirs = reflect(dirs, normal)
        ref_distance = x[self.in_distance_field]
        points = points + torch.abs(ref_distance) * ref_dirs

        if self.direction_offset_field in x:
            ref_dirs = ref_dirs + x[self.direction_offset_field].view(*points.shape)
            ref_dirs = nn.functional.normalize(ref_dirs, dim=-1)

        # Outputs
        x[self.out_points_field] = points
        x[self.out_direction_field] = ref_dirs

        return x

    def set_iter(self, i):
        self.cur_iter = i


class AdvectPointsEmbedding(nn.Module):
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

        self.in_points_field = cfg.in_points_field if 'in_points_field' in cfg else 'points'
        self.out_points_field = cfg.out_points_field if 'out_points_field' in cfg else 'points'
        self.save_points_field = cfg.save_points_field if 'save_points_field' in cfg else None

        self.out_offset_field = cfg.out_offset_field if 'out_offset_field' in cfg else 'offset'

        # Flow params
        self.use_spatial_flow = cfg.use_spatial_flow if 'use_spatial_flow' in cfg else False
        self.use_angular_flow = cfg.use_angular_flow if 'use_angular_flow' in cfg else False

        self.flow_keyframes = kwargs["system"].dm.train_dataset.num_keyframes
        self.total_frames = kwargs["system"].dm.train_dataset.num_frames
        self.flow_scale = cfg.flow_scale if 'flow_scale' in cfg else 0.0

        self.spatial_flow_activation = get_activation(
            cfg.spatial_flow_activation if 'spatial_flow_activation' in cfg else 'identity'
        )
        self.angular_flow_rotation_activation = get_activation(
            cfg.angular_flow_rotation_activation if 'angular_flow_rotation_activation' in cfg else 'identity'
        )
        self.angular_flow_anchor_activation = get_activation(
            cfg.angular_flow_anchor_activation if 'angular_flow_anchor_activation' in cfg else 'identity'
        )

    def forward(self, x: Dict[str, torch.Tensor], render_kwargs: Dict[str, str]):
        rays = x[self.rays_name]
        points = x[self.in_points_field]
        t = rays[..., -1:]

        if self.save_points_field is not None:
            x[self.save_points_field] = points

        # Get base time and time offset
        base_t = get_base_time(
            t,
            self.flow_keyframes,
            self.total_frames,
            self.flow_scale,
            self.training and (not 'no_flow_jitter' in render_kwargs)
        )

        time_offset = (t - base_t)[..., None, :]
        #time_offset = (t - base_t)[..., None, :] * (self.flow_keyframes)

        # Apply angular flow
        if self.use_angular_flow:
            angular_flow_rot = self.angular_flow_rotation_activation(x['angular_flow'][..., :3])
            angular_flow_anchor = self.angular_flow_anchor_activation(x['angular_flow'][..., 3:6])
            x['angular_flow_rot'] = angular_flow_rot
            x['angular_flow_anchor'] = angular_flow_anchor

            angular_flow_rot = axis_angle_to_matrix(angular_flow_rot * time_offset)

            points_shape = points.shape
            points = points - angular_flow_anchor
            points = (angular_flow_rot.view(-1, 3, 3) @ points.view(-1, 3, 1)).squeeze(-1)
            points = points.view(*points_shape)
            points = points + angular_flow_anchor

        # Apply spatial flow
        if self.use_spatial_flow:
            spatial_flow = self.spatial_flow_activation(x['spatial_flow'])
            x['spatial_flow'] = spatial_flow

            points = points + spatial_flow * time_offset

        # Update outputs
        x[self.out_points_field] = points
        x['base_times'] = base_t[..., None, :].repeat(1, points.shape[1], 1)
        x['time_offset'] = time_offset.repeat(1, points.shape[1], 1)

        if self.out_offset_field is not None:
            x[self.out_offset_field] = x[self.in_points_field] - points

        # Return
        return x

    def set_iter(self, i):
        self.cur_iter = i


class AddPointOutputsEmbedding(nn.Module):
    extra_outputs: List[str]

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

        # Extra outputs
        self.extra_outputs = list(cfg.extra_outputs)

    def forward(self, x: Dict[str, torch.Tensor], render_kwargs: Dict[str, str]):
        rays = x[self.rays_name]

        if 'times' in self.extra_outputs and 'times' not in x:
            x['times'] = rays[..., None, -1:].repeat(1, x['points'].shape[1], 1)

        if 'base_times' in self.extra_outputs and 'base_times' not in x:
            x['base_times'] = rays[..., None, -1:].repeat(1, x['points'].shape[1], 1)

        if 'viewdirs' in self.extra_outputs and 'viewdirs' not in x:
            x['viewdirs'] = rays[..., None, 3:6].repeat(1, x['points'].shape[1], 1)

        return x

    def set_iter(self, i):
        self.cur_iter = i


point_embedding_dict = {
    'point_prediction': PointPredictionEmbedding,
    'extract_fields': ExtractFieldsEmbedding,
    'create_points': CreatePointsEmbedding,
    'point_density': PointDensityEmbedding,
    'point_offset': PointOffsetEmbedding,
    'generate_samples': GenerateNumSamplesEmbedding,
    'select_points': SelectPointsEmbedding,
    'random_offset': RandomOffsetEmbedding,
    'color_transform': ColorTransformEmbedding,
    'contract': ContractEmbedding,
    'reflect': ReflectEmbedding,
    'advect_points': AdvectPointsEmbedding,
    'add_point_outputs': AddPointOutputsEmbedding,
}
