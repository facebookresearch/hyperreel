#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
from torch import nn

from nlf.activations import get_activation
from nlf.param import RayParam

from .mlp import mlp_dict
from .array_nd import array_dict


tensor_dict = {}


class TensorProduct(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        cfg,
        **kwargs
    ):
        super().__init__()

        self.group = cfg.group if 'group' in cfg else (kwargs['group'] if 'group' in kwargs else 'color')

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.latent_dim = kwargs['latent_dim'] if 'latent_dim' in kwargs else 0
        self.num_tensors = len(cfg.tensors.keys())
        self.num_basis = cfg.num_basis
        self.num_opacity_basis = cfg.num_opacity_basis if 'num_opacity_basis' in cfg else self.num_basis
        self.use_opacity = 'num_opacity_basis' in cfg

        if 'latent_dim' in cfg:
            self.latent_dim = cfg.latent_dim

        # Activation
        self.activation = cfg.activation if 'activation' in cfg else 'identity'
        self.out_layer = get_activation(self.activation)

        # Basis
        self.has_basis = 'basis' in cfg
        self.separate_basis = cfg.separate_basis if 'separate_basis' in cfg else False

        if 'basis' in cfg:
            self.basis = mlp_dict[cfg.basis.type](
                self.in_channels,
                self.num_basis * (self.out_channels - 1) + self.num_opacity_basis,
                cfg.basis,
                group=self.group
            )

            if self.use_opacity:
                tensor_out_channels = self.num_basis + self.num_opacity_basis
            else:
                tensor_out_channels = self.num_basis
        else:
            tensor_out_channels = self.num_basis * (self.out_channels - 1) + self.num_opacity_basis

        # Tensors
        self.tensors = []

        for idx, tensor_key in enumerate(cfg.tensors.keys()):
            tensor_cfg = cfg.tensors[tensor_key]
            cur_tensor = array_dict[tensor_cfg.type](
                self.in_channels,
                tensor_out_channels,
                tensor_cfg,
                group=self.group
            )
            self.tensors.append(cur_tensor)

        self.tensors = nn.ModuleList(self.tensors)

    # TODO: Add VBNF basis model (with options for discretized / non-discretized look-up)
    # TODO: Option for separate basis for each tensor

    def forward(self, x, render_kwargs):
        # Get coefficients
        outputs = []

        for idx, tensor in enumerate(self.tensors):
            outputs.append(tensor(x))

        coeffs = torch.stack(outputs, -1).prod(-1)[..., None]

        # Get basis
        if self.has_basis:
            basis = self.basis(x)

            # Separate into color, opacity
            if self.use_opacity:
                color_basis = basis[..., :-self.num_opacity_basis].view(
                    x.shape[0], self.num_basis, self.out_channels - 1
                )
                #opacity_basis = basis[..., -self.num_opacity_basis:].view(
                opacity_basis = basis.new_ones(
                    x.shape[0], self.num_opacity_basis, 1
                )

                color_coeffs = coeffs[..., :-self.num_opacity_basis, :]
                opacity_coeffs = coeffs[..., -self.num_opacity_basis:, :]
            else:
                basis = basis.view(x.shape[0], self.num_basis, self.out_channels)
        else:
            # Separate into color, opacity
            if self.use_opacity:
                color_coeffs = coeffs[..., :-self.num_opacity_basis, :].view(
                    x.shape[0], self.num_basis, self.out_channels - 1
                )
                opacity_coeffs = coeffs[..., -self.num_opacity_basis:, :].view(
                    x.shape[0], self.num_opacity_basis, 1
                )

                color_basis = torch.ones_like(color_coeffs)
                opacity_basis = torch.ones_like(opacity_coeffs)
            else:
                coeffs = coeffs.view(x.shape[0], self.num_basis, self.out_channels)
                basis = torch.ones_like(coeffs)

        # Return
        if self.use_opacity:
            color = self.out_layer((color_coeffs * color_basis).mean(1))
            opacity = self.out_layer((opacity_coeffs * opacity_basis).mean(1))
            return torch.cat([color, opacity], -1)
        else:
            return self.out_layer((coeffs * basis).mean(1))

    def set_iter(self, i):
        for tensor in self.tensors:
            tensor.set_iter(i)


tensor_dict['tensor_product'] = TensorProduct


class TensorConcat(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        cfg,
        **kwargs
    ):
        super().__init__()

        self.group = cfg.group if 'group' in cfg else (kwargs['group'] if 'group' in kwargs else 'color')

        # In channels, out channels
        self.latent_dim = kwargs['latent_dim'] if 'latent_dim' in kwargs else 0
        self.in_channels = in_channels
        self.out_channels = self.out_channels

        if 'latent_dim' in cfg:
            self.latent_dim = cfg.latent_dim

        # Num basis, num features
        self.num_tensors = len(cfg.tensors.keys())
        self.num_basis = cfg.num_basis
        self.num_features = self.out_channels // self.num_tensors
        self.num_extra = self.out_channels - self.num_features * self.num_tensors

        # Activation
        self.activation = cfg.activation if 'activation' in cfg else 'identity'
        self.out_layer = get_activation(self.activation)

        # Tensors
        self.tensors = []

        for idx, tensor_key in enumerate(cfg.tensors.keys()):
            tensor_cfg = cfg.tensors[tensor_key]

            cur_tensor = array_dict[tensor_cfg.type](
                self.input_channels,
                self.num_basis * (self.num_features + self.num_extra),
                tensor_cfg,
                group=self.group
            )
            self.tensors.append(cur_tensor)

        self.tensors = nn.ModuleList(self.tensors)

    def forward(self, x, **kwargs):
        outputs = []
        extras = []

        for idx, tensor in enumerate(self.tensors):
            cur_output = tensor(x)
            cur_output.view(x.shape[0], self.num_basis, self.num_features + self.num_extra)

            outputs.append(cur_output[..., :self.num_features])
            extras.append(cur_output[..., self.num_features:])

        extras = torch.stack(extras, -1).prod(-1)[..., None].mean(1)
        outputs = torch.cat(outputs, -1).mean(1)
        return self.out_layer(torch.cat([outputs, extras]), -1)

    def set_iter(self, i):
        for tensor in self.tensors:
            tensor.set_iter(i)


tensor_dict['tensor_concat'] = TensorConcat


class TensorPassthrough(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        cfg,
        **kwargs
    ):
        super().__init__()

        # Tensors
        self.tensors = []

        for idx, tensor_key in enumerate(cfg.tensors.keys()):
            tensor_cfg = cfg.tensors[tensor_key]

            cur_tensor = array_dict[tensor_cfg.type](
                in_channels,
                out_channels,
                tensor_cfg,
                **kwargs
            )
            self.tensors.append(cur_tensor)

        self.tensors = nn.ModuleList(self.tensors)

    def forward(self, x, **kwargs):
        return self.tensors[0](x)

    def set_iter(self, i):
        for tensor in self.tensors:
            tensor.set_iter(i)


tensor_dict['tensor_passthrough'] = TensorPassthrough


def mean(tensors, *args, **kwargs):
    return torch.mean(tensors, -2)


def over_composite_one(rgb, alphas, **kwargs):
    alphas_shifted = torch.cat(
        [
            torch.ones_like(alphas[:, :1]),
            1 - alphas + 1e-8
        ],
        -1
    )
    weights = alphas * torch.cumprod(
        alphas_shifted, -1
    )[:, :-1]
    accum = weights.sum(-1)

    rgb_final = torch.sum(weights.unsqueeze(-1) * rgb, -2)

    if 'white_background' in kwargs and kwargs['white_background']:
        rgb_final = rgb_final + (1.0 - accum.unsqueeze(-1))

    return rgb_final, accum, weights


def over(rgba, *args, **kwargs):
    rgb = rgba[..., :-1]
    alpha = rgba[..., -1]

    return over_composite_one(rgb, alpha)[0]

def _over_opacity(rgba, deltas):
    rgb = torch.sigmoid(rgba[..., :-1])
    density = torch.relu(rgba[..., -1])
    alpha = 1 - torch.exp(-deltas * density)

    return over_composite_one(rgb, alpha)[0]

def over_opacity(rgba, *args, **kwargs):
    rgb = torch.sigmoid(rgba[..., :-1])
    density = rgba[..., -1]
    density = torch.relu(density)
    alpha = 1 - torch.exp(-(4.0 / rgb.shape[1]) * density)

    return over_composite_one(rgb, alpha, **kwargs)[0]

def over_opacity_extra(rgba, *args, **kwargs):
    # RGB
    rgb = torch.sigmoid(rgba[..., :-1])

    # Density
    density = rgba[..., -1]

    # Remove samples with distance 0
    density = torch.where(
        (kwargs['distance'] * torch.ones_like(density)) < 1e-5,
        torch.zeros_like(density),
        density
    )

    #density = nn.functional.softplus(density - 10.0)
    density = torch.relu(density)

    # Density factor
    density = density * kwargs['density_factor']

    # Calculate alpha
    #alpha = 1 - torch.exp(-(4.0 / rgb.shape[1]) * density)

    deltas = torch.cat(
        [
            torch.abs(kwargs['distance'][:, 1:] - kwargs['distance'][:, :-1]),
            10000 * torch.ones_like(kwargs['distance'][:, :1])
        ],
        dim=1
    )
    alpha = 1 - torch.exp(-deltas * density)

    return over_composite_one(rgb, alpha, **kwargs)[0]

def concat(outputs, *args, **kwargs):
    return outputs.view(*(outputs.shape[0:-2] + (-1,)))


reduce_dict = {
    'mean': mean,
    'over': over,
    'over_opacity': over_opacity,
    'over_opacity_extra': over_opacity_extra,
    'concat': concat,
}


class TensorReduce(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        cfg,
        **kwargs
    ):
        super().__init__()

        self.group = cfg.group if 'group' in cfg else (kwargs['group'] if 'group' in kwargs else 'color')

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.latent_dim = kwargs['latent_dim'] if 'latent_dim' in kwargs else 0

        if 'latent_dim' in cfg:
            self.latent_dim = cfg.latent_dim

        self.num_partitions = cfg.num_partitions if 'num_partitions' in cfg else -1
        self.num_tensors = len(cfg.tensors.keys())

        self.out_feature_dim = cfg.out_feature_dim if 'out_feature_dim' in cfg else out_channels
        self.use_feature_net = ('feature_net' in cfg)
        self.white_background = False if 'white_background' not in cfg else cfg.white_background
        self.density_factor = 1.0 if 'density_factor' not in cfg else cfg.density_factor

        # Reduce fns
        if 'reduce' not in cfg:
            cfg.reduce = 'mean'

        if 'reduce_partitions' not in cfg:
            cfg.reduce_partitions = 'over_opacity_extra'

        self.reduce_fn = reduce_dict[cfg.reduce]
        self.reduce_partition_fn = reduce_dict[cfg.reduce_partitions]

        # Combined opacity and color network / tensors
        self.use_opacity = ('over' in cfg.reduce_partitions) and (self.num_partitions > 0)

        if self.use_feature_net or not self.use_opacity:
            tensor_out_channels = self.out_feature_dim
        else:
            tensor_out_channels = self.out_feature_dim + 1

        # Feature net
        if self.use_feature_net:
            # Create
            net_cfg = cfg.feature_net

            # Input channels
            self.feature_ray_channels = net_cfg.ray_channels if 'ray_channels' in net_cfg else 0
            self.remove_rays = net_cfg.remove_rays if 'remove_rays' in net_cfg else True
            self.in_channels -= self.feature_ray_channels

            # Ray param
            if self.feature_ray_channels > 0:
                self.feature_param = RayParam(net_cfg.param)
                self.feature_ray_in_channels = self.feature_param.out_channels
            else:
                self.feature_param = None
                self.feature_ray_in_channels = 0

            if self.num_partitions > 0 and not self.use_opacity:
                self.feature_net = mlp_dict[net_cfg.type](
                    self.out_feature_dim * self.num_partitions + self.feature_ray_in_channels,
                    self.out_channels + (1 if self.use_opacity else 0),
                    net_cfg,
                    group=self.group
                )
            else:
                self.feature_net = mlp_dict[net_cfg.type](
                    self.out_feature_dim * self.num_tensors + self.feature_ray_in_channels,
                    self.out_channels + 1,
                    net_cfg,
                    group=self.group
                )
        else:
            self.feature_param = None
            self.feature_ray_in_channels = 0

        # Activation
        self.activation = cfg.activation if 'activation' in cfg else 'identity'
        self.out_layer = get_activation(self.activation)

        # Tensors
        self.tensors = []

        for _, tensor_key in enumerate(cfg.tensors.keys()):
            tensor_cfg = cfg.tensors[tensor_key]
            cur_tensor = tensor_dict[tensor_cfg.type](
                self.in_channels,
                tensor_out_channels,
                tensor_cfg,
                latent_dim=0,
                group=self.group
            )
            self.tensors.append(cur_tensor)

        self.tensors = nn.ModuleList(self.tensors)

    def forward(self, x, render_kwargs):
        points = x['points']
        distances = x['distances']

        batch_size = points.shape[0]

        x = torch.cat(
            [
                points.view(batch_size, -1, 3),
                distances.view(batch_size, -1, 1),
            ],
            -1
        ).view(batch_size, -1)

        # Reshape
        if self.feature_param is not None:
            param_rays = self.feature_param(
                x[..., :self.feature_ray_channels]
            )

            if self.remove_rays:
                x = x[..., self.feature_ray_channels:]
        else:
            param_rays = x[..., 0:0]

        if self.latent_dim > 0:
            x = x[..., :-self.latent_dim]

        if self.num_partitions != -1:
            x = x.reshape(-1, x.shape[-1] // self.num_partitions)

        # Get colors and opacities
        outputs = []
        opacities = []

        for idx, tensor in enumerate(self.tensors):
            cur_output = tensor(x, render_kwargs)
            outputs.append(cur_output[..., :-1])
            opacities.append(cur_output[..., -1:])

        outputs = torch.stack(outputs, 1)
        opacities = torch.stack(opacities, 1)

        if self.use_feature_net or not self.use_opacity:
            outputs = torch.cat([outputs, opacities], -1)

        # Partitioned forward
        if self.num_partitions > 0:
            # Reduce
            outputs = outputs.view(batch_size, self.num_partitions, -1, outputs.shape[-1])
            outputs = self.reduce_fn(outputs)

            # Feature net
            if self.use_feature_net and self.use_opacity:
                outputs = outputs.view(-1, outputs.shape[-1])
                param_rays = param_rays.unsqueeze(1).repeat(1, self.num_partitions, 1).view(
                    outputs.shape[0], param_rays.shape[-1]
                )
                outputs = self.feature_net(torch.cat([param_rays, outputs], -1))
                outputs = outputs.view(batch_size, self.num_partitions, outputs.shape[-1])
            # Combine color and opacity
            elif self.use_opacity:
                opacities = opacities.view(batch_size, self.num_partitions, -1, opacities.shape[-1])
                opacities = self.reduce_fn(opacities)
                outputs = torch.cat([outputs, opacities], -1)

            # For visualization
            if 'keep_tensor_partitions' in render_kwargs:
                outputs = outputs[:, render_kwargs['keep_tensor_partitions'], :].view(
                    batch_size,
                    len(render_kwargs['keep_tensor_partitions']),
                    outputs.shape[-1]
                )

            # Reduce partitions
            x = x.view(batch_size, self.num_partitions, -1)

            outputs = self.reduce_partition_fn(
                outputs,
                inputs=x[..., :3],
                distance=x[..., -1],
                density_factor=self.density_factor,
                white_background=self.white_background
            )

            # Feature net
            if self.use_feature_net and not self.use_opacity:
                param_rays = param_rays.view(outputs.shape[0], param_rays.shape[-1])
                outputs = self.feature_net(torch.cat([param_rays, outputs], -1))
        else:
            # Reduce
            outputs = outputs.view(batch_size, -1, outputs.shape[-1])
            outputs = self.reduce_fn(outputs)

            # Feature net
            if self.use_feature_net:
                param_rays = param_rays.view(outputs.shape[0], param_rays.shape[-1])
                outputs = self.feature_net(torch.cat([param_rays, outputs], -1))

        rgb_map = self.out_layer(outputs)

        if "fields" in render_kwargs:
            return {
            }
        else:
            return rgb_map

    def set_iter(self, i):
        for tensor in self.tensors:
            tensor.set_iter(i)


tensor_dict['tensor_sum'] = TensorReduce
