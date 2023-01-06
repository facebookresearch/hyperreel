#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn

from nlf.activations import get_activation
from nlf.pe import IdentityPE, pe_dict


class ZeroMLP(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        cfg,
        **kwargs
    ):
        super().__init__()

        self.opt_group = cfg.group if 'group' in cfg else (kwargs['group'] if 'group' in kwargs else 'color')

        self.out_channels = out_channels
        self.layer = nn.Linear(1, 1)

    def forward(self, x):
        return x.new_zeros(x.shape[0], self.out_channels)

    def set_iter(self, i):
        self.cur_iter = i


class ConstantMLP(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        cfg,
        **kwargs
    ):
        super().__init__()

        self.opt_group = cfg.group if 'group' in cfg else (kwargs['group'] if 'group' in kwargs else 'color')

        self.out_channels = out_channels
        self.layer = nn.Linear(1, out_channels)
        self.out_layer = get_activation(cfg.activation if 'activation' in cfg else 'identity')

    def forward(self, x):
        out = self.out_layer(self.layer.bias).unsqueeze(0)
        return out.expand(x.shape[0], out.shape[-1])

    def set_iter(self, i):
        self.cur_iter = i


class BaseMLP(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        cfg,
        **kwargs
    ):
        super().__init__()

        self.opt_group = cfg.group if 'group' in cfg else (kwargs['group'] if 'group' in kwargs else 'color')

        self.start_channel = kwargs['start_channel'] if 'start_channel' in kwargs else 0
        self.in_channels = in_channels - self.start_channel
        self.out_channels = out_channels if cfg.depth != 0 else self.in_channels
        self.latent_dim = kwargs['latent_dim'] if 'latent_dim' in kwargs else 0
        self.pe_channels = cfg.pe_channels if 'pe_channels' in cfg else self.in_channels
        self.zero_before_channel = cfg.zero_before_channel if 'zero_before_channel' in cfg else None
        self.linear_last = cfg.linear_last if 'linear_last' in cfg else True
        self.bias = cfg.bias if 'bias' in cfg else True
        self.pad_to = cfg.pad_to if 'pad_to' in cfg else None

        if 'latent_dim' in cfg:
            self.latent_dim = cfg.latent_dim

        if self.pe_channels == 'all':
            self.pe_channels = self.in_channels + self.latent_dim
            self.in_channels = self.pe_channels
            self.latent_dim = 0

        # Global
        self.is_constant = cfg.is_constant if 'is_constant' in cfg else False

        if self.is_constant:
            if 'pe_channels' in cfg and 'latent_dim' not in cfg:
                self.latent_dim = self.in_channels - self.pe_channels
            else:
                self.pe_channels = self.latent_dim

            net_in_channels = self.latent_dim
        else:
            # PE
            if 'pe' in cfg:
                self.pe = pe_dict[cfg.pe.type](
                    self.pe_channels,
                    cfg.pe
                )
                self.pe_out_channels = self.pe.out_channels
            else:
                self.pe = IdentityPE(self.pe_channels)
                self.pe_out_channels = self.pe_channels

            net_in_channels = self.pe_out_channels + (self.in_channels - self.pe_channels) + self.latent_dim

        # Padding
        if self.pad_to is not None:
            net_in_channels = self.pad_to

        # MLP
        self.D = cfg.depth
        self.W = cfg.hidden_channels

        self.skips = list(cfg.skips) if 'skips' in cfg else []
        self.activation = cfg.activation if 'activation' in cfg else 'identity'
        self.layer_activation = cfg.layer_activation if 'layer_activation' in cfg else 'leaky_relu'
        self.layers = nn.ModuleList()

        for i in range(self.D + 2):
            if i == 0:
                layer = nn.Linear(net_in_channels, self.W, bias=self.bias)

                if self.zero_before_channel is not None:
                    zero_before_channel = self.zero_before_channel * self.pe_channels * (2 * cfg.pe.n_freqs + (0 if cfg.pe.exclude_identity else 1))

                    with torch.no_grad():
                        if self.latent_dim > 0:
                            layer.weight[..., zero_before_channel:-self.latent_dim] = 0.0
                        else:
                            layer.weight[..., zero_before_channel:] = 0.0

            elif i == self.D + 1:
                layer = nn.Linear(self.W, self.out_channels, bias=self.bias)
            elif i in self.skips:
                layer = nn.Linear(self.W + net_in_channels, self.W, bias=self.bias)
            else:
                layer = nn.Linear(self.W, self.W, bias=self.bias)

            if self.linear_last:
                if i < self.D:
                    layer = nn.Sequential(layer, get_activation(self.layer_activation))
            else:
                if i < self.D + 1:
                    layer = nn.Sequential(layer, get_activation(self.layer_activation))

            self.layers.append(layer)

        # Output
        self.out_layer = get_activation(self.activation)

    def forward(self, x):
        if self.pad_to is not None:
            x = torch.cat([x, x.new_ones(x.shape[0], self.pad_to - x.shape[-1])], -1)

        # Run forward
        input_x = x

        for i, layer in enumerate(self.layers):
            if i in self.skips:
                x = torch.cat([input_x, x], -1)

            x = layer(x)

        return self.out_layer(x)

    def set_iter(self, i):
        self.cur_iter = i

        if not self.is_constant:
            self.pe.set_iter(i)


class PartitionedConstantMLP(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        cfg,
        **kwargs
    ):
        super().__init__()

        self.opt_group = cfg.group if 'group' in cfg else (kwargs['group'] if 'group' in kwargs else 'color')

        self.out_channels = out_channels
        self.num_partitions = cfg.num_partitions
        self.layer = nn.Linear(1, self.out_channels * self.num_partitions)
        self.out_layer = get_activation(cfg.activation if 'activation' in cfg else 'identity')

    def forward(self, x):
        return self.out_layer(self.layer.bias)

    def set_iter(self, i):
        self.cur_iter = i


class PartitionedMLP(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        cfg,
        **kwargs
    ):
        super().__init__()

        self.opt_group = cfg.group if 'group' in cfg else (kwargs['group'] if 'group' in kwargs else 'color')

        self.start_channel = kwargs['start_channel'] if 'start_channel' in kwargs else 0
        self.in_channels = in_channels - self.start_channel
        self.out_channels = out_channels if cfg.depth != 0 else self.in_channels
        self.pe_channels = cfg.pe_channels if 'pe_channels' in cfg else self.in_channels
        self.latent_dim = kwargs['latent_dim'] if 'latent_dim' in kwargs else 0
        self.use_latent = cfg.use_latent if 'use_latent' in cfg else False
        self.zero_before_channel = cfg.zero_before_channel if 'zero_before_channel' in cfg else None
        self.linear_last = cfg.linear_last if 'linear_last' in cfg else True

        if 'latent_dim' in cfg:
            self.latent_dim = cfg.latent_dim

        # Global
        self.is_constant = cfg.is_constant if 'is_constant' in cfg else False

        if self.is_constant:
            if 'pe_channels' in cfg and 'latent_dim' not in cfg:
                self.latent_dim = self.in_channels - self.pe_channels
            else:
                self.pe_channels = self.latent_dim

            net_in_channels = self.latent_dim
        else:
            # PE
            if 'pe' in cfg:
                self.pe = pe_dict[cfg.pe.type](
                    self.pe_channels,
                    cfg.pe
                )
                self.pe_out_channels = self.pe.out_channels
            else:
                self.pe = IdentityPE(self.pe_channels)
                self.pe_out_channels = self.pe_channels

            if self.use_latent:
                net_in_channels = self.pe_out_channels + (self.in_channels - self.pe_channels) + self.latent_dim
            else:
                net_in_channels = self.pe_out_channels + (self.in_channels - self.pe_channels)

        # MLP
        self.num_partitions = cfg.num_partitions
        self.D = cfg.depth
        self.W = cfg.hidden_channels

        self.skips = list(cfg.skips) if 'skips' in cfg else []
        self.activation = cfg.activation if 'activation' in cfg else 'identity'
        self.layer_activation = cfg.layer_activation if 'layer_activation' in cfg else 'leaky_relu'

        # Partitioned layers
        for i in range(self.D + 2):
            if i == 0:
                layer = nn.Linear(net_in_channels, self.W * self.num_partitions)

                if self.zero_before_channel is not None:
                    zero_before_channel = self.zero_before_channel * self.pe_channels * (2 * cfg.pe.n_freqs + (0 if cfg.pe.exclude_identity else 1))

                    with torch.no_grad():
                        if self.latent_dim > 0:
                            layer.weight[..., zero_before_channel:-self.latent_dim] = 0.0
                        else:
                            layer.weight[..., zero_before_channel:] = 0.0

            elif i == (self.D + 1):
                layer = nn.Linear(self.W, self.out_channels * self.num_partitions)
            elif i in self.skips:
                layer = nn.Linear(self.W + net_in_channels, self.W * self.num_partitions)
            else:
                layer = nn.Linear(self.W, self.W * self.num_partitions)

            if self.linear_last:
                if i < self.D:
                    layer = nn.Sequential(layer, get_activation(self.layer_activation))
                else:
                    layer = nn.Sequential(layer, get_activation('identity'))
            else:
                if i < self.D + 1:
                    layer = nn.Sequential(layer, get_activation(self.layer_activation))
                else:
                    layer = nn.Sequential(layer, get_activation('identity'))

            setattr(self, f'encoding{i+1}', layer)

        # Output
        self.out_layer = get_activation(self.activation)

    def forward(self, x):
        # Apply PE
        if self.is_constant:
            x = x[..., -self.latent_dim:]
        else:
            if self.latent_dim > 0 and self.use_latent:
                x = torch.cat(
                    [
                        x[..., self.start_channel:self.start_channel+self.in_channels],
                        x[..., -self.latent_dim:]
                    ],
                    dim=-1
                )
            else:
                x = x[..., self.start_channel:self.start_channel+self.in_channels]

            x = torch.cat(
                [
                    self.pe(x[..., :self.pe_channels]),
                    x[..., self.pe_channels:]
                ],
                dim=-1
            )

        # Run forward
        batch_size = x.shape[0] // self.num_partitions
        input_x = x

        for i in range(self.D + 2):
            if i in self.skips:
                x = torch.cat([input_x, x], -1)

            # Batch matmul
            layer = getattr(self, f'encoding{i+1}')
            weight = layer[0].weight

            weight = weight.view(self.num_partitions, weight.shape[0] // self.num_partitions, weight.shape[1]).permute(0, 2, 1)
            x = x.view(batch_size, self.num_partitions, x.shape[-1]).permute(1, 0, 2)
            x = torch.bmm(x, weight).permute(1, 0, 2)

            # Bias
            x = x + layer[0].bias.view(1, self.num_partitions, -1)

            # Apply non-linearity
            x = layer[1](x)
            x = x.reshape(batch_size * self.num_partitions, -1)

        return self.out_layer(x)

    def _forward(self, x):
        # Apply PE
        if self.is_constant:
            x = x[..., -self.latent_dim:]
        else:
            if self.latent_dim > 0:
                x = torch.cat(
                    [
                        x[..., self.start_channel:self.start_channel+self.in_channels],
                        x[..., -self.latent_dim:]
                    ],
                    dim=-1
                )
            else:
                x = x[..., self.start_channel:self.start_channel+self.in_channels]

            x = torch.cat(
                [
                    self.pe(x[..., :self.pe_channels]),
                    x[..., self.pe_channels:]
                ],
                dim=-1
            )

        # Run forward
        input_x = x

        for i in range(self.D + 2):
            if i in self.skips:
                x = torch.cat([input_x, x], -1)

            layer = getattr(self, f'encoding{i+1}')
            x = x @ layer[0].weight.permute(1, 0) + layer[0].bias
            x = layer[1](x)

        return self.out_layer(x)

    def set_iter(self, i):
        self.cur_iter = i

        if not self.is_constant:
            self.pe.set_iter(i)


from .siren import Siren


mlp_dict = {
    'zero': ZeroMLP,
    'constant': ConstantMLP,
    'partitioned_constant': PartitionedConstantMLP,
    'base': BaseMLP,
    'partitioned': PartitionedMLP,
    'siren': Siren
}
