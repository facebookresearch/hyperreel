#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
from torch import nn

from nlf.activations import get_activation


class SineLayer(nn.Module):
    # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of omega_0.

    # If is_first=True, omega_0 is a frequency factor which simply multiplies the activations before the
    # nonlinearity. Different signals may require different omega_0 in the first layer - this is a
    # hyperparameter.

    # If is_first=False, then the weights will be divided by omega_0 so as to keep the magnitude of
    # activations constant, but boost gradients to the weight matrix (see supplement Sec. 1.5)

    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        is_first=False,
        omega_0=30,
        **kwargs
        ):

        super().__init__()

        self.omega_0 = omega_0
        self.is_first = is_first

        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features,
                                             1 / self.in_features)
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0,
                                             np.sqrt(6 / self.in_features) / self.omega_0)

    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))

    def forward_with_intermediate(self, input):
        # For visualization of activation distributions
        intermediate = self.omega_0 * self.linear(input)
        return torch.sin(intermediate), intermediate


class Siren(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        cfg,
        **kwargs
        ):

        super().__init__()

        self.opt_group = cfg.group if 'group' in cfg else (kwargs['group'] if 'group' in kwargs else 'color')

        self.depth = cfg.depth
        self.skips = cfg.skips if 'skips' in cfg else []
        self.with_norm = cfg.with_norm if 'with_norm' in cfg else False

        self.start_channel = kwargs['start_channel'] if 'start_channel' in kwargs else 0
        self.in_channels = kwargs['num_channels'] if 'num_channels' in kwargs else (in_channels - self.start_channel)
        self.out_channels = out_channels if self.depth != 0 else self.in_channels
        self.latent_dim = kwargs['latent_dim'] if 'latent_dim' in kwargs else 0

        # Siren config
        self.first_omega_0 = cfg.first_omega_0 if 'first_omega_0' in cfg else 30.0
        self.hidden_omega_0 = cfg.hidden_omega_0 if 'hidden_omega_0' in cfg else 30.0
        self.outermost_linear = cfg.outermost_linear if 'outermost_linear' in cfg else True

        # Net
        for i in range(self.depth + 2):
            if i == 0:
                layer = SineLayer(
                    self.in_channels + self.latent_dim,
                    cfg.hidden_channels,
                    is_first=True,
                    omega_0=self.first_omega_0
                )
            elif i in self.skips:
                layer = SineLayer(
                    cfg.hidden_channels + self.in_channels + self.latent_dim,
                    cfg.hidden_channels,
                    is_first=False,
                    omega_0=self.hidden_omega_0
                )
            else:
                layer = SineLayer(
                    cfg.hidden_channels,
                    cfg.hidden_channels,
                    is_first=False,
                    omega_0=self.hidden_omega_0
                )

            if self.with_norm:
                layer = nn.Sequential(layer, nn.LayerNorm(cfg.hidden_channels, elementwise_affine=True))

            setattr(self, f'encoding{i+1}', layer)

        if self.outermost_linear:
            self.final_layer = nn.Linear(cfg.hidden_channels, self.out_channels)

            with torch.no_grad():
                self.final_layer.weight.uniform_(-np.sqrt(6 / cfg.hidden_channels) / self.hidden_omega_0,
                                              np.sqrt(6 / cfg.hidden_channels) / self.hidden_omega_0)
        else:
            self.final_layer = SineLayer(
                cfg.hidden_channels,
                self.out_channels,
                is_first=False,
                omega_0=self.hidden_omega_0
            )

        # Final activation
        self.activation = cfg.activation if 'activation' in cfg else 'identity'
        self.out_layer = get_activation(self.activation)

    def forward(self, x):
        if self.latent_dim > 0:
            x = torch.cat(
                [
                    x[..., self.start_channel:self.start_channel+self.in_channels],
                    x[..., -self.latent_dim:]
                ],
                -1
            )
        else:
            x = x[..., self.start_channel:self.start_channel+self.in_channels]

        # Run forward
        input_x = x

        for i in range(self.depth):
            if i in self.skips:
                x = torch.cat([input_x, x], -1)

            x = getattr(self, f'encoding{i+1}')(x)

        return self.out_layer(self.final_layer(x))

    def set_iter(self, i):
        self.cur_iter = i
