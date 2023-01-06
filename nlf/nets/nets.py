#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn
from nlf.activations import get_activation


net_dict = {}

# Add MLPs
from .mlp import mlp_dict
for k, v in mlp_dict.items(): net_dict[k] = v

# Add tensors
from .tensor import tensor_dict
for k, v in tensor_dict.items(): net_dict[k] = v

# Add tensoRFs
from .tensorf_base import tensorf_base_dict
for k, v in tensorf_base_dict.items(): net_dict[k] = v

from .tensorf_no_sample import tensorf_no_sample_dict
for k, v in tensorf_no_sample_dict.items(): net_dict[k] = v

from .tensorf_reflect import tensorf_reflect_dict
for k, v in tensorf_reflect_dict.items(): net_dict[k] = v

from .tensorf_dynamic import tensorf_dynamic_dict
for k, v in tensorf_dynamic_dict.items(): net_dict[k] = v

# Multiple net
class MultipleNet(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        cfg,
        **kwargs
    ):
        super().__init__()

        self.group = cfg.group if 'group' in cfg else (kwargs['group'] if 'group' in kwargs else 'color')

        self.cfg = cfg
        self.cur_iter = 0
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.latent_dim = kwargs['latent_dim'] if 'latent_dim' in kwargs else 0

        self.use_feature_net = ('feature_net' in cfg)
        self.out_feature_dim = cfg.out_feature_dim if 'feature_net' in cfg else out_channels

        if 'latent_dim' in cfg:
            self.latent_dim = cfg.latent_dim

        # Create nets
        self.nets = nn.ModuleList()
        self.wait_iters = []
        self.stop_iters = []
        self.feature_dims = []
        self.scales = []

        for idx, net_key in enumerate(cfg.nets.keys()):
            # Current config
            net_cfg = cfg.nets[net_key]

            # Wait
            self.wait_iters.append(net_cfg.wait_iters)
            self.stop_iters.append(net_cfg.stop_iters)
            self.feature_dims.append(net_cfg.feature_dim if 'feature_dim' in net_cfg else 0)
            self.scales.append(net_cfg.scale if 'scale' in net_cfg else 1.0)

            # Create current net
            net = net_dict[net_cfg.type](
                in_channels,
                self.out_feature_dim + self.feature_dims[-1],
                net_cfg,
                latent_dim=self.latent_dim + (self.feature_dims[-2] if idx > 0 else 0),
                group=self.group
            )

            self.nets.append(net)

        # Feature net
        if self.use_feature_net:
            net_cfg = cfg.nets[net_key]

            self.feature_net = net_dict[net_cfg.type](
                self.out_feature_dim,
                self.out_channels,
                net_cfg,
                group=self.group
            )

        # Activation
        self.activation = cfg.activation if 'activation' in cfg else 'identity'
        self.out_layer = get_activation(self.activation)

    def forward(self, x, **render_kwargs):
        total_output = 0.0
        feature_vector = x.new_zeros(x.shape[0], 0)

        for idx, net in enumerate(self.nets):
            if self.cur_iter < self.wait_iters[idx] \
                or self.cur_iter >= self.stop_iters[idx]:
                continue

            # Run current net
            cur_output = net(torch.cat([x, feature_vector], -1), **render_kwargs)

            if self.feature_dims[idx] > 0:
                feature_vector = cur_output[..., -self.feature_dims[idx]:]
                cur_output = cur_output[..., :-self.feature_dims[idx]]

            # Apply feature transform
            if self.use_feature_net:
                cur_output = self.feature_net(cur_output)

            # Add
            total_output += cur_output * self.scales[idx]

        # Final non-linearity
        return self.out_layer(total_output)

    def set_iter(self, i):
        self.cur_iter = i

        for idx in range(len(self.nets)):
            self.nets[idx].set_iter(i - self.wait_iters[idx])


net_dict['multiple'] = MultipleNet
