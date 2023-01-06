#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn
from nlf.nets import net_dict
from nlf.activations import get_activation


class AffineEmbedding(nn.Module):
    def __init__(
        self,
        in_channels,
        cfg,
        **kwargs
    ):
        super().__init__()

        self.group = cfg.group if 'group' in cfg else (kwargs['group'] if 'group' in kwargs else 'embedding')

        self.in_channels = in_channels
        self.out_channels = cfg.out_channels if cfg.out_channels is not None else in_channels
        self.homogenous_layer = nn.Linear(self.in_channels, self.out_channels)

    def forward(self, x):
        return self.homogenous_layer(x)

    def set_iter(self, i):
        self.cur_iter = i


class LocalAffineEmbedding(nn.Module):
    def __init__(
        self,
        in_channels,
        cfg,
        **kwargs
    ):

        super().__init__()

        self.group = cfg.group if 'group' in cfg else (kwargs['group'] if 'group' in kwargs else 'embedding')

        self.cfg = cfg
        self.dummy_layer = nn.Linear(1, 1)

        # Param
        self.param_channels = in_channels if cfg.param_channels == 'all' else cfg.param_channels
        self.in_channels = in_channels
        self.latent_dim = kwargs['latent_dim'] if 'latent_dim' in kwargs else 0

        # Extra
        self.extra_in_channels = cfg.extra_in_channels if 'extra_in_channels' in cfg else 0
        self.extra_out_channels = cfg.extra_out_channels if 'extra_out_channels' in cfg else self.extra_in_channels
        self.extra_tform_size = self.extra_in_channels * self.extra_out_channels

        self.extra_tform_activation = cfg.extra_tform_activation if 'extra_tform_activation' in cfg else 'identity'
        self.extra_bias_activation = cfg.extra_bias_activation if 'extra_bias_activation' in cfg else 'zero'
        self.extra_activation = cfg.extra_activation if 'extra_activation' in cfg else 'identity'

        # Tform
        self.tform_in_channels = self.param_channels
        self.tform_out_channels = cfg.tform_out_channels
        self.tform_size = self.tform_in_channels * self.tform_out_channels

        self.tform_scale = cfg.tform_scale if 'tform_scale' in cfg else 1.0
        self.add_identity = cfg.add_identity if 'add_identity' in cfg else False

        self.tform_activation = cfg.tform_activation if 'tform_activation' in cfg else 'identity'
        self.bias_activation = cfg.bias_activation if 'bias_activation' in cfg else 'zero'
        self.activation = cfg.activation if 'activation' in cfg else 'identity'

        # Tform outputs
        self.total_pred_channels = self.tform_size + self.extra_tform_size
        self.out_channels_after_tform = cfg.tform_out_channels

        if self.bias_activation != 'zero':
            self.total_pred_channels += self.tform_out_channels

        if self.extra_bias_activation != 'zero':
            self.total_pred_channels += self.extra_out_channels

        # Outputs
        self.out_channels = self.extra_out_channels + self.out_channels_after_tform

        # Net
        if 'depth' in cfg.net:
            cfg.net['depth'] -= 2
            cfg.net['linear_last'] = False

        self.net = net_dict[cfg.net.type](
            self.in_channels,
            self.total_pred_channels,
            cfg.net,
            latent_dim=self.latent_dim,
            group=self.group
        )

        # Out
        self.out_extra_tform_layer = get_activation(self.extra_tform_activation)
        self.out_extra_bias_layer = get_activation(self.extra_bias_activation)

        self.out_tform_layer = get_activation(self.tform_activation)
        self.out_bias_layer = get_activation(self.bias_activation)

        self.out_extra_layer = get_activation(self.extra_activation)
        self.out_layer = get_activation(self.activation)

    def embed_params(self, x, **render_kwargs):
        if 'input_x' not in render_kwargs or render_kwargs['input_x'] is None:
            input_x = x
        else:
            input_x = render_kwargs['input_x']

        _, _, tform_flat, _ = self._embed_params(input_x)

        return tform_flat

    def _embed_params(self, x):
        # MLP
        x = self.net(x)

        # Outputs
        extra_tform_flat = self.out_extra_tform_layer(
            x[..., :self.extra_tform_size]
        )
        x = x[..., self.extra_tform_size:]

        if self.extra_bias_activation != 'zero':
            extra_bias = self.out_extra_bias_layer(
                x[..., :self.extra_out_channels]
            )
            x = x[..., self.extra_out_channels:]
        else:
            extra_bias = None

        if self.bias_activation == 'zero':
            tform_flat = self.out_tform_layer(x)
            bias = None
        else:
            tform_flat = self.out_tform_layer(x[..., :-self.out_channels_after_tform])
            bias = self.out_bias_layer(
                x[..., -self.out_channels_after_tform:]
            )

        return extra_tform_flat, extra_bias, tform_flat, bias

    def forward(self, x, **render_kwargs):
        batch_size = x.shape[0]

        if 'input_x' not in render_kwargs or render_kwargs['input_x'] is None:
            input_x = x
        else:
            input_x = render_kwargs['input_x']

        extra_tform, extra_bias, tform, bias = self._embed_params(
            input_x
        )

        # Extra channel transform
        extra_x = x[..., :self.extra_in_channels]

        if self.extra_tform_size > 0:
            extra_tform = extra_tform.view(
                -1, self.extra_out_channels, self.extra_in_channels
            )
            extra_x = (extra_tform @ extra_x.unsqueeze(-1)).squeeze(-1)

            if extra_bias is not None:
                extra_x = extra_x + extra_bias

        if self.extra_in_channels > 0:
            extra_x = extra_x.view(batch_size, -1)
            extra_x = self.out_extra_layer(extra_x)

        # Get transform
        if self.add_identity:
            tform = tform.reshape(
                -1, self.out_channels_after_tform, self.param_channels
            )
            tform = tform * self.tform_scale + torch.eye(
                self.out_channels_after_tform, self.param_channels, device=tform.device
            )

        tform = tform.view(
            -1, self.out_channels_after_tform, self.param_channels
        )

        # Apply transform
        x = x[..., :self.param_channels]
        x = self.out_layer((tform @ x.unsqueeze(-1)).squeeze(-1))

        # Add bias
        if bias is not None:
            x = x + bias

        # Return
        x = torch.cat([extra_x, x], -1)

        if 'embed_params' in render_kwargs and render_kwargs['embed_params']:
            if bias is not None:
                return torch.cat([tform.view(tform.shape[0], -1), bias], -1), x
            else:
                return tform.view(tform.shape[0], -1), x
        else:
            return x

    def set_iter(self, i):
        self.cur_iter = i
        self.net.set_iter(i)


affine_embedding_dict = {
    'affine': AffineEmbedding,
    'local_affine': LocalAffineEmbedding,
}
