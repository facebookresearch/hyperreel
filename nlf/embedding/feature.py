#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from torch import nn
from nlf.nets import net_dict


class FeatureEmbedding(nn.Module):
    def __init__(
        self,
        in_channels,
        cfg,
        **kwargs
    ):
        super().__init__()

        self.group = cfg.group if 'group' in cfg else (kwargs['group'] if 'group' in kwargs else 'embedding')

        self.cfg = cfg

        # Outputs
        if self.cfg.net.depth == 0:
            self.out_channels = in_channels
        else:
            self.out_channels = cfg.out_channels

        # Net
        if 'depth' in cfg.net:
            cfg.net['depth'] -= 2
            cfg.net['linear_last'] = False

        self.net = net_dict[cfg.net.type](
            self.in_channels,
            self.out_channels,
            cfg.net
        )

    def forward(self, x, **render_kwargs):
        if self.cfg.net.depth == 0:
            return x
        else:
            return self.net(x)

    def set_iter(self, i):
        self.cur_iter = i


feature_embedding_dict = {
    'feature': FeatureEmbedding
}
