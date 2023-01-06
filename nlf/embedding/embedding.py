#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn
from typing import Dict


# Identity embedding
class IdentityEmbedding(nn.Module):
    def __init__(
        self,
        in_channels,
        cfg,
        *args,
        **kwargs
    ):

        super().__init__()

        self.group = cfg.group if 'group' in cfg else (kwargs['group'] if 'group' in kwargs else 'embedding')

        self.in_channels = in_channels
        self.out_channels = in_channels
        self.layer = nn.Linear(1, 1)

    def forward(self, x: torch.Tensor, kwargs: Dict[str, str]):
        return x

    def set_iter(self, i):
        self.cur_iter = i


embedding_dict = {
    'identity': IdentityEmbedding,
}

# Add feature embeddings
from .feature import feature_embedding_dict
for k, v in feature_embedding_dict.items(): embedding_dict[k] = v

# Add affine embeddings
from .affine import affine_embedding_dict
for k, v in affine_embedding_dict.items(): embedding_dict[k] = v

# Add ray embeddings
from .ray import ray_embedding_dict
for k, v in ray_embedding_dict.items(): embedding_dict[k] = v

# Add point embeddings
from .point import point_embedding_dict
for k, v in point_embedding_dict.items(): embedding_dict[k] = v


# Ray point embedding
class RayPointEmbedding(nn.Module):
    def __init__(
        self,
        in_channels,
        cfg,
        **kwargs
    ):
        super().__init__()

        self.group = cfg.group if 'group' in cfg else (kwargs['group'] if 'group' in kwargs else 'embedding')
        self.cfg = cfg

        # In, out channels
        self.in_channels = in_channels

        # Track iterations
        self.cur_iter = 0
        self.wait_iters = []
        self.stop_iters = []

        # Create ray and point embeddings
        self.embedding_keys = list(cfg.embeddings.keys())
        self.embeddings = nn.ModuleList()
        in_channels = self.in_channels

        for embedding_key in cfg.embeddings.keys():
            embedding_cfg = cfg.embeddings[embedding_key]
            self.wait_iters.append(embedding_cfg.wait_iters if 'wait_iters' in embedding_cfg else 0)
            self.stop_iters.append(embedding_cfg.stop_iters if 'stop_iters' in embedding_cfg else float("inf"))

            # Create net
            embedding = embedding_dict[embedding_cfg.type](
                in_channels,
                embedding_cfg,
                **kwargs
            )
            self.embeddings.append(embedding)

        # Out channels
        self.out_channels = in_channels

    def forward(self, rays: torch.Tensor, render_kwargs: Dict[str, str]):
        # Forward
        x = {
            'rays': rays,
        }

        for idx, embedding in enumerate(self.embeddings):
            if self.cur_iter >= self.wait_iters[idx] and self.cur_iter < self.stop_iters[idx]:
                x = embedding(
                    x, render_kwargs
                )

        # Flatten
        for key in x.keys():
            x[key] = x[key].view(rays.shape[0], -1)

        # Return
        return x

    def set_iter(self, i):
        self.cur_iter = i

        for idx in range(len(self.embeddings)):
            self.embeddings[idx].set_iter(i - self.wait_iters[idx])


embedding_dict['ray_point'] = RayPointEmbedding
