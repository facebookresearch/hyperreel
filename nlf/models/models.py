#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import numpy as np

from typing import Dict, List
from torch import nn

from nlf.param import RayParam

from nlf.embedding import (
    embedding_dict,
)

from nlf.nets import net_dict

from nlf.activations import get_activation


class BaseColorModel(nn.Module):
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
        self.color_channels = out_channels

        # Total out channels
        self.out_channels = self.color_channels

        # MLP
        self.net = net_dict[cfg.net.type](
            self.in_channels,
            self.out_channels,
            cfg.net,
            group=self.group,
            system=kwargs['system']
        )

    def forward(self, x: Dict[str, torch.Tensor], render_kwargs: Dict[str, str]):
        return self.net(x, render_kwargs)

    def set_iter(self, i):
        self.cur_iter = i
        self.net.set_iter(i)


color_model_dict = {
    'base': BaseColorModel,
}


ray_model_dict = {}
pos_model_dict = {}
model_dict = {}


class BaseLightfieldModel(nn.Module):
    def __init__(
        self,
        cfg,
        **kwargs
    ):
        super().__init__()

        self.cfg = cfg
        self.embeddings = []
        self.models = []

        if 'is_subdivided' in kwargs:
            self.is_subdivided = kwargs['is_subdivided']
        else:
            self.is_subdivided = False

        if 'num_outputs' in kwargs:
            self.num_outputs = kwargs['num_outputs']
        else:
            self.num_outputs = cfg.num_outputs if 'num_outputs' in cfg else 3

        # Ray parameterization
        self.param = RayParam(cfg.param)

    def set_iter(self, i):
        self.cur_iter = i

        for emb in self.embeddings:
            emb.set_iter(i)

        for model in self.models:
            model.set_iter(i)


class LightfieldModel(BaseLightfieldModel):
    def __init__(
        self,
        cfg,
        **kwargs
    ):
        super().__init__(cfg, **kwargs)

        # Embedding
        self.embedding_model = embedding_dict[cfg.embedding.type](
            self.param.out_channels,
            cfg.embedding,
            system=kwargs['system']
        )

        self.embeddings += [self.embedding_model]

        # Color
        self.color_model = color_model_dict[cfg.color.type](
            self.embedding_model.out_channels,
            self.num_outputs,
            cfg.color,
            system=kwargs['system']
        )

        self.models += [self.color_model]

    def embed(self, rays: torch.Tensor, render_kwargs: Dict[str, str]):
        param_rays = self.param(rays)
        return self.embedding_model(param_rays, render_kwargs)

    def forward(self, rays, render_kwargs: Dict[str, str]):
        embed_rays = self.embed(rays, render_kwargs)
        outputs = self.color_model(embed_rays, render_kwargs)
        return outputs


ray_model_dict['lightfield'] = LightfieldModel
pos_model_dict['lightfield'] = LightfieldModel
model_dict['lightfield'] = LightfieldModel
