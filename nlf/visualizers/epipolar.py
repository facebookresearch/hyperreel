#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch

from .base import BaseVisualizer

from utils.ray_utils import get_epi_rays
from utils.visualization import (
    get_warp_dimensions,
    visualize_warp
)
from datasets.lightfield import LightfieldDataset
from datasets.base import Base6DDataset


class EPIVisualizer(BaseVisualizer):
    def __init__(
        self,
        system,
        cfg
    ):
        super().__init__(system, cfg)

        # Setup parametrization
        if system.is_subdivided:
            param_cfg = system.cfg.model.ray.param
        else:
            param_cfg = system.cfg.model.param

        # Vars
        self.v = cfg.v if 'v' in cfg else None
        self.t = cfg.v if 't' in cfg else None
        self.H = cfg.H if 'H' in cfg else None

        self.near = cfg.near if 'near' in cfg else -1.0
        self.far = cfg.far if 'far' in cfg else 0.0

        if self.near is None:
            if 'near' in param_cfg:
                self.near = param_cfg.near
            else:
                self.near = -1.0

        if self.far is None:
            if 'far' in param_cfg:
                self.far = param_cfg.far
            else:
                self.far = 0.0

        if 'st_scale' in cfg and cfg.st_scale is not None:
            self.st_scale = cfg.st_scale
        elif 'lightfield' in system.cfg.dataset and 'st_scale' in system.cfg.dataset.lightfield:
            self.st_scale = system.cfg.dataset.lightfield.st_scale
        else:
            self.st_scale = 1.0

        if 'uv_scale' in cfg and cfg.uv_scale is not None:
            self.uv_scale = cfg.uv_scale
        elif 'lightfield' in system.cfg.dataset and 'uv_scale' in system.cfg.dataset.lightfield:
            self.uv_scale = system.cfg.dataset.lightfield.uv_scale
        else:
            self.uv_scale = 1.0

    def validation(self, batch, batch_idx):
        if batch_idx > 0:
            return

        system = self.get_system()
        dataset = system.trainer.datamodule.train_dataset
        W = system.cur_wh[0]
        H = system.cur_wh[1]

        # Coordinates
        if self.t is not None:
            t = self.t
        else:
            t = 0

        if self.v is not None:
            v = self.v
        else:
            v = 0

        if self.H is not None:
            H = self.H

        ## Forward
        outputs = {}

        # Ground truth EPI
        if isinstance(dataset, LightfieldDataset) and dataset.keyframe_subsample == 1:
            all_rgb = dataset.all_rgb.view(
                dataset.num_rows, dataset.num_cols, dataset.img_wh[1], dataset.img_wh[0], 3
            )
            rgb = all_rgb[dataset.num_rows // 2, :, dataset.img_wh[1] // 2, :, :]

            rgb = rgb.view(rgb.shape[0], rgb.shape[1], 3).cpu()
            rgb = rgb.permute(2, 0, 1)
            outputs['gt'] = rgb

        # Generate EPI rays
        rays = get_epi_rays(
            W, v, H, t, dataset.aspect,
            st_scale=self.st_scale,
            uv_scale=self.uv_scale,
            near=self.near, far=self.far
        ).type_as(batch['coords'])

        # Add time
        if isinstance(dataset, Base6DDataset):
            rays = torch.cat([rays, torch.zeros_like(rays[..., :1])], dim=-1)

        # RGB
        rgb = system(rays)['rgb']

        if isinstance(rgb, list):
            rgb = rgb[-1]

        rgb = rgb.view(H, W, 3).cpu()
        rgb = rgb.permute(2, 0, 1)

        outputs['pred'] = rgb

        return outputs

    def validation_image(self, batch, batch_idx):
        if batch_idx > 0:
            return {}

        # Outputs
        temp_outputs = self.validation(batch, batch_idx)
        outputs = {}

        for key in temp_outputs.keys():
            outputs[f'images/epi_{key}'] = temp_outputs[key]

        return outputs
