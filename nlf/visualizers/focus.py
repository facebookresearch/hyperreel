#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch

from .base import BaseVisualizer
from utils.ray_utils import get_lightfield_rays


class FocusVisualizer(BaseVisualizer):
    def __init__(
        self,
        system,
        cfg
    ):
        super().__init__(system, cfg)

        self.s = cfg.s if 's' in cfg else None
        self.t = cfg.t if 't' in cfg else None

        self.ds = cfg.ds if 'ds' in cfg else None
        self.dt = cfg.dt if 'dt' in cfg else None

        self.near = cfg.near if 'near' in cfg else -1.0
        self.far = cfg.far if 'far' in cfg else 0.0
        self.focal = cfg.focal if 'focal' in cfg else 0.0

        self.n_freqs = system.cfg.model.color_pe.n_freqs
        self.freq_bands = 2 ** torch.linspace(0, self.n_freqs - 1, self.n_freqs)

        if 'st_scale' in cfg and cfg.st_scale is not None:
            self.st_scale = cfg.st_scale
        elif 'lightfield' in system.cfg.dataset and 'st_scale' in system.cfg.dataset.lightfield:
            self.st_scale = system.cfg.dataset.lightfield.st_scale
        else:
            self.st_scale = 1.0

        if 'uv_scale' in cfg and cfg.uv_scale is not None:
            self.uv_scale = cfg.uv_scale
        else:
            self.uv_scale = 1.0

    def validation(self, batch, batch_idx):
        if batch_idx > 0:
            return

        system = self.get_system()
        dataset = system.trainer.datamodule.train_dataset

        W = system.cur_wh[0]
        H = system.cur_wh[1]

        param_channels = 4

        ## Forward
        outputs = {}

        # Generate image rays
        if self.s is not None:
            s = self.s
        else:
            s = 0.0

        if self.t is not None:
            t = self.t
        else:
            t = 0.0

        rays = get_lightfield_rays(
            W, H, s, t, dataset.aspect,
            st_scale=self.st_scale,
            uv_scale=self.uv_scale,
            near=self.near, far=self.far
        ).type_as(batch['coords'])

        # Cone
        if self.ds is None:
            ds = 1.0
        else:
            ds = self.ds

        if self.dt is None:
            dt = 1.0
        else:
            dt = self.dt

        du = (self.focal - self.far) * ds / (self.far - self.near)
        dv = (self.focal - self.far) * dt / (self.far - self.near)

        ds_vec = torch.zeros((1, param_channels, 1)).type_as(rays)
        ds_vec[..., 0, :] = ds
        ds_vec[..., 2, :] = du

        dt_vec = torch.zeros((1, param_channels, 1)).type_as(rays)
        dt_vec[..., 1, :] = dt
        dt_vec[..., 3, :] = dv

        # Warp
        params = system.render('embed_params', rays)['params']
        out_channels = (params.shape[-1] // (param_channels + 1))
        tform = params[..., :-out_channels].reshape(
            -1, out_channels, param_channels
        )
        tform = torch.nn.functional.normalize(tform, p=2.0, dim=-1)

        s_response = (tform @ ds_vec).squeeze(-1) / W
        t_response = (tform @ dt_vec).squeeze(-1) / W

        max_response = torch.maximum(
            torch.abs(s_response),
            torch.abs(t_response),
        )
        max_freq = 1.0 / max_response

        # Calculate weights
        pe_weight = {}

        for j, freq in enumerate(self.freq_bands):
            #weight = (max_freq / freq - 0.5) * 2
            weight = max_freq / freq
            pe_weight[j] = torch.clamp(weight, torch.zeros_like(weight), torch.ones_like(weight))

        print(pe_weight[0].shape, rays.shape)
        print(pe_weight[len(self.freq_bands) - 1][0])

        # RGB out of focus
        rgb = system(rays, pe_weight=pe_weight, chunk_args=['pe_weight'])['rgb']
        if isinstance(rgb, list):
            rgb = rgb[-1]
        rgb = rgb.view(H, W, 3).cpu()
        rgb = rgb.permute(2, 0, 1)

        outputs['rgb_cone'] = rgb

        # RGB in focus
        rgb = system(rays)['rgb']
        if isinstance(rgb, list):
            rgb = rgb[-1]
        rgb = rgb.view(H, W, 3).cpu()
        rgb = rgb.permute(2, 0, 1)

        outputs['rgb_ray'] = rgb

        return outputs

    def validation_image(self, batch, batch_idx):
        if batch_idx > 0:
            return {}

        # Outputs
        temp_outputs = self.validation(batch, batch_idx)
        outputs = {}

        for key in temp_outputs.keys():
            outputs[f'images/focus_{key}'] = temp_outputs[key]

        return outputs
