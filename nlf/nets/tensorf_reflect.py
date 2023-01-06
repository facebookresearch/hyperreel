#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright (c) 2022 Anpei Chen
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import time

from typing import Dict

import numpy as np
import torch
import torch.nn
import torch.nn.functional as F
from torch import autograd

from utils.sh_utils import eval_sh_bases
from utils.tensorf_utils import cal_n_samples, N_to_reso

from .tensorf_no_sample import TensorVMNoSample

from utils.tensorf_utils import (
    raw2alpha,
    alpha2weights
)

class TensorVMReflect(TensorVMNoSample):
    __constants__ = ["density_plane"]

    def __init__(self, in_channels, out_channels, cfg, **kwargs):
        super().__init__(in_channels, out_channels, cfg, **kwargs)

    def compute_density_normal(
        self,
        points,
        weights
    ):
        has_grad = torch.is_grad_enabled()
        points = points.view(-1, 3)

        # Calculate gradient with respect to points
        with torch.enable_grad():
            points = points.requires_grad_(True)

            density = super().compute_densityfeature(points)
            density = density * weights.view(density.shape[0])
            density = self.feature2density(density)

            normal = -autograd.grad(
                density,
                points,
                torch.ones_like(density, device=points.device),
                create_graph=has_grad,
                retain_graph=has_grad,
                only_inputs=True
            )[0]

        return density, torch.nn.functional.normalize(normal, dim=-1)

    def forward(self, x: Dict[str, torch.Tensor], render_kwargs: Dict[str, str]):
        fields = render_kwargs.get("fields", [])
        no_over_fields = render_kwargs.get("no_over_fields", [])
        pred_weights_fields = render_kwargs.get("pred_weights_fields", [])

        # Batch size
        batch_size = x["viewdirs"].shape[0]

        # Positions
        nSamples = x["points"].shape[-1] // 3
        xyz_sampled = x["points"].view(batch_size, nSamples, 3)

        # Distances
        distances = x["distances"].view(batch_size, -1)
        deltas = torch.cat(
            [
                distances[..., 1:] - distances[..., :-1],
                1e10 * torch.ones_like(distances[:, :1]),
            ],
            dim=1,
        )
        #deltas = torch.cat(
        #    [
        #        distances[..., 0:1],
        #        distances[..., 1:] - distances[..., :-1],
        #    ],
        #    dim=1,
        #)

        # Viewdirs
        viewdirs = x["viewdirs"].view(batch_size, nSamples, 3)

        # Weights
        weights = x["weights"].view(batch_size, -1, 1)

        # Mask out
        ray_valid = self.valid_mask(xyz_sampled) & (distances > 0)

        if self.alphaMask is not None and False:
            alphas = self.alphaMask.sample_alpha(xyz_sampled[ray_valid])
            alpha_mask = alphas > 0
            ray_invalid = ~ray_valid
            ray_invalid[ray_valid] |= ~alpha_mask
            ray_valid = ~ray_invalid

        assert ray_valid is not None

        # Get densities
        xyz_sampled = self.normalize_coord(xyz_sampled)
        sigma = xyz_sampled.new_zeros(xyz_sampled.shape[:-1], device=xyz_sampled.device)

        if 'render_normal' in fields:
            render_normal = xyz_sampled.new_zeros(xyz_sampled.shape, device=xyz_sampled.device)

            if ray_valid.any():
                valid_sigma, valid_render_normal = self.compute_density_normal(
                    xyz_sampled[ray_valid],
                    weights[ray_valid]
                )

                # Update valid
                assert valid_render_normal is not None
                assert valid_sigma is not None

                render_normal[ray_valid] = valid_render_normal
                x['render_normal'] = render_normal

                sigma[ray_valid] = valid_sigma
        else:
            if ray_valid.any():
                sigma_feature = self.compute_densityfeature(xyz_sampled[ray_valid])
                sigma_feature = sigma_feature * weights[ray_valid].view(sigma_feature.shape[0])
                valid_sigma = self.feature2density(sigma_feature)

                # Update valid
                assert valid_sigma is not None

                sigma[ray_valid] = valid_sigma

        alpha, weight, bg_weight = raw2alpha(sigma, deltas * self.distance_scale)
        app_mask = weight > self.rayMarch_weight_thres

        #if len(self.update_AlphaMask_list) == 0 or self.cur_iter < self.update_AlphaMask_list[0]:
        #    app_mask = torch.ones_like(app_mask)

        # Get colors
        rgb = xyz_sampled.new_zeros(
            (xyz_sampled.shape[0], xyz_sampled.shape[1], 3), device=xyz_sampled.device
        )

        if app_mask.any():
            app_features = self.compute_appfeature(xyz_sampled[app_mask])
            valid_rgbs = self.renderModule(
                xyz_sampled[app_mask], viewdirs[app_mask], app_features, {}
            )
            assert valid_rgbs is not None
            assert app_mask is not None
            rgb[app_mask] = valid_rgbs

        # Over composite
        acc_map = torch.sum(weight, -1)
        rgb_map = torch.sum(weight[:, :, None] * rgb, -2)

        # White background
        if self.white_bg or (self.training and torch.rand((1,)) < 0.5):
            rgb_map = rgb_map + (1.0 - acc_map[:, None])

        # Clamp and return
        rgb_map = rgb_map.clamp(0, 1)

        # Other fields
        outputs = {
            "rgb": rgb_map
        }

        if len(fields) == 0:
            return outputs

        if len(pred_weights_fields) > 0:
            pred_weights = alpha2weights(weights[..., 0])

        for key in fields:
            if key == 'render_weights':
                outputs[key] = weight
            elif key in no_over_fields:
                outputs[key] = x[key].view(batch_size, -1)
            elif key in pred_weights_fields:
                outputs[key] = torch.sum(
                    pred_weights[..., None] * x[key].view(batch_size, nSamples, -1),
                    -2,
                )
            else:
                outputs[key] = torch.sum(
                    weight[..., None] * x[key].view(batch_size, nSamples, -1),
                    -2,
                )

        return outputs


tensorf_reflect_dict = {
    "tensor_vm_split_reflect": TensorVMReflect,
}

