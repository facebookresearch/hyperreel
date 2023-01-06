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

from .tensorf_base import TensorVMSplit

from utils.tensorf_utils import (
    raw2alpha,
    alpha2weights,
    scale_shift_color_all,
    scale_shift_color_one,
    transform_color_all,
    transform_color_one
)

from utils.ray_utils import dot


class TensorVMNoSample(TensorVMSplit):
    __constants__ = ["density_plane"]

    def __init__(self, in_channels, out_channels, cfg, **kwargs):
        super().__init__(in_channels, out_channels, cfg, **kwargs)
    
        if kwargs['system'].cfg.dataset.collection in ["bulldozer"]:
            self.black_bg = 1

        if kwargs['system'].cfg.dataset.name in ["blender"]:
            self.white_bg = 1

    def compute_densityfeature(self, xyz_sampled):
        coordinate_plane = torch.stack(
            (
                xyz_sampled[:, self.matMode[0]],
                xyz_sampled[:, self.matMode[1]],
                xyz_sampled[:, self.matMode[2]],
            )
        ).view(3, -1, 1, 2)
        coordinate_line = torch.stack(
            (
                xyz_sampled[:, self.vecMode[0]],
                xyz_sampled[:, self.vecMode[1]],
                xyz_sampled[:, self.vecMode[2]],
            )
        )
        coordinate_line = torch.stack(
            (torch.zeros_like(coordinate_line), coordinate_line), dim=-1
        ).view(3, -1, 1, 2)
        sigma_feature = torch.zeros((xyz_sampled.shape[0],), device=xyz_sampled.device)

        for idx_plane, (plane, line) in enumerate(
            zip(self.density_plane, self.density_line)
        ):
            plane_coef_point = F.grid_sample(
                plane, coordinate_plane[[idx_plane]], align_corners=True
            ).view(-1, xyz_sampled.shape[0])
            line_coef_point = F.grid_sample(
                line, coordinate_line[[idx_plane]], align_corners=True
            ).view(-1, xyz_sampled.shape[0])
            sigma_feature = sigma_feature + torch.sum(
                plane_coef_point * line_coef_point, dim=0
            )

        return sigma_feature

    def feature2density(self, density_features):
        if self.fea2denseAct == "softplus":
            return F.softplus(density_features + self.density_shift)
        elif self.fea2denseAct == "relu":
            return F.relu(density_features)
        elif self.fea2denseAct == "relu_abs":
            return F.relu(torch.abs(density_features))

    def compute_appfeature(self, xyz_sampled):
        # plane + line basis
        coordinate_plane = torch.stack(
            (
                xyz_sampled[:, self.matMode[0]],
                xyz_sampled[:, self.matMode[1]],
                xyz_sampled[:, self.matMode[2]],
            )
        ).view(3, -1, 1, 2)
        coordinate_line = torch.stack(
            (
                xyz_sampled[:, self.vecMode[0]],
                xyz_sampled[:, self.vecMode[1]],
                xyz_sampled[:, self.vecMode[2]],
            )
        )
        coordinate_line = torch.stack(
            (torch.zeros_like(coordinate_line), coordinate_line), dim=-1
        ).view(3, -1, 1, 2)

        plane_coef_point, line_coef_point = [], []
        for idx_plane, (plane, line) in enumerate(zip(self.app_plane, self.app_line)):
            plane_coef_point.append(
                F.grid_sample(
                    plane, coordinate_plane[[idx_plane]], align_corners=True
                ).view(-1, xyz_sampled.shape[0])
            )
            line_coef_point.append(
                F.grid_sample(
                    line, coordinate_line[[idx_plane]], align_corners=True
                ).view(-1, xyz_sampled.shape[0])
            )

        plane_coef_point, line_coef_point = torch.cat(plane_coef_point), torch.cat(
            line_coef_point
        )
        return self.basis_mat((plane_coef_point * line_coef_point).T)

    def forward(self, x: Dict[str, torch.Tensor], render_kwargs: Dict[str, str]):
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

        # Viewdirs
        viewdirs = x["viewdirs"].view(batch_size, nSamples, 3)

        # Weights
        weights = x["weights"].view(batch_size, -1, 1)

        if "weights_shift" in x:
            weights_shift = x["weights_shift"].view(batch_size, -1, 1)

        # Mask out
        ray_valid = self.valid_mask(xyz_sampled) & (distances > 0)

        # Filter
        if self.apply_filter_weights and self.cur_iter >= self.filter_wait_iters:
            weights = weights.view(batch_size, -1)
            min_weight = torch.topk(weights, self.filter_max_samples, dim=-1, sorted=False)[0].min(-1)[0].unsqueeze(-1)

            ray_valid = ray_valid \
                & (weights >= (min_weight - 1e-8)) \
                & (weights > self.filter_weight_thresh)

            weights = weights.view(batch_size, -1, 1)
        else:
            pass
    
        if self.alphaMask is not None and False:
        #if self.alphaMask is not None:
            alphas = self.alphaMask.sample_alpha(xyz_sampled[ray_valid])
            alpha_mask = alphas > 0
            ray_invalid = ~ray_valid
            ray_invalid[ray_valid] |= ~alpha_mask
            ray_valid = ~ray_invalid

        # Get densities
        xyz_sampled = self.normalize_coord(xyz_sampled)
        sigma = xyz_sampled.new_zeros(xyz_sampled.shape[:-1], device=xyz_sampled.device)

        if ray_valid.any():
            sigma_feature = self.compute_densityfeature(xyz_sampled[ray_valid])

            # Convert to density
            sigma_feature = sigma_feature * weights[ray_valid].view(sigma_feature.shape[0])

            if "weights_shift" in x:
                sigma_feature = sigma_feature + weights_shift[ray_valid].view(sigma_feature.shape[0])

            valid_sigma = self.feature2density(sigma_feature)

            # Update valid
            assert valid_sigma is not None
            assert ray_valid is not None

            sigma[ray_valid] = valid_sigma

        alpha, weight, bg_weight = raw2alpha(sigma, deltas * self.distance_scale)
        app_mask = weight > self.rayMarch_weight_thres

        # Get colors
        rgb = xyz_sampled.new_zeros(
            (xyz_sampled.shape[0], xyz_sampled.shape[1], 3), device=xyz_sampled.device
        )

        if app_mask.any():
            app_features = self.compute_appfeature(xyz_sampled[app_mask])

            valid_rgbs = self.renderModule(
                xyz_sampled[app_mask], viewdirs[app_mask], app_features, {}
            )
            rgb = valid_rgbs.new_zeros( # TODO: maybe remove
                (xyz_sampled.shape[0], xyz_sampled.shape[1], 3), device=xyz_sampled.device
            )
            assert valid_rgbs is not None
            assert app_mask is not None
            rgb[app_mask] = valid_rgbs

        # Transform colors
        if 'color_scale' in x:
            color_scale = x['color_scale'].view(rgb.shape[0], rgb.shape[1], 3)
            color_shift = x['color_shift'].view(rgb.shape[0], rgb.shape[1], 3)
            rgb = scale_shift_color_all(rgb, color_scale, color_shift)
        elif 'color_transform' in x:
            color_transform = x['color_transform'].view(rgb.shape[0], rgb.shape[1], 9)
            color_shift = x['color_shift'].view(rgb.shape[0], rgb.shape[1], 3)
            rgb = transform_color_all(rgb, color_transform, color_shift)

        # Over composite
        acc_map = torch.sum(weight, -1)
        rgb_map = torch.sum(weight[:, :, None] * rgb, -2)

        # White background
        if (self.white_bg or (self.training and torch.rand((1,)) < 0.5)) and not self.black_bg:
            rgb_map = rgb_map + (1.0 - acc_map[:, None])

        # Transform colors
        if 'color_scale_global' in x:
            rgb_map = scale_shift_color_one(rgb, rgb_map, x)
        elif 'color_transform_global' in x:
            rgb_map = transform_color_one(rgb, rgb_map, x)

        # Clamp and return
        if not self.training:
            rgb_map = rgb_map.clamp(0, 1)

        # Other fields
        outputs = {
            "rgb": rgb_map
        }

        fields = render_kwargs.get("fields", [])
        no_over_fields = render_kwargs.get("no_over_fields", [])
        pred_weights_fields = render_kwargs.get("pred_weights_fields", [])

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


tensorf_no_sample_dict = {
    "tensor_vm_split_no_sample": TensorVMNoSample,
}
