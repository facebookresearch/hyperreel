#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright (c) 2022 Anpei Chen
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import time

from typing import Dict

import cv2
import numpy as np
import torch
import torch.nn
import torch.nn.functional as F
from torch import autograd

from utils.sh_utils import eval_sh_bases
from utils.tensorf_utils import cal_n_samples, N_to_reso

from .tensorf_base import TensorBase

from utils.tensorf_utils import (
    raw2alpha,
    alpha2weights,
    scale_shift_color_all,
    scale_shift_color_one,
    transform_color_all,
    transform_color_one,
    DensityRender,
    DensityLinearRender,
    DensityFourierRender,
    RGBIdentityRender,
    RGBtFourierRender,
    RGBtLinearRender
)

from utils.intersect_utils import (
    sort_z,
    sort_with
)


class TensorVMKeyframeTime(TensorBase):
    def __init__(self, in_channels, out_channels, cfg, **kwargs):
        self.matModeSpace = [[0, 1], [0, 2], [1, 2]]
        self.matModeTime = [[2, 3], [1, 3], [0, 3]]
        self.num_keyframes = kwargs["system"].dm.train_dataset.num_keyframes
        self.total_num_frames = kwargs["system"].dm.train_dataset.num_frames
        self.frames_per_keyframe = (
            cfg.frames_per_keyframe
            if "frames_per_keyframe" in cfg
            else self.total_num_frames // self.num_keyframes
        )
        # self.time_scale_factor = ((self.num_keyframes) / (self.num_keyframes - 1)) * (self.total_num_frames - 1) / self.total_num_frames
        # self.time_pixel_offset = 0.0 / self.num_keyframes
        self.time_scale_factor = (self.total_num_frames - 1) / self.total_num_frames
        self.time_pixel_offset = 0.5 / self.num_keyframes

        # Number of outputs for color and density
        if cfg.shadingMode == "RGBtLinear":
            cfg.data_dim_color = (2) * 3
        elif cfg.shadingMode == "RGBtFourier":
            cfg.data_dim_color = (self.frames_per_keyframe * 2 + 1) * 3

        self.densityMode = cfg.densityMode

        if self.densityMode == "Density":
            self.data_dim_density = 1
        elif self.densityMode == "DensityLinear":
            self.data_dim_density = 2
        elif self.densityMode == "DensityFourier":
            self.data_dim_density = self.frames_per_keyframe * 2 + 1

        super().__init__(in_channels, out_channels, cfg, **kwargs)

        if "MLP" in self.shadingMode:
            self.opt_group = {
                "color": [
                    self.density_plane_space,
                    self.density_plane_time,
                    self.app_plane_space,
                    self.app_plane_time,
                ],
                "color_impl": [self.basis_mat, self.basis_mat_density],
            }
        else:
            self.opt_group = {
                "color": [
                    self.density_plane_space,
                    self.density_plane_time,
                    self.app_plane_space,
                    self.app_plane_time,
                    self.basis_mat,
                    self.basis_mat_density,
                ],
            }

        if isinstance(self.renderModule, torch.nn.Module):
            if "MLP" in self.shadingMode:
                self.opt_group["color_impl"] += [self.renderModule]
            else:
                self.opt_group["color_impl"] = [self.renderModule]

    def init_svd_volume(self, res, device):
        if self.fea2denseAct == 'softplus':
            self.density_plane_space, self.density_plane_time = self.init_one_svd_density(
                self.density_n_comp, self.gridSize, self.num_keyframes, 0.1, device
            )
        else:
            self.density_plane_space, self.density_plane_time = self.init_one_svd_density(
                self.density_n_comp, self.gridSize, self.num_keyframes, 1e-2, device
            )

        self.app_plane_space, self.app_plane_time = self.init_one_svd(
            self.app_n_comp, self.gridSize, self.num_keyframes, 0.1, device
        )
        self.basis_mat = torch.nn.Linear(
            sum(self.app_n_comp), self.app_dim, bias=False
        ).to(device)
        self.basis_mat_density = torch.nn.Linear(
            sum(self.density_n_comp), self.data_dim_density, bias=False
        ).to(device)

    def init_one_svd(self, n_component, gridSize, numFrames, scale, device):
        plane_coef_space, plane_coef_time = [], []

        for i in range(len(self.matModeSpace)):
            mat_id_space_0, mat_id_space_1 = self.matModeSpace[i]
            mat_id_time_0, mat_id_time_1 = self.matModeTime[i]

            if n_component[i] == 0 and False:
                plane_coef_space.append(
                    torch.nn.Parameter(
                        torch.zeros(
                            (1, n_component[i], self.gridSizeStart[mat_id_space_1], self.gridSizeStart[mat_id_space_0])
                        )
                    )
                )
                plane_coef_time.append(
                    torch.nn.Parameter(
                        torch.zeros(
                            (1, n_component[i], numFrames, self.gridSizeStart[mat_id_time_0])
                        )
                    )
                )
            else:
                plane_coef_space.append(
                    torch.nn.Parameter(
                        scale
                        * torch.randn(
                            (
                                1,
                                n_component[i],
                                gridSize[mat_id_space_1],
                                gridSize[mat_id_space_0],
                            )
                        )
                    )
                )
                plane_coef_time.append(
                    torch.nn.Parameter(
                        scale
                        * torch.randn(
                            (1, n_component[i], numFrames, gridSize[mat_id_time_0])
                        )
                    )
                )

        return torch.nn.ParameterList(plane_coef_space).to(
            device
        ), torch.nn.ParameterList(plane_coef_time).to(device)

    def init_one_svd_density(self, n_component, gridSize, numFrames, scale, device):
        plane_coef_space, plane_coef_time = [], []

        for i in range(len(self.matModeSpace)):
            mat_id_space_0, mat_id_space_1 = self.matModeSpace[i]
            mat_id_time_0, mat_id_time_1 = self.matModeTime[i]

            if n_component[i] == 0 and False:
                plane_coef_space.append(
                    torch.nn.Parameter(
                        torch.zeros(
                            (1, n_component[i], self.gridSizeStart[mat_id_space_1], self.gridSizeStart[mat_id_space_0])
                        )
                    )
                )
                plane_coef_time.append(
                    torch.nn.Parameter(
                        torch.zeros(
                            (1, n_component[i], numFrames, self.gridSizeStart[mat_id_time_0])
                        )
                    )
                )
            elif self.fea2denseAct == 'softplus':
                plane_coef_space.append(
                    torch.nn.Parameter(
                        scale
                        * torch.randn(
                            (
                                1,
                                n_component[i],
                                gridSize[mat_id_space_1],
                                gridSize[mat_id_space_0],
                            )
                        )
                    )
                )
                plane_coef_time.append(
                    torch.nn.Parameter(
                        scale
                        * torch.randn(
                            (1, n_component[i], numFrames, gridSize[mat_id_time_0])
                        )
                    )
                )
            else:
                plane_coef_space.append(
                    torch.nn.Parameter(
                        scale
                        * torch.rand(
                            (
                                1,
                                n_component[i],
                                gridSize[mat_id_space_1],
                                gridSize[mat_id_space_0],
                            )
                        ).clamp(1e-2, 1e8)
                    )
                )
                plane_coef_time.append(
                    torch.nn.Parameter(
                        scale
                        * torch.rand(
                            (1, n_component[i], numFrames, gridSize[mat_id_time_0])
                        ).clamp(1e-2, 1e8)
                    )
                )

        return torch.nn.ParameterList(plane_coef_space).to(
            device
        ), torch.nn.ParameterList(plane_coef_time).to(device)

    def density_L1(self):
        total = 0

        for idx in range(len(self.density_plane_space)):
            if self.density_plane_space[idx].shape[1] == 0:
                continue

            total = (
                total
                + torch.mean(torch.abs(self.density_plane_space[idx]))
                + torch.mean(torch.abs(self.density_plane_time[idx]))
            )

        return total

    def TV_loss_density(self, reg):
        total = 0

        for idx in range(len(self.density_plane_space)):
            if self.density_plane_space[idx].shape[1] == 0 or self.density_plane_time[idx].shape[1] == 0:
                continue

            total = (
                total + reg(self.density_plane_space[idx]) * 1e-2
            )# + reg(self.density_plane_time[idx]) * 1e-2

        return total

    def TV_loss_app(self, reg):
        total = 0

        for idx in range(len(self.app_plane_space)):
            if self.density_plane_space[idx].shape[1] == 0 or self.density_plane_time[idx].shape[1] == 0:
                continue

            total = (
                total + reg(self.app_plane_space[idx]) * 1e-2
            )# + reg(self.app_plane_time[idx]) * 1e-2

        return total

    def compute_densityfeature(self, xyz_sampled):
        # plane + line basis
        coordinate_plane_space = torch.stack(
            (
                xyz_sampled[:, self.matModeSpace[0]],
                xyz_sampled[:, self.matModeSpace[1]],
                xyz_sampled[:, self.matModeSpace[2]],
            )
        ).view(3, -1, 1, 2)

        coordinate_plane_time = torch.stack(
            (
                xyz_sampled[:, self.matModeTime[0]],
                xyz_sampled[:, self.matModeTime[1]],
                xyz_sampled[:, self.matModeTime[2]],
            )
        ).view(3, -1, 1, 2)

        plane_coef_space, plane_coef_time = [], []

        for idx_plane, (plane_space, plane_time) in enumerate(
            zip(self.density_plane_space, self.density_plane_time)
        ):
            if self.density_plane_space[idx_plane].shape[1] == 0:
                continue

            cur_plane = F.grid_sample(
                plane_space, coordinate_plane_space[[idx_plane]], align_corners=True
            ).view(-1, xyz_sampled.shape[0])
            cur_time = F.grid_sample(
                plane_time, coordinate_plane_time[[idx_plane]], align_corners=True
            ).view(-1, xyz_sampled.shape[0])

            plane_coef_space.append(cur_plane)
            plane_coef_time.append(cur_time)

        plane_coef_space, plane_coef_time = torch.cat(plane_coef_space), torch.cat(
            plane_coef_time
        )

        if self.densityMode != "Density":
            return self.basis_mat_density((plane_coef_space * plane_coef_time).T)
        else:
            return torch.sum((plane_coef_space * plane_coef_time), dim=0).unsqueeze(-1)

    def compute_appfeature(self, xyz_sampled):
        # plane + line basis
        coordinate_plane_space = torch.stack(
            (
                xyz_sampled[:, self.matModeSpace[0]],
                xyz_sampled[:, self.matModeSpace[1]],
                xyz_sampled[:, self.matModeSpace[2]],
            )
        ).view(3, -1, 1, 2)

        coordinate_plane_time = torch.stack(
            (
                xyz_sampled[:, self.matModeTime[0]],
                xyz_sampled[:, self.matModeTime[1]],
                xyz_sampled[:, self.matModeTime[2]],
            )
        ).view(3, -1, 1, 2)

        plane_coef_space, plane_coef_time = [], []

        for idx_plane, (plane_space, plane_time) in enumerate(
            zip(self.app_plane_space, self.app_plane_time)
        ):
            if self.density_plane_space[idx_plane].shape[1] == 0:
                continue

            cur_plane = F.grid_sample(
                plane_space, coordinate_plane_space[[idx_plane]], align_corners=True
            ).view(-1, *xyz_sampled.shape[:1])
            cur_time = F.grid_sample(
                plane_time, coordinate_plane_time[[idx_plane]], align_corners=True
            ).view(-1, *xyz_sampled.shape[:1])

            plane_coef_space.append(cur_plane)
            plane_coef_time.append(cur_time)

        plane_coef_space, plane_coef_time = torch.cat(plane_coef_space), torch.cat(
            plane_coef_time
        )
        return self.basis_mat((plane_coef_space * plane_coef_time).T)

    def feature2density(
        self, density_features: torch.Tensor, x: Dict[str, torch.Tensor]
    ):
        if self.densityMode == "Density":
            density_features = DensityRender(density_features, x)
        elif self.densityMode == "DensityLinear":
            density_features = DensityLinearRender(density_features, x)
        elif self.densityMode == "DensityFourier":
            density_features = DensityFourierRender(density_features, x)

        density_features = density_features * x["weights"].view(
            density_features.shape[0]
        )

        if self.fea2denseAct == "softplus":
            return F.softplus(density_features + self.density_shift)
        elif self.fea2denseAct == "relu":
            return F.relu(density_features)
        elif self.fea2denseAct == "relu_abs":
            return F.relu(torch.abs(density_features))

    @torch.no_grad()
    def up_sampling_VM(self, n_component, plane_coef_space, plane_coef_time, res_target, numFrames):

        for i in range(len(self.matModeSpace)):
            mat_id_space_0, mat_id_space_1 = self.matModeSpace[i]
            mat_id_time_0, mat_id_time_1 = self.matModeTime[i]

            if self.density_plane_space[i].shape[1] == 0:
                plane_coef_space[i] = torch.nn.Parameter(
                    plane_coef_space[i].data.new_zeros(1, n_component[i], res_target[mat_id_space_1], res_target[mat_id_space_0]),
                )
                plane_coef_time[i] = torch.nn.Parameter(
                    plane_coef_time[i].data.new_zeros(1, n_component[i], numFrames, res_target[mat_id_time_0])
                )
            else:
                plane_coef_space[i] = torch.nn.Parameter(
                    F.interpolate(
                        plane_coef_space[i].data,
                        size=(res_target[mat_id_space_1], res_target[mat_id_space_0]),
                        mode="bilinear",
                        align_corners=True,
                    )
                )
                plane_coef_time[i] = torch.nn.Parameter(
                    F.interpolate(
                        plane_coef_time[i].data,
                        size=(numFrames, res_target[mat_id_time_0]),
                        mode="bilinear",
                        align_corners=True,
                    )
                )

        return plane_coef_space, plane_coef_time

    @torch.no_grad()
    def upsample_volume_grid(self, res_target):
        self.app_plane_space, self.app_plane_time = self.up_sampling_VM(
            self.app_n_comp, self.app_plane_space, self.app_plane_time, res_target, self.num_keyframes
        )
        self.density_plane_space, self.density_plane_time = self.up_sampling_VM(
            self.density_n_comp,
            self.density_plane_space,
            self.density_plane_time,
            res_target,
            self.num_keyframes,
        )
        self.update_stepSize(res_target)
        print(f"upsamping to {res_target}")

    @torch.no_grad()
    def shrink(self, new_aabb):
        print("====> shrinking ...")

        xyz_min, xyz_max = new_aabb
        t_l, b_r = (xyz_min - self.aabb[0]) / self.units, (
            xyz_max - self.aabb[0]
        ) / self.units
        # print(new_aabb, self.aabb)

        # print(t_l, b_r,self.alphaMask.alpha_volume.shape)
        t_l, b_r = torch.round(torch.round(t_l)).long(), torch.round(b_r).long() + 1
        b_r = torch.stack([b_r, self.gridSize]).amin(0)

        for i in range(len(self.matModeSpace)):
            mat_id_space_0, mat_id_space_1 = self.matModeSpace[i]
            mat_id_time_0, mat_id_time_1 = self.matModeTime[i]

            self.density_plane_space[i] = torch.nn.Parameter(
                self.density_plane_space[i].data[
                    ...,
                    t_l[mat_id_space_1] : b_r[mat_id_space_1],
                    t_l[mat_id_space_0] : b_r[mat_id_space_0],
                ]
            )
            self.density_plane_time[i] = torch.nn.Parameter(
                self.density_plane_time[i].data[
                    ..., :, t_l[mat_id_time_0] : b_r[mat_id_time_0]
                ]
            )
            self.app_plane_space[i] = torch.nn.Parameter(
                self.app_plane_space[i].data[
                    ...,
                    t_l[mat_id_space_1] : b_r[mat_id_space_1],
                    t_l[mat_id_space_0] : b_r[mat_id_space_0],
                ]
            )
            self.app_plane_time[i] = torch.nn.Parameter(
                self.app_plane_time[i].data[
                    ..., :, t_l[mat_id_time_0] : b_r[mat_id_time_0]
                ]
            )

        if not torch.all(self.alphaMask.gridSize == self.gridSize):
            t_l_r, b_r_r = t_l / (self.gridSize - 1), (b_r - 1) / (self.gridSize - 1)
            correct_aabb = torch.zeros_like(new_aabb)
            correct_aabb[0] = (1 - t_l_r) * self.aabb[0] + t_l_r * self.aabb[1]
            correct_aabb[1] = (1 - b_r_r) * self.aabb[0] + b_r_r * self.aabb[1]
            print("aabb", new_aabb, "\ncorrect aabb", correct_aabb)
            new_aabb = correct_aabb

        newSize = b_r - t_l
        self.aabb = new_aabb
        self.update_stepSize((newSize[0], newSize[1], newSize[2]))

    @torch.no_grad()
    def getDenseAlpha(self, gridSize=None):
        samples = torch.stack(
            torch.meshgrid(
                torch.linspace(0, 1, gridSize[0]),
                torch.linspace(0, 1, gridSize[1]),
                torch.linspace(0, 1, gridSize[2]),
            ),
            -1,
        ).to(self.device)
        dense_xyz = self.aabb[0] * (1 - samples) + self.aabb[1] * samples

        # dense_xyz = dense_xyz
        # print(self.stepSize, self.distance_scale*self.aabbDiag)
        alpha = torch.zeros_like(dense_xyz[..., 0])
        time_scale_factor = (self.total_num_frames - 1) / self.total_num_frames

        for t in np.linspace(0, 1, self.total_num_frames):
            cur_alpha = torch.zeros_like(alpha)
            times = torch.ones_like(dense_xyz[..., -1:]) * t
            base_times = torch.round(
                (times * time_scale_factor).clamp(0.0, self.num_keyframes - 1)
            ) * (1.0 / time_scale_factor)
            time_offset = times - base_times

            for i in range(gridSize[0]):
                cur_xyz = dense_xyz[i].view(-1, 3)
                cur_base_times = base_times[i].view(-1, 1)
                cur_times = times[i].view(-1, 1)
                cur_time_offset = time_offset[i].view(-1, 1)

                cur_xyzt = torch.cat([cur_xyz, cur_base_times], -1)
                cur_alpha[i] = self.compute_alpha(
                    cur_xyzt, 0.01, times=cur_times, time_offset=cur_time_offset
                ).view((gridSize[1], gridSize[2]))

            alpha = torch.maximum(alpha, cur_alpha)

        return alpha, dense_xyz

    @torch.no_grad()
    def getMPI(self, t, density_fac, gridSize=None):
        dense_xyz = torch.stack(
            torch.meshgrid(
                torch.linspace(-1.5, 1.5, gridSize[0]),
                torch.linspace(-1.5, 1.5, gridSize[1]),
                torch.linspace(1.5, -1.5, gridSize[2]),
            ),
            -1,
        ).to(self.device)

        # dense_xyz = dense_xyz
        # print(self.stepSize, self.distance_scale*self.aabbDiag)
        time_scale_factor = (self.total_num_frames - 1) / self.total_num_frames

        t = np.linspace(0, 1, self.total_num_frames)[t]

        cur_alpha = torch.zeros_like(dense_xyz[..., 0])
        cur_rgb = torch.zeros_like(dense_xyz)
        times = torch.ones_like(dense_xyz[..., -1:]) * t
        base_times = torch.round(
            (times * time_scale_factor).clamp(0.0, self.num_keyframes - 1)
        ) * (1.0 / time_scale_factor)
        time_offset = times - base_times

        for i in range(gridSize[0]):
            cur_xyz = dense_xyz[i].view(-1, 3)
            cur_base_times = base_times[i].view(-1, 1)
            cur_times = times[i].view(-1, 1)
            cur_time_offset = time_offset[i].view(-1, 1)

            cur_viewdirs = torch.zeros_like(cur_xyz)
            cur_viewdirs[..., -1] = -1

            cur_xyzt = torch.cat([cur_xyz, cur_base_times], -1)
            cur_alpha[i] = self.compute_alpha(
                cur_xyzt, density_fac, times=cur_times, time_offset=cur_time_offset
            ).view((gridSize[1], gridSize[2]))

            cur_xyzt = torch.cat([self.normalize_coord(cur_xyz), self.normalize_time_coord(cur_base_times)], -1)
            app_features = self.compute_appfeature(cur_xyzt)
            temp_rgb = self.renderModule(
                cur_xyzt,
                cur_viewdirs,
                app_features,
                {
                    "frames_per_keyframe": self.frames_per_keyframe,
                    "num_keyframes": self.num_keyframes,
                    "total_num_frames": self.total_num_frames,
                    "times": cur_times,
                    "time_offset": cur_time_offset,
                },
            )

            cur_rgb[i] = temp_rgb.reshape(gridSize[1], gridSize[2], 3)
        
        for i in range(gridSize[2]):
            layer = cur_alpha[..., i].detach().cpu().numpy()
            layer = layer.transpose(1, 0)
            layer = layer[::-1]

            layer_rgb = cur_rgb[..., i, :].detach().cpu().numpy()
            layer_rgb = layer_rgb.transpose(1, 0, 2)
            layer_rgb = layer_rgb[::-1]
            layer_rgb = np.stack(
                [layer_rgb[..., 2], layer_rgb[..., 1], layer_rgb[..., 0]],
                axis=-1
            )

            layer_rgb_mult = (layer[..., None]) * layer_rgb

            cv2.imwrite(f'tmp/{i}_alpha.png', np.uint8(layer * 255))
            cv2.imwrite(f'tmp/{i}_color.png', np.uint8(layer_rgb * 255))
            cv2.imwrite(f'tmp/{i}_color_mult.png', np.uint8(layer_rgb_mult * 255))

        return cur_alpha, dense_xyz

    def normalize_time_coord(self, time):
        return (time * self.time_scale_factor + self.time_pixel_offset) * 2 - 1

    def compute_alpha(self, xyzt_locs, length=0.01, times=None, time_offset=None):
        sigma = torch.zeros(xyzt_locs.shape[:-1], device=xyzt_locs.device)

        xyzt_sampled = torch.cat(
            [
                self.normalize_coord(xyzt_locs[..., :3]),
                self.normalize_time_coord(xyzt_locs[..., -1:]),
            ],
            dim=-1,
        )
        sigma_feature = self.compute_densityfeature(xyzt_sampled)
        sigma = self.feature2density(
            sigma_feature,
            {
                "frames_per_keyframe": self.frames_per_keyframe,
                "num_keyframes": self.num_keyframes,
                "total_num_frames": self.total_num_frames,
                "times": times,
                "time_offset": time_offset,
                "weights": torch.ones_like(times),
            },
        )

        alpha = 1 - torch.exp(-sigma * length).view(xyzt_locs.shape[:-1])

        return alpha

    def forward(self, x, render_kwargs):
        #if 'rendering' in render_kwargs and render_kwargs['rendering']:
        #    self.getMPI(25, 1.5, (200,200,64))
        #    exit()

        batch_size = x["viewdirs"].shape[0]

        # Positions + times
        nSamples = x["points"].shape[-1] // 3
        xyzt_sampled = torch.cat(
            [
                x["points"].view(batch_size, -1, 3),
                x["base_times"].view(batch_size, -1, 1),
            ],
            -1,
        )

        # Distances
        distances = x["distances"].view(batch_size, -1)
        deltas = torch.cat(
            [
                distances[..., 1:] - distances[..., :-1],
                1e10 * torch.ones_like(distances[:, :1]),
            ],
            dim=1,
        )
        #deltas = torch.ones_like(deltas) * (1.0 / deltas.shape[1])

        #deltas = torch.cat(
        #    [
        #        distances[..., 0:1],
        #        distances[..., 1:] - distances[..., :-1],
        #    ],
        #    dim=1,
        #)

        # Times & viewdirs
        times = x["times"].view(batch_size, -1, 1)
        time_offset = x["time_offset"].view(batch_size, -1, 1)
        viewdirs = x["viewdirs"].view(batch_size, nSamples, 3)

        # Weights
        weights = x["weights"].view(batch_size, -1, 1)

        # Mask out
        ray_valid = self.valid_mask(xyzt_sampled[..., :3]) & (distances > 0)

        # Filter
        if self.apply_filter_weights and self.cur_iter >= self.filter_wait_iters:
            weights = weights.view(batch_size, -1)
            min_weight = torch.topk(weights, self.filter_max_samples, dim=-1, sorted=False)[0].min(-1)[0].unsqueeze(-1)

            ray_valid = ray_valid \
                & (weights >= (min_weight - 1e-8)) \
                & (weights > self.filter_weight_thresh)

            weights = weights.view(batch_size, -1, 1)
            weights = torch.ones_like(weights) # TODO: maybe remove
        else:
            weights = torch.ones_like(weights) # TODO: maybe remove
            pass

        if self.alphaMask is not None and False:
        #if self.alphaMask is not None:
            alphas = self.alphaMask.sample_alpha(xyzt_sampled[..., :3][ray_valid])
            alpha_mask = alphas > 0
            ray_invalid = ~ray_valid
            ray_invalid[ray_valid] |= ~alpha_mask
            ray_valid = ~ray_invalid

        # Get densities
        xyzt_sampled = torch.cat(
            [
                self.normalize_coord(xyzt_sampled[..., :3]),
                self.normalize_time_coord(xyzt_sampled[..., -1:]),
            ],
            dim=-1,
        )
        sigma = xyzt_sampled.new_zeros(
            xyzt_sampled.shape[:-1], device=xyzt_sampled.device
        )

        if ray_valid.any():
            sigma_feature = self.compute_densityfeature(xyzt_sampled[ray_valid])

            # Convert to density
            valid_sigma = self.feature2density(
                sigma_feature,
                {
                    "frames_per_keyframe": self.frames_per_keyframe,
                    "num_keyframes": self.num_keyframes,
                    "total_num_frames": self.total_num_frames,
                    "times": times[ray_valid],
                    "time_offset": time_offset[ray_valid],
                    "weights": weights[ray_valid],
                },
            )

            # Update valid
            assert valid_sigma is not None
            assert ray_valid is not None

            sigma[ray_valid] = valid_sigma

        alpha, weight, bg_weight = raw2alpha(sigma, deltas * self.distance_scale)
        app_mask = weight > self.rayMarch_weight_thres

        # Get colors
        rgb = xyzt_sampled.new_zeros(
            (xyzt_sampled.shape[0], xyzt_sampled.shape[1], 3),
            device=xyzt_sampled.device,
        )

        if app_mask.any():
            app_features = self.compute_appfeature(xyzt_sampled[app_mask])

            valid_rgbs = self.renderModule(
                xyzt_sampled[app_mask],
                viewdirs[app_mask],
                app_features,
                {
                    "frames_per_keyframe": self.frames_per_keyframe,
                    "num_keyframes": self.num_keyframes,
                    "total_num_frames": self.total_num_frames,
                    "times": times[app_mask],
                    "time_offset": time_offset[app_mask],
                },
            )
            rgb = valid_rgbs.new_zeros( # TODO: maybe remove
                (xyzt_sampled.shape[0], xyzt_sampled.shape[1], 3), device=xyzt_sampled.device
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


tensorf_dynamic_dict = {
    "tensor_vm_split_time": TensorVMKeyframeTime,
}
