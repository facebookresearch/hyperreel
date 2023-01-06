#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import numpy as np
import torch.nn.functional as F

from .base import BaseRegularizer
from losses import loss_dict

import copy
from omegaconf import OmegaConf # @manual //github/third-party/omry/omegaconf:omegaconf

from nlf.param import ray_param_dict, ray_param_pos_dict
from utils.ray_utils import (
    jitter_ray_origins,
    jitter_ray_directions,
    compute_sigma_angle,
    compute_sigma_dot,
    get_random_pixels,
    get_ray_directions_from_pixels_K,
)
from nlf.intersect import (
    intersect_dict,
)


def sample_simplex(batch_size, n, device):
    samples = torch.rand(batch_size, n, device=device)
    samples = torch.cat(
        [
            torch.zeros_like(samples[:, :1]),
            samples
        ],
        dim=-1
    )
    samples, _ = torch.sort(samples, dim=-1)
    return samples[:, 1:] - samples[:, :-1]


class RayDensityRegularizer(BaseRegularizer):
    def __init__(
        self,
        system,
        cfg
    ):
        super().__init__(system, cfg)

        self.num_views_for_random = cfg.num_views_for_random
        self.num_views_for_ray = cfg.num_views_for_ray

        self.extrapolate_freq = cfg.extrapolate_freq
        self.extrapolate_scale = cfg.extrapolate_scale
        self.batch_size = system.cfg.training.batch_size

        self.dataset = system.dm.train_dataset
        self.all_poses = torch.tensor(self.dataset.poses).cuda().float()
        self.all_centers = self.all_poses[:, :3, -1]
        self.K = torch.tensor(self.dataset.K).cuda().float()
        self.use_ndc = self.dataset.use_ndc

        # Perturb
        self.use_jitter = cfg.use_jitter if 'use_jitter' in cfg else False
        self.jitter_pos_std = cfg.jitter.pos_std
        self.jitter_dir_std = cfg.jitter.dir_std

        # Intersect fn
        if system.is_subdivided:
            model_cfg = system.cfg.model.ray
        else:
            model_cfg = system.cfg.model

        self.z_channels = model_cfg.embedding.z_channels

        isect_cfg = model_cfg.embedding.intersect
        self.intersect_fn = intersect_dict[isect_cfg.type](
            self.z_channels, isect_cfg
        )

        # Sigma computation
        self.angle_std = float(np.radians(self.cfg.angle_std)) if 'angle_std' in cfg else -1.0
        self.dot_std = float(cfg.dot_std) if 'dot_std' in cfg else -1.0
        self.angle_std = self.angle_std / self.dataset.num_images
        self.dot_std = self.dot_std / self.dataset.num_images

        # Loss
        self.loss_fn = loss_dict[self.cfg.loss.type](self.cfg.loss)

    def get_random_views(self, n_views, n_images):
        return list(np.random.choice(
            np.arange(0, n_images),
            size=n_views,
            replace=False
        ))

    def _loss(self, train_batch, batch_results, batch_idx):
        #### Prepare ####
        system = self.get_system()
        W, H = self.dataset.img_wh[0], self.dataset.img_wh[1]

        with torch.no_grad():
            # Subset of cameras
            views_idx = torch.randint(0, self.dataset.num_images - 1, (self.batch_size * self.num_views_for_random,), device='cuda')
            anchor_poses = self.all_poses[views_idx].reshape(-1, 3, 4)
            anchor_positions = self.all_centers[views_idx].reshape(-1, 3)

            # Generate random rays
            pixels = get_random_pixels(self.batch_size * self.num_views_for_random, H, W, device='cuda')
            anchor_directions = get_ray_directions_from_pixels_K(
                pixels,
                self.K,
                centered_pixels=True
            ).float()
            anchor_directions = (anchor_poses[:, :3, :3] @ anchor_directions.unsqueeze(-1)).squeeze(-1)
            anchor_directions = torch.nn.functional.normalize(anchor_directions, dim=-1)

            # Reshape
            anchor_positions = anchor_positions.view(self.batch_size, self.num_views_for_random, 3)
            anchor_directions = anchor_directions.view(self.batch_size, self.num_views_for_random, 3)

            # Extrapolate (sometimes)
            if (batch_idx % 3) == 1:
                print("Extrapolating")

                # Offset positions
                anchor_centroids = anchor_positions.mean(1).unsqueeze(1)
                anchor_positions = (
                    anchor_positions - anchor_centroids
                ) * self.extrapolate_scale + anchor_centroids

                # Offset directions
                anchor_dir_centroids = anchor_directions.mean(1).unsqueeze(1)
                anchor_dir_centroids = torch.nn.functional.normalize(anchor_dir_centroids, dim=-1)
                anchor_directions = (
                    anchor_directions - anchor_dir_centroids
                ) * self.extrapolate_scale + anchor_dir_centroids
                anchor_directions = torch.nn.functional.normalize(anchor_directions, dim=-1)

            # Interpolate (sometimes)
            if (batch_idx % 2) == 1:
                print("Interpolating")
                # Sample weights from unit simplex
                weights = sample_simplex(self.batch_size, self.num_views_for_random, 'cuda')

                # Interpolated positions, directions
                anchor_positions = (weights.unsqueeze(-1) * anchor_positions).sum(1)
                anchor_directions = torch.nn.functional.normalize(
                    (weights.unsqueeze(-1) * anchor_directions).sum(1),
                    dim=-1
                )
            else:
                print("Choosing position")

                # Grab first position, direction
                anchor_positions = anchor_positions[:, 0]
                anchor_directions = anchor_directions[:, 0]

            # Jitter
            if self.use_jitter:
                anchor_positions = anchor_positions + torch.randn(
                    anchor_positions.shape, device='cuda'
                ) * self.jitter_pos_std
                anchor_directions = anchor_directions + torch.randn(
                    anchor_directions.shape, device='cuda'
                ) * self.jitter_dir_std
                anchor_directions = torch.nn.functional.normalize(anchor_directions, dim=-1)

            # Rays
            random_rays = torch.cat([anchor_positions, anchor_directions], dim=-1)
            print("Rays shape:", random_rays.shape)

            # Get closest cameras to random rays
            centers = self.all_centers[None].repeat(self.batch_size, 1, 1)
            poses = self.all_poses[None].repeat(self.batch_size, 1, 1, 1)

            camera_dists = torch.linalg.norm(
                random_rays[..., None, :3] - centers, dim=-1
            )
            sort_idx = torch.argsort(camera_dists, dim=-1)

            centers = centers.permute(0, 2, 1)
            centers = torch.gather(centers, -1, sort_idx[:, None, :].repeat(1, 3, 1))
            centers = centers.permute(0, 2, 1)[:, :self.num_views_for_ray]

            poses = poses.permute(0, 2, 3, 1)
            poses = torch.gather(poses, -1, sort_idx[:, None, None, :].repeat(1, 3, 4, 1))
            poses = poses.permute(0, 3, 1, 2)[:, :self.num_views_for_ray]

            # Get intersection points along ray
            random_rays = random_rays.view(-1, 6)
            z_vals = random_rays.new_zeros(random_rays.shape[0], self.z_channels, 1)

            if self.use_ndc:
                random_rays_ndc = self.dataset.to_ndc(random_rays)
                t_p = self.intersect_fn.intersect(random_rays_ndc, random_rays_ndc, z_vals)

                o_z = -self.dataset.near
                t = (o_z / (1 - t_p) - o_z) / random_rays[..., 5, None]
                t = t + (o_z - random_rays[..., None, 2]) / random_rays[..., None, 5]
            else:
                t = self.intersect_fn.intersect(random_rays, random_rays_ndc, z_vals)

            points = random_rays[..., None, :3] + t[..., None] * random_rays[..., None, 3:6]

            # TODO: Project points into cameras
            # - Apply poses
            # - Apply intrinsics
            # - Find points outside of pixel bounds

            # Get directions
            camera_points = points.unsqueeze(1) - centers.unsqueeze(-2)
            dirs = torch.nn.functional.normalize(
                camera_points, dim=-1
            )

            # TODO: Automatically determine size of kernels for computing density
            #   - Unstrucured Lumigraph: Use max angle
            #   - Ours: Use angle standard deviation
            #   - Other: something more robust? MAD (median absolute deviation?)

            # Compute sigma
            h_sigma = compute_sigma_angle(random_rays[..., None, None, 3:6], dirs, angle_std=self.angle_std)
            #h_sigma = compute_sigma_dot(random_rays[..., None, None, 3:6], dirs, dot_std=self.dot_std)

            h_sigma = (torch.sigmoid(h_sigma * 1e-1) - 0.5) * 2.0
            h_sigma[torch.isnan(h_sigma)] = 1

            print("Sigma", h_sigma[0])

        # Loss
        random_rays = random_rays.view(-1, 6)
        sigma = system.render('embed_params', random_rays)['params']
        sigma = sigma.view(*h_sigma.shape)
        total_loss = self.loss_fn(h_sigma, sigma)

        return total_loss

class SimpleRayDensityRegularizer(BaseRegularizer):
    def __init__(
        self,
        system,
        cfg
    ):
        super().__init__(system, cfg)

        self.num_views_for_random = cfg.num_views_for_random
        self.num_views_for_ray = cfg.num_views_for_ray

        self.extrapolate_freq = cfg.extrapolate_freq
        self.extrapolate_scale = cfg.extrapolate_scale
        self.batch_size = system.cfg.training.batch_size

        self.dataset = system.dm.train_dataset
        self.all_poses = torch.tensor(self.dataset.poses).cuda().float()
        self.all_centers = self.all_poses[:, :3, -1]
        self.K = torch.tensor(self.dataset.K).cuda().float()

        self.use_ndc = self.dataset.use_ndc

        # Perturb
        self.use_jitter = cfg.use_jitter if 'use_jitter' in cfg else False
        self.jitter_pos_std = cfg.jitter.pos_std
        self.jitter_dir_std = cfg.jitter.dir_std

        # Intersect fn
        if system.is_subdivided:
            model_cfg = system.cfg.model.ray
        else:
            model_cfg = system.cfg.model

        self.z_channels = model_cfg.embedding.z_channels

        isect_cfg = model_cfg.embedding.intersect
        self.intersect_fn = intersect_dict[isect_cfg.type](
            self.z_channels, isect_cfg
        )

        # Loss
        self.loss_fn = loss_dict[self.cfg.loss.type](self.cfg.loss)

    def get_random_views(self, n_views, n_images):
        return list(np.random.choice(
            np.arange(0, n_images),
            size=n_views,
            replace=False
        ))

    def _loss(self, train_batch, batch_results, batch_idx):
        #### Prepare ####
        system = self.get_system()
        W, H = self.dataset.img_wh[0], self.dataset.img_wh[1]

        with torch.no_grad():
            # Subset of cameras
            views_idx = torch.randint(0, self.dataset.num_images - 1, (self.batch_size * self.num_views_for_random,), device='cuda')
            anchor_poses = self.all_poses[views_idx].reshape(-1, 3, 4)
            anchor_positions = self.all_centers[views_idx].reshape(-1, 3)

            # Generate random rays
            pixels = get_random_pixels(self.batch_size * self.num_views_for_random, H, W, device='cuda')
            anchor_directions = get_ray_directions_from_pixels_K(
                pixels,
                self.K,
                centered_pixels=True
            ).float()
            anchor_directions = (anchor_poses[:, :3, :3] @ anchor_directions.unsqueeze(-1)).squeeze(-1)
            anchor_directions = torch.nn.functional.normalize(anchor_directions, dim=-1)

            # Reshape
            anchor_positions = anchor_positions.view(self.batch_size, self.num_views_for_random, 3)
            anchor_directions = anchor_directions.view(self.batch_size, self.num_views_for_random, 3)

            # Extrapolate (sometimes)
            if (batch_idx % 3) == 0:
                print("Extrapolating")

                # Offset positions
                anchor_centroids = anchor_positions.mean(1).unsqueeze(1)
                anchor_positions = (
                    anchor_positions - anchor_centroids
                ) * self.extrapolate_scale + anchor_centroids

                # Offset directions
                anchor_dir_centroids = anchor_directions.mean(1).unsqueeze(1)
                anchor_dir_centroids = torch.nn.functional.normalize(anchor_dir_centroids, dim=-1)
                anchor_directions = (
                    anchor_directions - anchor_dir_centroids
                ) * self.extrapolate_scale + anchor_dir_centroids
                anchor_directions = torch.nn.functional.normalize(anchor_directions, dim=-1)

            # Interpolate (sometimes)
            if (batch_idx % 2) == 1:
                print("Interpolating")
                # Sample weights from unit simplex
                weights = sample_simplex(self.batch_size, self.num_views_for_random, 'cuda')

                # Interpolated positions, directions
                anchor_positions = (weights.unsqueeze(-1) * anchor_positions).sum(1)
                anchor_directions = torch.nn.functional.normalize(
                    (weights.unsqueeze(-1) * anchor_directions).sum(1),
                    dim=-1
                )
            else:
                print("Choosing position")

                # Grab first position, direction
                anchor_positions = anchor_positions[:, 0]
                anchor_directions = anchor_directions[:, 0]

            # Jitter
            if self.use_jitter:
                anchor_positions = anchor_positions + torch.randn(
                    anchor_positions.shape, device='cuda'
                ) * self.jitter_pos_std
                anchor_directions = anchor_directions + torch.randn(
                    anchor_directions.shape, device='cuda'
                ) * self.jitter_dir_std
                anchor_directions = torch.nn.functional.normalize(anchor_directions, dim=-1)

            # Rays
            random_rays = torch.cat([anchor_positions, anchor_directions], dim=-1)

            if self.use_ndc:
                random_rays = self.dataset.to_ndc(random_rays)
                random_rays[..., :3] = random_rays[..., :3].clamp(-2, 2)

        # Predicted sigma
        sigma = system.render('embed_params', random_rays)['params']

        # Weight map
        N = self.dataset.num_images

        if (batch_idx % 3) == 0:
            #weights = 1 - torch.exp(
            #    -torch.square(random_rays[..., :2]).mean(-1)
            #)
            #weights += 1 - torch.exp(
            #    -torch.square(random_rays[..., 3:5]).mean(-1)
            #)
            #weights = 2 * weights.unsqueeze(-1) / N

            weights = 4.0 * (1 - torch.exp(
                -torch.square(random_rays[..., :2]).mean(-1) \
                + -torch.square(random_rays[..., 3:5]).mean(-1)
            )) / N

            #weights = 4.0 / N
        else:
            weights = 1.0 / N

        ## Total loss
        sigma = sigma.view(self.batch_size, -1)
        total_loss = self.loss_fn(sigma * weights, torch.ones_like(sigma) * weights)
        print("Ray Density loss:", total_loss)

        return total_loss
