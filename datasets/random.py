#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import numpy as np
from torch.utils.data import Dataset
from utils.ray_utils import (
    get_random_pixels,
    get_pixels_for_image,
    get_ray_directions_from_pixels_K,
    get_rays,
    sample_images_at_xy,
)
import matplotlib.pyplot as plt


class RandomRayDataset(Dataset):
    def __init__(
        self,
        cfg,
        train_dataset=None,
        **kwargs
    ):
        super().__init__()

        self.cfg = cfg
        self.img_wh = train_dataset.img_wh
        self.near = train_dataset.near
        self.far = train_dataset.far
        self.use_ndc = train_dataset.use_ndc
        self.num_images = train_dataset.num_images
        self.batch_size = cfg.batch_size

        if 'save_data' not in kwargs or kwargs['save_data']:
            self.all_rays = torch.clone(train_dataset.all_rays)
            self.all_rgb = torch.clone(train_dataset.all_rgb)

            # Current
            self.current_rays = self.all_rays
            self.current_rgb = self.all_rgb

            # Prepare
            self.prepare_data()

    def compute_stats(self):
        all_rays = self.all_rays.view(
            self.num_images, self.img_wh[1] * self.img_wh[0], -1
        )

        ## Per view statistics
        self.all_means = []
        self.all_stds = []

        for idx in range(self.num_images):
            cur_rays = all_rays[idx]

            self.all_means += [cur_rays.mean(0)]
            self.all_stds += [cur_rays.std(0)]

        self.all_means = torch.stack(self.all_means, 0)
        self.all_stds = torch.stack(self.all_stds, 0)

        ## Full dataset statistics
        self.pos_mean = self.all_rays[..., :3].mean(0)
        self.pos_std = self.all_rays[..., :3].std(0)

        self.dir_mean = self.all_rays[..., 3:].mean(0)
        self.dir_std = self.all_rays[..., 3:].std(0)

        self.rgb_mean = self.all_rgb.mean(0)
        self.rgb_std = self.all_rgb.std(0)

    def prepare_data(self):
        self.compute_stats()
        self.shuffle()

    def shuffle(self):
        pass

    def __len__(self):
        return len(self.all_rays)

    def jitter(self, rays, jitter=None):
        if jitter is not None:
            jitter_rays = rays

            if 'pos' in jitter:
                jitter_rays = self.jitter_ray_origins(jitter_rays, jitter)

            if 'dir' in jitter:
                jitter_rays = self.jitter_ray_directions(jitter_rays, jitter)

            return jitter_rays
        else:
            return rays

    def get_batch(self, batch_idx, batch_size, jitter=None):
        batch = {}

        ## Get random rays
        batch['rays'] = self.get_random_rays(batch_size, self.cfg.range)

        ## Jitter
        batch['jitter_rays'] = self.jitter(batch['rays'], jitter)

        return batch

    def get_random_rays(self, num_rays, ray_range):
        ray_dim = self.all_rays.shape[-1] // 2

        pos_rand = torch.randn(
            (num_rays, ray_dim)
        ) * self.pos_std[None] * ray_range.pos
        rays_o = self.pos_mean[None] + pos_rand

        dir_rand = torch.randn(
            (num_rays, ray_dim)
        ) * self.dir_std[None] * ray_range.dir
        rays_d = self.dir_mean[None] + dir_rand
        rays_d = torch.nn.functional.normalize(rays_d, p=2.0, dim=-1)

        return torch.cat([rays_o, rays_d], -1)

    def jitter_ray_origins(self, rays, jitter):
        ray_dim = self.all_rays.shape[-1] // 2

        pos_rand = torch.randn(
            (rays.shape[0], jitter.bundle_size, ray_dim), device=rays.device
        ) * self.pos_std[None].type_as(rays) * jitter.pos

        rays = rays.view(rays.shape[0], -1, rays.shape[-1])
        if rays.shape[1] == 1:
            rays = rays.repeat(1, jitter.bundle_size, 1)

        rays_o = rays[..., :ray_dim] + pos_rand.type_as(rays)

        return torch.cat([rays_o, rays[..., ray_dim:]], -1)

    def jitter_ray_directions(self, rays, jitter):
        ray_dim = self.all_rays.shape[-1] // 2

        dir_rand = torch.randn(
            (rays.shape[0], jitter.bundle_size, ray_dim), device=rays.device
        ) * self.dir_std[None].type_as(rays) * jitter.dir

        rays = rays.view(rays.shape[0], -1, rays.shape[-1])
        if rays.shape[1] == 1:
            rays = rays.repeat(1, jitter.bundle_size, 1)

        rays_d = rays[..., ray_dim:] + dir_rand.type_as(rays)
        rays_d = torch.nn.functional.normalize(rays_d, p=2.0, dim=-1)

        return torch.cat([rays[..., :ray_dim], rays_d], -1)


class RandomPixelDataset(Dataset):
    def __init__(
        self,
        cfg,
        train_dataset=None,
        **kwargs
    ):
        super().__init__()

        self.cfg = cfg
        self.pixels_per_image = cfg.batch_size if 'pixels_per_image' in cfg else None
        self.use_ndc = train_dataset.use_ndc
        self.prepare_data(train_dataset)

    def prepare_data(self, train_dataset):
        # Create tensors
        self.all_rays = []
        self.all_rgb = []

        if self.use_ndc:
            self.all_ndc_rays = []

        # Random rays for each training image
        if self.pixels_per_image is None:
            self.pixels_per_image = train_dataset.img_wh[1] * train_dataset.img_wh[0]

        H, W = train_dataset.img_wh[1], train_dataset.img_wh[0]

        for i in range(train_dataset.num_images):
            # Get random directions
            cur_pixels = get_random_pixels(
                self.pixels_per_image, H, W,
            )
            cur_directions = get_ray_directions_from_pixels_K(
                cur_pixels,
                train_dataset.K, centered_pixels=train_dataset.centered_pixels
            )

            # Sample rays
            c2w = torch.FloatTensor(train_dataset.poses[i])
            cur_rays = torch.cat(list(get_rays(cur_directions, c2w)), -1)

            # Sample pixel colors
            cur_rgb = train_dataset.all_rgb.view(
                train_dataset.num_images, H, W, -1
            )[i].unsqueeze(0)
            cur_rgb = sample_images_at_xy(
                cur_rgb, cur_pixels, H, W
            )

            # Append
            self.all_rays.append(cur_rays.reshape(-1, 6))
            self.all_rgb.append(cur_rgb.reshape(-1, 3))

            if self.use_ndc:
                self.all_ndc_rays.append(
                    train_dataset.to_ndc(self.all_rays[-1])
                )

        # Concat tensors
        self.all_rays = torch.cat(self.all_rays, 0)
        self.all_rgb = torch.cat(self.all_rgb, 0)

        if self.use_ndc:
            self.all_ndc_rays = torch.cat(self.all_ndc_rays, 0)

    def shuffle(self):
        perm = torch.tensor(
            np.random.permutation(len(self))
        )
        self.all_rays = self.all_rays[perm]
        self.all_rgb = self.all_rgb[perm]

        if self.use_ndc:
            self.all_ndc_rays = self.all_ndc_rays[perm]

    def __len__(self):
        return len(self.all_rays)

    def jitter(self, rays, jitter=None):
        return rays

    def get_batch(self, batch_idx, batch_size, jitter=None):
        batch = {}
        batch_start = batch_size * batch_idx

        if self.use_ndc:
            batch['rays'] = self.all_ndc_rays[batch_start:batch_start+batch_size]
        else:
            batch['rays'] = self.all_rays[batch_start:batch_start+batch_size]

        batch['rgb'] = self.all_rgb[batch_start:batch_start+batch_size]

        return batch


class RandomViewSubsetDataset(RandomRayDataset):
    def __init__(
        self,
        cfg,
        train_dataset=None,
        **kwargs
    ):
        self.num_images = len(train_dataset.image_paths)
        self.num_views = train_dataset.num_images \
            if cfg.dataset.num_views == 'all' else cfg.dataset.num_views

        self.poses = np.tile(np.eye(4)[None], (self.num_images, 1, 1))
        self.poses[..., :3, :4] = train_dataset.poses[..., :3, :4]
        self.poses_inv = np.linalg.inv(self.poses)
        self.intrinsics = train_dataset.get_intrinsics_screen_space()
        self.current_poses_inv = self.poses_inv

        super().__init__(cfg, train_dataset=train_dataset, **kwargs)

    def shuffle(self):
        ## Get random view subset
        self.current_views = self.get_random_views(self.num_views)

        self.current_rays = self.all_rays.view(
            self.num_images, self.img_wh[1] * self.img_wh[0], -1
        )[self.current_views]
        self.current_rgb = self.all_rgb.view(
            self.num_images, self.img_wh[1] * self.img_wh[0], -1
        )[self.current_views]
        self.current_poses = self.poses[self.current_views]
        self.current_poses_inv = np.linalg.inv(self.current_poses)

        self.current_means = self.all_means[self.current_views]
        self.current_stds = self.all_stds[self.current_views]

        print(self.current_views)

    def __len__(self):
        return len(self.all_rays)

    def __getitem__(self, idx):
        return {
            'rays': self.random_rays[idx],
            'jitter_rays': self.jitter_rays[idx],
        }

    def get_random_views(self, n_views):
        if self.num_views == self.num_images:
            return list(range(self.num_images))
        else:
            return list(np.random.choice(
                np.arange(0, self.num_images),
                size=n_views,
                replace=False
            ))

    def get_random_rays_convex_hull(
        self, num_rays, ray_range
    ):
        rays = self.current_rays
        rays = rays[:, torch.randperm(rays.shape[1])]
        rays = rays[:, :num_rays]

        weights = torch.rand(
            num_rays, self.num_views
        ).type_as(rays)
        weights = weights / (weights.sum(-1).unsqueeze(-1) + 1e-8)
        weights = weights.permute(1, 0)

        rays = rays * weights.unsqueeze(-1)
        rays = rays.sum(0)

        rays_o = rays[..., 0:3]
        rays_d = rays[..., 3:6]
        rays_d = torch.nn.functional.normalize(rays_d, p=2.0, dim=-1)

        return torch.cat([rays_o, rays_d], -1)

    def project_points(self, P, points):
        points = torch.cat(
            [points, torch.ones_like(points[..., -1:])],
            dim=-1
        )
        points = points.unsqueeze(0)
        points = (P @ points.permute(0, 2, 1)).permute(
            0, 2, 1
        )
        pixels = points[..., :2] / (-points[..., -1:])

        return pixels

    def lookup_points(self, points):
        # Projection matrix
        poses_inv = torch.Tensor(
            self.current_poses_inv
        ).type_as(points)[..., :3, :4]
        K = torch.Tensor(
            self.intrinsics
        ).type_as(points).unsqueeze(0)
        P = (K @ poses_inv)

        # Project points
        pixels = self.project_points(P, points)

        # Valid mask
        valid_mask = (pixels[..., 0] > -1) & (pixels[..., 0] < 1) \
            & (pixels[..., 1] > -1) & (pixels[..., 1] < 1)
        valid_mask = valid_mask.type_as(points).detach()[..., None]

        # Weights
        camera_centers = torch.Tensor(
            self.current_poses
        ).type_as(points)[..., None, :3, -1].repeat(1, points.shape[0], 1)
        camera_dirs = torch.nn.functional.normalize(
            points.unsqueeze(0) - camera_centers, p=2.0, dim=-1
        )
        camera_rays = torch.cat([camera_centers, camera_dirs], dim=-1)

        # Lookup
        pixels = pixels.view(self.num_views, -1, 1, 2)
        rgb = self.current_rgb.permute(0, 2, 1).view(
            self.num_views, 3, self.img_wh[1], self.img_wh[0]
        ).type_as(points)
        values = torch.nn.functional.grid_sample(
            rgb, pixels
        )
        values = values.permute(0, 2, 3, 1).reshape(
            self.num_views, -1, 3
        )

        return values, camera_rays, valid_mask

    def project_points_single(self, P, points):
        points = torch.cat(
            [points, torch.ones_like(points[..., -1:])],
            dim=-1
        )
        points = (P @ points.permute(0, 2, 1)).permute(
            0, 2, 1
        )
        pixels = points[..., :2] / (-points[..., -1:])

        return pixels

    def lookup_points_single(self, points, weights=None):
        # Projection matrix
        poses_inv = torch.Tensor(
            self.current_poses_inv
        ).type_as(points)[..., :3, :4]
        K = torch.Tensor(
            self.intrinsics
        ).type_as(points).unsqueeze(0)
        P = (K @ poses_inv)

        # Project points
        pixels = self.project_points_single(P, points)

        # Valid mask
        valid_mask = (pixels[..., 0] > -1) & (pixels[..., 0] < 1) \
            & (pixels[..., 1] > -1) & (pixels[..., 1] < 1)
        valid_mask = valid_mask.type_as(points).detach()[..., None]

        # Weights
        camera_centers = torch.Tensor(
            self.current_poses
        ).type_as(points)[..., None, :3, -1].repeat(1, points.shape[1], 1)
        camera_dirs = torch.nn.functional.normalize(
            points - camera_centers, p=2.0, dim=-1
        )
        camera_rays = torch.cat([camera_centers, camera_dirs], dim=-1)

        # Lookup
        pixels = pixels.view(self.num_views, -1, 1, 2)
        rgb = self.current_rgb.permute(0, 2, 1).view(
            self.num_views, 3, self.img_wh[1], self.img_wh[0]
        ).type_as(points)
        values = torch.nn.functional.grid_sample(
            rgb, pixels
        )
        values = values.permute(0, 2, 3, 1).reshape(
            self.num_views, -1, 3
        )

        return values, camera_rays, valid_mask


class RandomRayLightfieldDataset(RandomRayDataset):
    def __init__(
        self,
        cfg,
        train_dataset=None
    ):
        self.num_images = len(train_dataset.image_paths)
        self.size = len(train_dataset)

        self.uv_plane = cfg.dataset.uv_plane
        self.st_plane = cfg.dataset.st_plane

        if 'st_scale' in cfg.dataset and cfg.dataset.st_scale is not None:
            self.st_scale = cfg.dataset.st_scale
        elif train_dataset is not None and 'lightfield' in train_dataset.dataset_cfg:
            self.st_scale = train_dataset.st_scale
        else:
            self.st_scale = 1.0

        super().__init__(cfg, train_dataset, save_data=False)

    def get_random_rays(self, num_rays, ray_range):
        st = (torch.rand(
            (num_rays, 2)
        ) * 2 - 1) * ray_range.pos

        s = st[..., 0] * self.st_scale
        t = st[..., 1] * self.st_scale

        uv = (torch.rand(
            (num_rays, 2)
        ) * 2 - 1) * ray_range.dir

        u = uv[..., 0]
        v = uv[..., 1]

        rays = torch.stack(
            [
                s,
                t,
                self.st_plane * torch.ones_like(s),
                u - s,
                v - t,
                (self.uv_plane - self.st_plane) * torch.ones_like(s),
            ],
            -1
        )

        rays = torch.cat(
            [
                rays[..., 0:3],
                torch.nn.functional.normalize(rays[..., 3:6], p=2.0, dim=-1)
            ],
            -1
        )

        return rays


    def jitter_ray_directions(self, rays, jitter):
        dir_rand = torch.randn(
            (rays.shape[0], jitter.bundle_size, 2), device=rays.device
        ) * jitter.dir

        rays = rays.view(rays.shape[0], -1, rays.shape[-1])
        if rays.shape[1] == 1:
            rays = rays.repeat(1, jitter.bundle_size, 1)

        rays_d = torch.cat(
            [
                rays[..., 3:5] + dir_rand.type_as(rays),
                rays[..., 5:]
            ],
            -1
        )

        rays_d = torch.nn.functional.normalize(rays_d, p=2.0, dim=-1)

        rays = torch.cat(
            [
                rays[..., :3],
                rays_d,
            ],
            -1
        )

        return rays

    def jitter_ray_origins(self, rays, jitter):
        pos_rand = torch.randn(
            (rays.shape[0], jitter.bundle_size, 2), device=rays.device
        ) * jitter.pos * self.st_scale

        rays = rays.view(rays.shape[0], -1, rays.shape[-1])
        if rays.shape[1] == 1:
            rays = rays.repeat(1, jitter.bundle_size, 1)

        rays_o = rays[..., :2] + pos_rand.type_as(rays)

        rays = torch.cat(
            [
                rays_o,
                rays[..., 2:],
            ],
            -1
        )

        return rays

    def __len__(self):
        return len(self.random_rays)

    def __getitem__(self, idx):
        return {
            'rays': self.random_rays[idx],
            'jitter_rays': self.jitter_rays[idx],
        }
