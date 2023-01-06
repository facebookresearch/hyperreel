#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import torch
import numpy as np

from PIL import Image

from .base import Base5DDataset
from utils.pose_utils import (
    interpolate_poses,
    correct_poses_bounds,
    create_spiral_poses,
)
from utils.ray_utils import (
    get_rays,
    get_ray_directions_K,
    get_ndc_rays_fx_fy,
)


class EikonalDataset(Base5DDataset):
    def __init__(
        self,
        cfg,
        split='train',
        **kwargs
        ):
        self.num_views = cfg.dataset.num_views if 'num_views' in cfg.dataset else -1

        super().__init__(cfg, split, **kwargs)

    def read_meta(self):
        with self.pmgr.open(
            os.path.join(self.root_dir, 'poses_bounds.npy'), 'rb'
        ) as f:
            poses_bounds = np.load(f)

        self.image_paths = sorted(
            self.pmgr.ls(os.path.join(self.root_dir, 'images/'))
        )

        if self.img_wh is None:
            image_path = self.image_paths[0]

            with self.pmgr.open(
                os.path.join(self.root_dir, 'images', image_path),
                'rb'
            ) as im_file:
                img = np.array(Image.open(im_file).convert('RGB'))

            self._img_wh = (img.shape[1] // self.downsample, img.shape[0] // self.downsample)
            self.img_wh = (img.shape[1] // self.downsample, img.shape[0] // self.downsample)
            self.aspect = (float(self.img_wh[0]) / self.img_wh[1])

        if self.split in ['train', 'val']:
            assert len(poses_bounds) == len(self.image_paths), \
                'Mismatch between number of images and number of poses! Please rerun COLMAP!'

        poses = poses_bounds[:, :15].reshape(-1, 3, 5)
        self.bounds = poses_bounds[:, -2:]

        if self.num_views > 0:
            poses = poses[:self.num_views]
            self.image_paths = self.image_paths[:self.num_views]

        # Step 1: rescale focal length according to training resolution
        H, W, self.focal = poses[0, :, -1]
        self.cx, self.cy = W / 2.0, H / 2.0

        self.K = np.eye(3)
        self.K[0, 0] = self.focal * self.img_wh[0] / W
        self.K[0, 2] = self.cx * self.img_wh[0] / W
        self.K[1, 1] = self.focal * self.img_wh[1] / H
        self.K[1, 2] = self.cy * self.img_wh[1] / H

        # Step 2: correct poses, bounds
        self.poses, self.poses_avg, self.bounds = correct_poses_bounds(
            poses, self.bounds, center=True
        )

        if not self.use_ndc:
            self.bounds = self.bounds / np.max(np.abs(poses[..., :3, 3]))
            self.poses[..., :3, 3] = self.poses[..., :3, 3] / np.max(np.abs(poses[..., :3, 3]))

        self.near = self.bounds.min()
        self.far = self.bounds.max()

        # Step 3: Ray directions for all pixels
        self.centered_pixels = True
        self.directions = get_ray_directions_K(
            self.img_wh[1], self.img_wh[0], self.K, centered_pixels=self.centered_pixels
        )

        # Step 4: Holdout validation images
        if len(self.val_set) > 0:
            val_indices = self.val_set
        elif self.val_skip != 'inf':
            self.val_skip = min(
                len(self.image_paths), self.val_skip
            )
            val_indices = list(range(0, len(self.image_paths), self.val_skip))
        else:
            val_indices = []

        train_indices = [i for i in range(len(self.image_paths)) if i not in val_indices]

        if self.val_all:
            val_indices = [i for i in train_indices] # noqa

        if self.split == 'val' or self.split == 'test':
            self.image_paths = [self.image_paths[i] for i in val_indices]
            self.poses = self.poses[val_indices]
        elif self.split == 'train':
            self.image_paths = [self.image_paths[i] for i in train_indices]
            self.poses = self.poses[train_indices]

    def get_intrinsics(self):
        return self.K

    def to_ndc(self, rays):
        return get_ndc_rays_fx_fy(
            self.img_wh[1], self.img_wh[0], self.K[0, 0], self.K[1, 1], self.near, rays
        )

    def get_coords(self, idx):
        c2w = torch.FloatTensor(self.poses[idx])
        rays_o, rays_d = get_rays(self.directions, c2w)

        if self.use_ndc:
            rays = self.to_ndc(torch.cat([rays_o, rays_d], dim=-1))

            if self.include_world:
                rays = torch.cat([rays, rays_o, rays_d], dim=-1)

            return rays
        else:
            return torch.cat([rays_o, rays_d], dim=-1)

    def get_rgb(self, idx):
        # Colors
        image_path = self.image_paths[idx]

        with self.pmgr.open(
            os.path.join(self.root_dir, 'images', image_path),
            'rb'
        ) as im_file:
            img = Image.open(im_file).convert('RGB')

        img = img.resize(self._img_wh, Image.LANCZOS)

        if self.img_wh[0] != self._img_wh[0] or self.img_wh[1] != self._img_wh[1]:
            img = img.resize(self.img_wh, Image.BOX)

        img = self.transform(img)
        img = img.view(3, -1).permute(1, 0)

        return img

    def prepare_render_data(self):
        if not self.render_interpolate:
            close_depth, inf_depth = self.bounds.min()*.9, self.bounds.max()*5.

            dt = .75
            mean_dz = 1./(((1.-dt)/close_depth + dt/inf_depth))
            focus_depth = mean_dz

            radii = np.percentile(np.abs(self.poses[:16, ..., 3]), 50, axis=0)
            self.poses = create_spiral_poses(self.poses[:16], radii, focus_depth * 100)

            self.poses = np.stack(self.poses, axis=0)
            self.poses[..., :3, 3] = self.poses[..., :3, 3] - 0.1 * close_depth * self.poses[..., :3, 2]
        else:
            self.poses = interpolate_poses(self.poses, self.render_supersample)
