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
    correct_poses_bounds,
)
from utils.ray_utils import (
    get_rays,
    get_ray_directions_K,
    get_ndc_rays_fx_fy,
)


class LLFFDataset(Base5DDataset):
    def __init__(
        self,
        cfg,
        split='train',
        **kwargs
        ):
        super().__init__(cfg, split, **kwargs)

    def read_meta(self):
        with self.pmgr.open(
            os.path.join(self.root_dir, 'poses_bounds.npy'), 'rb'
        ) as f:
            poses_bounds = np.load(f)

        self.image_paths = sorted(
            self.pmgr.ls(os.path.join(self.root_dir, 'images/'))
        )
        self.camera_ids = np.linspace(0, len(self.image_paths) - 1, len(self.image_paths))
        self.total_num_views = len(self.image_paths)

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
            poses, self.bounds
        )

        if not self.use_ndc:
            self.bounds = self.bounds / np.max(np.abs(poses[..., :3, 3]))

        self.near = self.bounds.min() * 0.95
        self.far = self.bounds.max() * 1.05
        self.depth_range = np.array([self.near * 2.0, self.far])

        # Step 3: Ray directions for all pixels
        self.directions = get_ray_directions_K(
            self.img_wh[1], self.img_wh[0], self.K, centered_pixels=True
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
            self.camera_ids = self.camera_ids[val_indices]
            self.poses = self.poses[val_indices]
        elif self.split == 'train':
            self.image_paths = [self.image_paths[i] for i in train_indices]
            self.camera_ids = self.camera_ids[train_indices]
            self.poses = self.poses[train_indices]

    def get_intrinsics(self):
        return self.K

    def to_ndc(self, rays):
        return get_ndc_rays_fx_fy(
            self.img_wh[1], self.img_wh[0], self.K[0, 0], self.K[1, 1], self.near, rays
        )

    def get_coords(self, idx):
        if self.split != 'train' or self.split == 'render':
            camera_id = 1
        else:
            camera_id = self.camera_ids[idx]

        c2w = torch.FloatTensor(self.poses[idx])
        rays_o, rays_d = get_rays(self.directions, c2w)

        print(f"Loading image {idx}")

        rays = torch.cat([rays_o, rays_d], dim=-1)

        if self.use_ndc:
            rays = self.to_ndc(rays)
        
        # Add camera idx
        rays = torch.cat([rays, torch.ones_like(rays[..., :1]) * camera_id], dim=-1)
        return rays

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


class DenseLLFFDataset(LLFFDataset):
    def __init__(
        self,
        cfg,
        split='train',
        **kwargs
        ):
        super().__init__(cfg, split, **kwargs)

    def read_meta(self):
        ## Bounds
        with self.pmgr.open(
            os.path.join(self.root_dir, 'bounds.npy'), 'rb'
        ) as f:
            bounds = np.load(f)

        self.bounds = bounds[:, -2:]

        ## Poses
        with self.pmgr.open(
            os.path.join(self.root_dir, 'poses.npy'), 'rb'
        ) as f:
            poses = np.load(f)

        ## Image paths
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

        ## Skip
        row_skip = self.dataset_cfg.train_row_skip
        col_skip = self.dataset_cfg.train_col_skip

        poses_skipped = []
        image_paths_skipped = []

        for row in range(self.dataset_cfg.num_rows):
            for col in range(self.dataset_cfg.num_cols):
                idx = row * self.dataset_cfg.num_cols + col

                if self.split == 'train' and (
                    (row % row_skip) != 0 or (col % col_skip) != 0 or (idx % self.val_skip) == 0
                    ):
                    continue

                if (self.split == 'val' or self.split == 'test') and (
                    ((row % row_skip) == 0 and (col % col_skip) == 0) and (idx % self.val_skip) != 0
                    ):
                    continue

                poses_skipped.append(poses[idx])
                image_paths_skipped.append(self.image_paths[idx])

        poses = np.stack(poses_skipped, axis=0)
        self.poses = poses.reshape(-1, 3, 5)
        self.image_paths = image_paths_skipped

        # Step 1: rescale focal length according to training resolution
        H, W, self.focal = poses[0, :, -1]
        self.cx, self.cy = W / 2.0, H / 2.0

        self.K = np.eye(3)
        self.K[0, 0] = self.focal * self.img_wh[0] / W
        self.K[0, 2] = self.cx * self.img_wh[0] / W
        self.K[1, 1] = self.focal * self.img_wh[1] / H
        self.K[1, 2] = self.cy * self.img_wh[1] / H

        # Step 2: correct poses, bounds
        self.near = self.bounds.min()
        self.far = self.bounds.max()

        # Step 3: Ray directions for all pixels
        self.directions = get_ray_directions_K(
            self.img_wh[1], self.img_wh[0], self.K, centered_pixels=self.centered_pixels
        )
