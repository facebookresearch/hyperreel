#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import torch
import numpy as np
from PIL import Image

from .llff import LLFFDataset

from utils.pose_utils import (
    interpolate_poses,
    correct_poses_bounds,
    create_spiral_poses,
)

from utils.ray_utils import (
    get_ray_directions_K,
    get_ndc_rays_fx_fy,
)

from utils.intersect_utils import intersect_axis_plane


class ShinyDataset(LLFFDataset):
    def __init__(
        self,
        cfg,
        split='train',
        **kwargs
        ):
        self.dense = cfg.dataset.collection == 'cd' or cfg.dataset.collection == 'lab'

        super().__init__(cfg, split, **kwargs)

    def read_meta(self):
        with self.pmgr.open(
            os.path.join(self.root_dir, 'poses_bounds.npy'),
            'rb'
        ) as f:
            poses_bounds = np.load(f)

        with self.pmgr.open(
            os.path.join(self.root_dir, 'hwf_cxcy.npy'),
            'rb'
        ) as f:
            hwfc = np.load(f)

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

        poses = poses_bounds[:, :12].reshape(-1, 3, 4)
        self.bounds = poses_bounds[:, -2:]

        # Step 1: rescale focal length according to training resolution
        H, W, self.focal = hwfc[:3, 0]
        self.cx, self.cy = hwfc[-2:, 0]

        self.K = np.eye(3)
        self.K[0, 0] = self.focal * self.img_wh[0] / W
        self.K[0, 2] = self.cx * self.img_wh[0] / W
        self.K[1, 1] = self.focal * self.img_wh[1] / H
        self.K[1, 2] = self.cy * self.img_wh[1] / H

        # Step 2: correct poses, bounds
        self.poses, self.poses_avg, self.bounds = correct_poses_bounds(
            poses, self.bounds, use_train_pose=True
        )

        with self.pmgr.open(
            os.path.join(self.root_dir, 'planes.txt'),
            'r'
        ) as f:
            planes = [float(i) for i in f.read().strip().split(' ')]

        self.near = planes[0] * 0.95
        self.far = planes[1] * 1.05
        self.depth_range = np.array([self.near * 2.0, self.far])

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
            self.camera_ids = self.camera_ids[val_indices]
            self.poses = self.poses[val_indices]
        elif self.split == 'train':
            self.image_paths = [self.image_paths[i] for i in train_indices]
            self.camera_ids = self.camera_ids[train_indices]
            self.poses = self.poses[train_indices]

    def to_ndc(self, rays):
        return get_ndc_rays_fx_fy(
            self.img_wh[1], self.img_wh[0], self.K[0, 0], self.K[1, 1], self.near, rays
        )

    def prepare_render_data(self):
        if not self.render_interpolate:
            close_depth, inf_depth = self.bounds.min()*.9, self.bounds.max()*5.

            dt = .75
            mean_dz = 1./(((1.-dt)/close_depth + dt/inf_depth))
            focus_depth = mean_dz

            if self.dense:
                radii = np.percentile(np.abs(self.poses[..., 3]), 50, axis=0)
                self.poses = create_spiral_poses(self.poses, radii, focus_depth * 100)
            else:
                radii = np.percentile(np.abs(self.poses[..., 3]), 85, axis=0)
                self.poses = create_spiral_poses(self.poses, radii, focus_depth * 2)

            self.poses = np.stack(self.poses, axis=0)
            self.poses[..., :3, 3] = self.poses[..., :3, 3] - 0.1 * close_depth * self.poses[..., :3, 2]
        else:
            self.poses = interpolate_poses(self.poses, self.render_supersample)


class DenseShinyDataset(ShinyDataset):
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

        ## Intrinsics
        with self.pmgr.open(
            os.path.join(self.root_dir, 'hwf_cxcy.npy'),
            'rb'
        ) as f:
            hwfc = np.load(f)

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
        H, W, self.focal = hwfc[:3, 0]
        self.cx, self.cy = hwfc[-2:, 0]

        self.K = np.eye(3)
        self.K[0, 0] = self.focal * self.img_wh[0] / W
        self.K[0, 2] = self.cx * self.img_wh[0] / W
        self.K[1, 1] = self.focal * self.img_wh[1] / H
        self.K[1, 2] = self.cy * self.img_wh[1] / H

        # Step 2: correct poses, bounds
        self.near = self.bounds.min()
        self.far = self.bounds.max()

        # Step 3: Ray directions for all pixels
        self.centered_pixels = True
        self.directions = get_ray_directions_K(
            self.img_wh[1], self.img_wh[0], self.K, centered_pixels=self.centered_pixels
        )
