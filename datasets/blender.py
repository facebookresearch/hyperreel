#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import torch
import numpy as np

import json

import cv2
from PIL import Image

from .base import Base5DDataset
from .lightfield import LightfieldDataset

from utils.ray_utils import (
    get_rays,
    get_ray_directions_K
)


class BlenderLightfieldDataset(LightfieldDataset):
    def __init__(
        self,
        cfg,
        split='train',
        **kwargs
    ):
        super().__init__(cfg, split, **kwargs)

    def read_meta(self):
        # Read meta
        transforms_path = os.path.join(self.root_dir, 'transforms.json')

        with self.pmgr.open(transforms_path, 'r') as f:
            self.meta = json.load(f)

        # Image paths and pose
        self.image_paths = []
        self.poses = []

        for frame in self.meta['frames']:
            # Image path
            image_path = frame['file_path'].split('/')[-1]
            self.image_paths += [image_path]

            # Pose
            pose = np.array(frame['transform_matrix'])[:3, :4]
            self.poses += [pose]

    def get_rgb(self, idx):
        image_path = self.image_paths[idx]

        with self.pmgr.open(
            os.path.join(self.root_dir, f'{image_path}.png'),
            'rb'
        ) as im_file:
            img = Image.open(im_file).convert('RGBA')

        img = img.resize(self._img_wh, Image.LANCZOS)

        if self.img_wh[0] != self._img_wh[0] or self.img_wh[1] != self._img_wh[1]:
            img = img.resize(self.img_wh, Image.BOX)

        img = self.transform(img)
        img = img.view(4, -1).permute(1, 0)
        img = img[:, :3] * img[:, -1:] + (1 - img[:, -1:])

        return img


class BlenderDataset(Base5DDataset):
    def __init__(
        self,
        cfg,
        split='train',
        **kwargs
        ):

        super().__init__(cfg, split, **kwargs)

    def read_meta(self):
        if self.split == 'render':
            self.read_meta_for_split('test')
        elif self.split == 'train':
            self.read_meta_for_split('train')
        elif self.split == 'val':
            self.read_meta_for_split('test')
        else:
            self.read_meta_for_split(self.split)

    def read_meta_for_split(self, split):
        with self.pmgr.open(
            os.path.join(self.root_dir, f'transforms_{split}.json'),
            'r'
        ) as f:
            self.meta = json.load(f)

        if split == 'val':
            self.meta['frames'] = self.meta['frames'][:self.val_num]

        W, H = self.img_wh

        self.focal = 0.5 * 800 / np.tan(
            0.5 * self.meta['camera_angle_x']
        )
        self.focal *= self.img_wh[0] / 800

        self.K = np.eye(3)
        self.K[0, 0] = self.focal
        self.K[0, 2] = (W / 2.0)
        self.K[1, 1] = self.focal
        self.K[1, 2] = (H / 2.0)

        # Bounds, common for all scenes
        self.near = 2.0
        self.far = 6.0
        self.bounds = np.array([self.near, self.far])
        self.depth_range = np.array([self.near, self.far])

        # Ray directions for all pixels, same for all images (same H, W, focal)
        self.centered_pixels = True
        self.directions = get_ray_directions_K(
            H, W, self.K, centered_pixels=True
        )

        # Image paths and pose
        self.image_paths = []
        self.poses = []

        for frame in self.meta['frames']:
            # Image path
            self.image_paths += [frame['file_path']]

            # Pose
            pose = np.array(frame['transform_matrix'])[:3, :4]
            self.poses += [pose]

        self.poses = np.stack(self.poses, axis=0)

    def prepare_render_data(self):
        self.prepare_test_data()

    def get_coords(self, idx):
        c2w = torch.FloatTensor(self.poses[idx])
        rays_o, rays_d = get_rays(self.directions, c2w)
        return torch.cat([rays_o, rays_d], dim=-1)

    def get_rgb(self, idx):
        image_path = self.image_paths[idx]

        with self.pmgr.open(
            os.path.join(self.root_dir, f'{image_path}.png'),
            'rb'
        ) as im_file:
            img = Image.open(im_file).convert('RGBA')

        img = img.resize(self._img_wh, Image.LANCZOS)

        if self.img_wh[0] != self._img_wh[0] or self.img_wh[1] != self._img_wh[1]:
            img = img.resize(self.img_wh, Image.BOX)

        img = self.transform(img)
        img = img.view(4, -1).permute(1, 0)
        img = img[:, :3] * img[:, -1:] + (1 - img[:, -1:])

        return img

    def get_intrinsics(self):
        K = np.eye(3)
        K[0, 0] = self.focal
        K[0, 2] = self.img_wh[0] / 2
        K[1, 1] = self.focal
        K[1, 2] = self.img_wh[1] / 2

        return K


class DenseBlenderDataset(Base5DDataset):
    def __init__(
        self,
        cfg,
        split='train',
        **kwargs
        ):
        super().__init__(cfg, split, **kwargs)

    def read_meta(self):
        if self.split == 'render':
            self.read_meta_for_split('test')
        elif self.split == 'train':
            self.read_meta_for_split('test')
        elif self.split == 'val':
            self.read_meta_for_split('test')
        else:
            self.read_meta_for_split(self.split)

    def read_meta_for_split(self, split):
        with self.pmgr.open(
            os.path.join(self.root_dir, f'transforms_{split}.json'),
            'r'
        ) as f:
            self.meta = json.load(f)

        W, H = self.img_wh

        self.focal = 0.5 * 800 / np.tan(
            0.5 * self.meta['camera_angle_x']
        )
        self.focal *= self.img_wh[0] / 800

        self.K = np.eye(3)
        self.K[0, 0] = self.focal
        self.K[0, 2] = (W / 2.0)
        self.K[1, 1] = self.focal
        self.K[1, 2] = (H / 2.0)

        # Bounds, common for all scenes
        self.near = 2.0
        self.far = 6.0
        self.bounds = np.array([self.near, self.far])
        self.depth_range = np.array([self.near, self.far])

        # Ray directions for all pixels, same for all images (same H, W, focal)
        self.centered_pixels = True
        self.directions = get_ray_directions_K(
            H, W, self.K, centered_pixels=self.centered_pixels
        )

        # Image paths and pose
        self.image_paths = []
        self.poses = []

        for frame in self.meta['frames']:
            # Image path
            self.image_paths += [frame['file_path']]

            # Pose
            pose = np.array(frame['transform_matrix'])[:3, :4]
            self.poses += [pose]

        self.poses = np.stack(self.poses, axis=0)

        ## Holdout validation images
        if self.val_set == 'lightfield':
            step = self.dataset_cfg.lightfield_step
            rows = self.dataset_cfg.lightfield_rows
            cols = self.dataset_cfg.lightfield_cols
            val_indices = []

            for row in range(0, rows, 1):
                for col in range(0, cols, 1):
                    idx = row * cols + col

                    if row % step != 0 or col % step != 0:
                        val_indices.append(idx)

        elif len(self.val_set) > 0:
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

    def prepare_render_data(self):
        self.prepare_test_data()

    def get_coords(self, idx):
        c2w = torch.FloatTensor(self.poses[idx])
        rays_o, rays_d = get_rays(self.directions, c2w)
        return torch.cat([rays_o, rays_d], dim=-1)

    def get_rgb(self, idx):
        image_path = self.image_paths[idx]

        with self.pmgr.open(
            os.path.join(self.root_dir, f'{image_path}.png'),
            'rb'
        ) as im_file:
            img = Image.open(im_file).convert('RGBA')

        img = img.resize(self._img_wh, Image.LANCZOS)

        if self.img_wh[0] != self._img_wh[0] or self.img_wh[1] != self._img_wh[1]:
            img = img.resize(self.img_wh, Image.BOX)

        img = self.transform(img)
        img = img.view(4, -1).permute(1, 0)
        img = img[:, :3] * img[:, -1:] + (1 - img[:, -1:])

        return img

    def get_intrinsics(self):
        K = np.eye(3)
        K[0, 0] = self.focal
        K[0, 2] = self.img_wh[0] / 2
        K[1, 1] = self.focal
        K[1, 2] = self.img_wh[1] / 2

        return K
