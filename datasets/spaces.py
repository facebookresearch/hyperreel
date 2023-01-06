#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import torch
import numpy as np
import json

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
from utils.pose_utils import (
    average_poses,
    correct_poses_bounds,
    create_rotating_spiral_poses,
    create_spiral_poses,
    interpolate_poses,
    center_poses_with,
)
from scipy.spatial.transform import Rotation


class SpacesDataset(Base5DDataset):
    def __init__(
        self,
        cfg,
        split='train',
        **kwargs
        ):
        super().__init__(cfg, split, **kwargs)

    def read_meta(self):
        # Load meta
        with self.pmgr.open(os.path.join(self.root_dir, "models.json"), "r") as f:
            self.meta = json.load(f)
        
        # Train and test split paths
        with self.pmgr.open(os.path.join(self.root_dir, "train_image.txt"), "r") as f:
            self.train_images = f.readlines()
            self.train_images = [
                os.path.join(self.root_dir, l.strip()) for l in self.train_images
            ]

        with self.pmgr.open(os.path.join(self.root_dir, "val_image.txt"), "r") as f:
            self.val_images = f.readlines()
            self.val_images = [
                os.path.join(self.root_dir, l.strip()) for l in self.val_images
            ]

        with self.pmgr.open(os.path.join(self.root_dir, "ref_image.txt"), "r") as f:
            self.ref_image = os.path.join(self.root_dir, f.read().split(' ')[0].strip())
        
        # Populate vars
        self.image_paths = []
        self.intrinsics = []
        self.poses = []

        for rig in self.meta:
            for camera in rig:
                image_path = os.path.join(self.root_dir, camera["relative_path"])

                if image_path not in self.train_images and image_path not in self.val_images:
                    continue

                self.image_paths.append(image_path)

                width_factor = self.img_wh[0] / camera["width"]
                height_factor = self.img_wh[1] / camera["height"]
                
                if camera["height"] != self.img_wh[1]:
                    print(camera["height"], camera['principal_point'][1])

                pa = camera["pixel_aspect_ratio"]
                K = np.eye(3)
                K = np.array(
                    [
                        [camera['focal_length'] * width_factor, 0.0, camera['principal_point'][0] * width_factor],
                        [0.0, pa * camera['focal_length'] * height_factor, camera['principal_point'][1] * height_factor],
                        [0.0, 0.0, 1.0]
                    ]
                )

                self.intrinsics.append(K)

                # Pose
                R = Rotation.from_rotvec(camera['orientation']).as_matrix()
                T = np.array(camera["position"])

                pose = np.eye(4)
                pose[:3, :3] = R.T
                pose[:3, -1] = T

                pose_pre = np.eye(4)
                pose_pre[1, 1] *= -1
                pose_pre[2, 2] *= -1
                pose = pose_pre @ pose @ pose_pre

                self.poses.append(pose[:3, :4])

        # Camera IDs & other
        self.K = self.intrinsics[0]
        self.ref_idx = self.image_paths.index(self.ref_image)
        self.intrinsics = np.stack(self.intrinsics)
        self.poses = np.stack(self.poses)

        self.camera_ids = np.linspace(0, len(self.image_paths) - 1, len(self.image_paths))
        self.total_num_views = len(self.image_paths)

        # Bounds
        with self.pmgr.open(
            os.path.join(self.root_dir, 'planes.txt'),
            'r'
        ) as f:
            planes = [float(i) for i in f.read().strip().split(' ')]

        self.bounds = np.array([planes[0], planes[1]])

        # Correct poses & bounds
        poses = np.copy(self.poses)

        self.poses, self.poses_avg = center_poses_with(
            poses, poses[self.ref_idx:self.ref_idx+1]
        )

        if not self.use_ndc:
            self.poses, self.poses_avg, self.bounds = correct_poses_bounds(
                poses, self.bounds, flip=False, center=False
            )

        self.near = self.bounds.min() * 0.95
        self.far = self.bounds.max() * 1.05
        self.depth_range = np.array([self.near * 2.0, self.far])

        # Holdout
        val_indices = [i for i in range(len(self.image_paths)) if self.image_paths[i] in self.val_images]
        train_indices = [i for i in range(len(self.image_paths)) if i not in val_indices]

        if self.val_all:
            val_indices = [i for i in train_indices] # noqa

        if self.split == 'val' or self.split == 'test':
            self.image_paths = [self.image_paths[i] for i in val_indices]
            self.camera_ids = self.camera_ids[val_indices]
            self.poses = self.poses[val_indices]
            self.intrinsics = self.intrinsics[val_indices]
        elif self.split == 'train':
            self.image_paths = [self.image_paths[i] for i in train_indices]
            self.camera_ids = self.camera_ids[train_indices]
            self.poses = self.poses[train_indices]
            self.intrinsics = self.intrinsics[train_indices]

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

        if self.split != "render":
            K = torch.FloatTensor(self.intrinsics[idx])
        else:
            K = torch.FloatTensor(self.intrinsics[0])

        print(f"Loading image {idx}")

        # Get rays
        directions = get_ray_directions_K(
            self.img_wh[1], self.img_wh[0], K, centered_pixels=True
        ).view(-1, 3)
        c2w = torch.FloatTensor(self.poses[idx])
        rays_o, rays_d = get_rays(directions, c2w)
        rays = torch.cat([rays_o, rays_d], dim=-1)

        # Convert to NDC
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
