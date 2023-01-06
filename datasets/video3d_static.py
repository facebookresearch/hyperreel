#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import json
import os

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2

import numpy as np
import torch

from PIL import Image

from utils.pose_utils import (
    average_poses,
    correct_poses_bounds,
    create_rotating_spiral_poses,
    create_spiral_poses,
    interpolate_poses,
)

from utils.ray_utils import (
    get_ndc_rays_fx_fy,
    get_pixels_for_image,
    get_ray_directions_K,
    get_rays,
    sample_images_at_xy,
)

from .base import Base5DDataset


class Video3DDataset(Base5DDataset):
    def __init__(self, cfg, split="train", **kwargs):
        self.use_reference = (
            cfg.dataset.use_reference if "use_reference" in cfg.dataset else False
        )
        self.correct_poses = (
            cfg.dataset.correct_poses if "correct_poses" in cfg.dataset else False
        )
        self.use_ndc = cfg.dataset.use_ndc if "use_ndc" in cfg.dataset else False

        super().__init__(cfg, split, **kwargs)

    def read_meta(self):
        W, H = self.img_wh

        ## Image paths
        self.image_paths = sorted(self.pmgr.ls(os.path.join(self.root_dir, "images")))

        ## Load poses
        self.pose_paths = sorted(self.pmgr.ls(os.path.join(self.root_dir, "cameras")))

        poses = []

        for i, pose_path in enumerate(self.pose_paths):
            with self.pmgr.open(
                os.path.join(self.root_dir, "cameras", pose_path), "r"
            ) as f:
                meta = json.load(f)

            if i == 0:
                self.meta = meta
                self.focal_x = self.meta["normalized_focal_length_x"]
                self.focal_y = self.meta["normalized_focal_length_y"]
                self.principal_point_x = self.meta["normalized_principal_point_x"]
                self.principal_point_y = self.meta["normalized_principal_point_y"]

                # Correct pose
                if "reference_world_to_camera" in meta and self.use_reference:
                    self.reference_matrix = np.array(meta["reference_world_to_camera"])
                else:
                    self.reference_matrix = np.eye(4)

            frame_matrix = np.array(meta["camera_to_world"])
            pose = (self.reference_matrix @ frame_matrix)[:3, :4]
            poses += [pose]

        poses = np.stack(poses, axis=0)

        ## Intrinsics
        self.K = np.eye(3)
        self.K[0, 0] = self.focal_x * W
        self.K[0, 2] = self.principal_point_x * W
        self.K[1, 1] = self.focal_y * H
        self.K[1, 2] = self.principal_point_x * H

        ## Bounds, common for all scenes
        self.near = 0.75
        self.far = 4.0
        self.bounds = np.array([self.near, self.far])

        ## Correct poses, bounds
        if self.use_ndc or self.correct_poses:
            self.poses, self.poses_avg, self.bounds = correct_poses_bounds(
                poses, self.bounds, flip=False, center=True
            )
        else:
            self.poses = poses

        self.near = self.bounds.min() * 0.95
        self.far = self.bounds.max() * 1.05

        ## Ray directions for all pixels, same for all images (same H, W, focal)
        self.centered_pixels = True
        self.directions = get_ray_directions_K(
            H, W, self.K, centered_pixels=self.centered_pixels
        )

        ## Holdout validation images
        if self.val_set == "lightfield":
            step = self.dataset_cfg.lightfield_step
            cols = self.dataset_cfg.lightfield_cols
            val_indices = []

            self.val_pairs = self.dataset_cfg.val_pairs if 'val_pairs' in self.dataset_cfg else []
            self.val_all = ((step == 1 and len(self.val_pairs) == 0) or self.val_all)

            for idx, path in enumerate(self.image_paths):
                n = int(path.split('_')[-1].split('.')[0])
                row = n // cols
                col = n % cols

                if row % step != 0 or col % step != 0 or ((row, col) in self.val_pairs):
                    val_indices.append(idx)

        elif len(self.val_set) > 0:
            val_indices = self.val_set
        elif self.val_skip != "inf":
            self.val_skip = min(len(self.image_paths), self.val_skip)
            val_indices = list(range(0, len(self.image_paths), self.val_skip))
        else:
            val_indices = []

        train_indices = [
            i for i in range(len(self.image_paths)) if i not in val_indices
        ]

        if self.val_all:
            val_indices = [i for i in train_indices]  # noqa

        if self.split == "val" or self.split == "test":
            self.image_paths = [self.image_paths[i] for i in val_indices]
            self.poses = self.poses[val_indices]
        elif self.split == "train":
            self.image_paths = [self.image_paths[i] for i in train_indices]
            self.poses = self.poses[train_indices]

    def prepare_render_data(self):
        if not self.render_interpolate:
            close_depth, inf_depth = self.bounds.min() * 0.9, self.bounds.max() * 5.0

            dt = 0.75
            mean_dz = 1.0 / (((1.0 - dt) / close_depth + dt / inf_depth))
            focus_depth = mean_dz

            radii = np.percentile(np.abs(self.poses[..., 3]), 50, axis=0)
            camera_radius = 0.35

            # self.poses = create_rotating_spiral_poses(
            #    [0.0, -0.2, 0.0],
            #    self.poses,
            #    camera_radius,
            #    [0.0, radii[1], camera_radius * 0.25],
            #    focus_depth * 100,
            #    [-1.0, 1.0],
            #    N=360
            # )
            self.poses = create_rotating_spiral_poses(
                [0.0, 0.0, 0.0],
                self.poses,
                camera_radius,
                [0.0, radii[1], camera_radius * 0.25],
                focus_depth * 100,
                [-1.0, 1.0],
                N=360,
            )
            # self.poses = create_rotating_spiral_poses(
            #    [0.0, 0.0, 0.35],
            #    self.poses,
            #    camera_radius,
            #    [0.0, radii[1], camera_radius * 0.25],
            #    focus_depth * 100,
            #    [-0.2, 0.2]
            # )

            self.poses = np.stack(self.poses, axis=0)
        else:
            self.poses = interpolate_poses(self.poses, self.render_supersample)

    def to_ndc(self, rays):
        return get_ndc_rays_fx_fy(
            self.img_wh[1], self.img_wh[0], self.K[0, 0], self.K[1, 1], self.near, rays
        )

    def get_coords(self, idx):
        c2w = torch.FloatTensor(self.poses[idx])
        rays_o, rays_d = get_rays(self.directions, c2w)

        if self.use_ndc:
            return self.to_ndc(torch.cat([rays_o, rays_d], dim=-1))
        else:
            return torch.cat([rays_o, rays_d], dim=-1)

    def get_rgb(self, idx):
        image_path = self.image_paths[idx]

        with self.pmgr.open(
            os.path.join(self.root_dir, "images", image_path), "rb"
        ) as im_file:
            img = Image.open(im_file).convert("RGBA")

        img = img.resize(self._img_wh, Image.LANCZOS)

        if self.img_wh[0] != self._img_wh[0] or self.img_wh[1] != self._img_wh[1]:
            img = img.resize(self.img_wh, Image.BOX)

        img = self.transform(img)
        img = img.view(4, -1).permute(1, 0)
        img = img[:, :3] * img[:, -1:] + (1 - img[:, -1:])

        return img

    def get_intrinsics(self):
        K = np.eye(3)
        K[0, 0] = self.focal_x * self.img_wh[0]
        K[0, 2] = self.principal_point_x * self.img_wh[0]
        K[1, 1] = self.focal_y * self.img_wh[1]
        K[1, 2] = self.principal_point_x * self.img_wh[1]

        return K

