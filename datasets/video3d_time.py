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

from .base import Base6DDataset


class Video3DTimeDataset(Base6DDataset):
    def __init__(self, cfg, split="train", **kwargs):
        self.use_reference = (
            cfg.dataset.use_reference if "use_reference" in cfg.dataset else False
        )
        self.correct_poses = (
            cfg.dataset.correct_poses if "correct_poses" in cfg.dataset else False
        )

        self.use_ndc = cfg.dataset.use_ndc if "use_ndc" in cfg.dataset else False

        self.num_frames = cfg.dataset.num_frames if "num_frames" in cfg.dataset else 1
        self.start_frame = cfg.dataset.start_frame if "start_frame" in cfg.dataset else 1
        self.keyframe_step = cfg.dataset.keyframe_step if "keyframe_step" in cfg.dataset else 1
        self.num_keyframes = cfg.dataset.num_keyframes if "num_keyframes" in cfg.dataset else self.num_frames // self.keyframe_step

        self.load_full_step = cfg.dataset.load_full_step if "load_full_step" in cfg.dataset else 1
        self.subsample_keyframe_step = cfg.dataset.subsample_keyframe_step if "subsample_keyframe_step" in cfg.dataset else 1
        self.subsample_keyframe_frac = cfg.dataset.subsample_keyframe_frac if "subsample_keyframe_frac" in cfg.dataset else 1.0
        self.subsample_frac = cfg.dataset.subsample_frac if "subsample_frac" in cfg.dataset else 1.0

        self.keyframe_offset = 0
        self.frame_offset = 0

        super().__init__(cfg, split, **kwargs)

    def read_meta(self):
        W, H = self.img_wh

        # Image paths
        self.num_rows = self.dataset_cfg.lightfield_rows
        self.num_cols = self.dataset_cfg.lightfield_cols

        rows = self.num_rows
        cols = self.num_cols

        self.images_per_frame = rows * cols
        self.total_num_views = rows * cols

        # Video paths
        self.frame_paths = sorted(self.pmgr.ls(os.path.join(self.root_dir)))[self.start_frame:self.start_frame+self.num_frames]

        # Image and pose paths
        self.image_paths = []
        self.pose_paths = []

        for frame_path in self.frame_paths:
            all_paths = sorted(self.pmgr.ls(os.path.join(self.root_dir, frame_path)))

            self.image_paths += [os.path.join(self.root_dir, frame_path, p) for p in all_paths if p.endswith(".png")]
            self.pose_paths += [os.path.join(self.root_dir, frame_path, p) for p in all_paths if p.endswith(".json")]

        # Set up poses, times, frames
        self.poses = []
        self.times = []
        self.frames = []
        self.intrinsics = []

        for i, pose_path in enumerate(self.pose_paths):
            with self.pmgr.open(pose_path, "r") as f:
                meta = json.load(f)

            if "frame" in meta:
                frame = meta["frame"]
            else:
                frame = int(pose_path.split("/")[-2].split("frame_")[-1])

            # Intrinsics
            if i == 0:
                self.global_meta = meta
                self.start_frame = frame
                self.end_frame = self.start_frame + self.num_frames - 1

            # Intrinsics
            focal_x = meta["normalized_focal_length_x"]
            focal_y = meta["normalized_focal_length_y"]
            principal_point_x = meta["normalized_principal_point_x"]
            principal_point_y = meta["normalized_principal_point_y"]

            K = np.eye(3)
            K[0, 0] = focal_x * W
            K[0, 2] = principal_point_x * W
            K[1, 1] = focal_y * H
            K[1, 2] = principal_point_y * H
            self.intrinsics.append(K)

            # Get current pose
            frame_matrix = np.array(meta["camera_to_world"])
            pose = frame_matrix[:3, :4]
            self.poses += [pose]

            # Time
            self.times.append(
                (frame - self.start_frame) / (self.num_frames - 1)
            )
            self.frames.append(frame - self.start_frame)

        self.poses = np.stack(self.poses, axis=0)
        self.intrinsics = np.stack(self.intrinsics, axis=0)
        self.times = np.array(self.times)
        self.frames = np.array(self.frames)
        self.K = self.intrinsics[0]

        ## Bounds, common for all scenes
        self.near = self.global_meta["near_clip"]
        self.far = self.global_meta["far_clip"]
        self.bounds = np.array([self.near, self.far])

        ## Correct poses, bounds
        if self.use_ndc or self.correct_poses:
            self.poses, self.poses_avg, self.bounds = correct_poses_bounds(
                np.copy(self.poses), self.bounds, flip=False, center=True
            )

        self.near = self.bounds.min() * 0.95
        self.far = self.bounds.max() * 1.05
        self.depth_range = np.array([2 * self.near, self.far])

        ## Holdout validation images
        if self.val_set == "lightfield":
            step = self.dataset_cfg.lightfield_step
            rows = self.dataset_cfg.lightfield_rows
            cols = self.dataset_cfg.lightfield_cols
            val_indices = []

            self.val_pairs = self.dataset_cfg.val_pairs if 'val_pairs' in self.dataset_cfg else []
            self.val_all = ((step == 1 and len(self.val_pairs) == 0) or self.val_all)

            for row in range(rows):
                for col in range(cols):
                    idx = row * rows + col

                    if row % step != 0 or col % step != 0 or ([row, col] in self.val_pairs):
                        val_indices += [frame * self.images_per_frame + idx for frame in range(self.num_frames)]

            if self.val_num > 0:
                val_indices = val_indices[:self.val_num]

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
            self.intrinsics = self.intrinsics[val_indices]
            self.frames = self.frames[val_indices]
            self.times = self.times[val_indices]
        elif self.split == "train":
            self.image_paths = [self.image_paths[i] for i in train_indices]
            self.poses = self.poses[train_indices]
            self.intrinsics = self.intrinsics[train_indices]
            self.frames = self.frames[train_indices]
            self.times = self.times[train_indices]

    def subsample(self, coords, rgb, frame):
        if (frame % self.load_full_step) == 0:
            return coords, rgb
        elif (frame % self.subsample_keyframe_step) == 0:
            subsample_every = int(np.round(1.0 / self.subsample_keyframe_frac))
            offset = self.keyframe_offset
            self.keyframe_offset += 1
            #num_take = int(np.round(coords.shape[0] * self.subsample_keyframe_frac))
            #perm = torch.tensor(
            #    np.random.permutation(coords.shape[0])
            #)[:num_take]
        else:
            subsample_every = int(np.round(1.0 / self.subsample_frac))
            offset = self.frame_offset
            self.frame_offset += 1
            #num_take = int(np.round(coords.shape[0] * self.subsample_frac))
            #perm = torch.tensor(
            #    np.random.permutation(coords.shape[0])
            #)[:num_take]

        #return coords[perm].view(-1, coords.shape[-1]), rgb[perm].view(-1, rgb.shape[-1])
        pixels = get_pixels_for_image(
            self.img_wh[1], self.img_wh[0]
        ).reshape(-1, 2).long()
        mask = ((pixels[..., 0] + pixels[..., 1] + offset) % subsample_every) == 0.0
        return coords[mask].view(-1, coords.shape[-1]), rgb[mask].view(-1, rgb.shape[-1])

    def prepare_train_data(self, reset=False):
        self.num_images = len(self.image_paths)

        ## Collect training data
        self.all_coords = []
        self.all_rgb = []
        num_pixels = 0

        for idx in range(len(self.image_paths)):
        #for idx in range(1):
            cur_coords = self.get_coords(idx)
            cur_rgb = self.get_rgb(idx)
            cur_frame = int(np.round(self.times[idx] * (self.num_frames - 1)))

            # Subsample
            cur_coords, cur_rgb = self.subsample(cur_coords, cur_rgb, cur_frame)

            # Coords
            self.all_coords += [cur_coords]

            # Color
            self.all_rgb += [cur_rgb]

            # Number of pixels
            num_pixels += cur_rgb.shape[0]

            print("Full res images loaded:", num_pixels / (self.img_wh[0] * self.img_wh[1]))

        # Format / save loaded data
        self.all_coords = torch.cat(self.all_coords, 0)
        self.all_rgb = torch.cat(self.all_rgb, 0)
        self.update_all_data()

    def update_all_data(self):
        self.all_weights = self.get_weights()

        ## All inputs
        self.all_inputs = torch.cat(
            [
                self.all_coords,
                self.all_rgb,
                self.all_weights,
            ],
            -1,
        )

    def format_batch(self, batch):
        batch["coords"] = batch["inputs"][..., : self.all_coords.shape[-1]]
        batch["rgb"] = batch["inputs"][
            ..., self.all_coords.shape[-1] : self.all_coords.shape[-1] + 3
        ]
        batch["weight"] = batch["inputs"][..., -1:]
        del batch["inputs"]

        return batch

    def prepare_render_data(self):
        # Get poses
        if not self.render_interpolate:
            close_depth, inf_depth = self.bounds.min() * 0.9, self.bounds.max() * 5.0

            dt = 0.75
            mean_dz = 1.0 / (((1.0 - dt) / close_depth + dt / inf_depth))
            focus_depth = mean_dz

            poses_per_frame = self.poses.shape[0] // self.num_frames
            poses_one_frame = self.poses[
                (self.num_frames // 2)
                * poses_per_frame : (self.num_frames // 2 + 1)
                * poses_per_frame
            ]
            poses_each_frame = interpolate_poses(
                self.poses[::poses_per_frame], self.render_supersample
            )
            radii = np.percentile(np.abs(poses_one_frame[..., 3]), 60, axis=0)

            if self.num_frames > 1:
                poses = create_spiral_poses(
                    poses_one_frame,
                    radii,
                    focus_depth * 100,
                    N=self.num_frames * self.render_supersample,
                )

                reference_pose = np.eye(4)
                reference_pose[:3, :4] = self.poses[
                    (self.num_frames // 2) * poses_per_frame
                ]
                reference_pose = np.linalg.inv(reference_pose)

                for pose_idx in range(len(poses)):
                    cur_pose = np.eye(4)
                    cur_pose[:3, :4] = poses[pose_idx]
                    poses[pose_idx] = poses_each_frame[pose_idx] @ (
                        reference_pose @ cur_pose
                    )
            else:
                poses = create_spiral_poses(
                    poses_one_frame, radii, focus_depth * 100, N=120
                )

            self.poses = np.stack(poses, axis=0)
        else:
            self.poses = interpolate_poses(self.poses, self.render_supersample)

        # Get times
        if (self.num_frames - 1) > 0:
            self.times = np.linspace(0, self.num_frames - 1, len(self.poses))

            if not self.render_interpolate_time:
                self.times = np.round(self.times)

            self.times = self.times / (self.num_frames - 1)
        else:
            self.times = [0.0 for p in self.poses]

    def to_ndc(self, rays):
        return get_ndc_rays_fx_fy(
            self.img_wh[1], self.img_wh[0], self.K[0, 0], self.K[1, 1], self.near, rays
        )

    def get_coords(self, idx):
        if self.split != 'train' and not self.val_all:
            cam_idx = 3
        else:
            cam_idx = idx % self.images_per_frame

        if self.split != 'render':
            K = torch.FloatTensor(self.intrinsics[idx])
        else:
            K = torch.FloatTensor(self.intrinsics[0])

        c2w = torch.FloatTensor(self.poses[idx])
        time = self.times[idx]

        print("Loading time:", np.round(time * (self.num_frames - 1)))

        directions = get_ray_directions_K(
            self.img_wh[1], self.img_wh[0], K, centered_pixels=True
        )

        # Convert to world space / NDC
        rays_o, rays_d = get_rays(directions, c2w)

        if self.use_ndc:
            rays = self.to_ndc(torch.cat([rays_o, rays_d], dim=-1))
        else:
            rays = torch.cat([rays_o, rays_d], dim=-1)

        # Add camera idx
        rays = torch.cat([rays, torch.ones_like(rays[..., :1]) * cam_idx], dim=-1)

        # Add times
        rays = torch.cat([rays, torch.ones_like(rays[..., :1]) * time], dim=-1)

        # Return
        return rays

    def get_rgb(self, idx):
        image_path = self.image_paths[idx]

        with self.pmgr.open(os.path.join(self.root_dir, "images", image_path), "rb") as im_file:
            img = Image.open(im_file)
            img = img.convert("RGB")

        if img.size[0] != self._img_wh[0] or img.size[1] != self._img_wh[1]:
            img = img.resize(self._img_wh, Image.LANCZOS)

        if self.img_wh[0] != self._img_wh[0] or self.img_wh[1] != self._img_wh[1]:
            img = img.resize(self.img_wh, Image.BOX)

        img = self.transform(img)
        img = img.view(3, -1).permute(1, 0)

        #img = img.view(4, -1).permute(1, 0)
        #img = img[:, :3] * img[:, -1:] + (1 - img[:, -1:])

        return img

    def get_intrinsics(self):
        return self.K

    def __getitem__(self, idx):
        if self.split == "render":
            batch = {
                "coords": self.get_coords(idx),
                "pose": self.poses[idx],
                "time": self.times[idx],
                "idx": idx,
            }

            batch["weight"] = torch.ones_like(batch["coords"][..., -1:])

        elif self.split == "test":
            batch = {
                "coords": self.get_coords(idx),
                "rgb": self.get_rgb(idx),
                "idx": idx,
            }

            batch["weight"] = torch.ones_like(batch["coords"][..., -1:])
        elif self.split == "val":
            batch = {
                "coords": self.get_coords(idx),
                "rgb": self.get_rgb(idx),
                "idx": idx,
            }

            batch["weight"] = torch.ones_like(batch["coords"][..., -1:])
        else:
            batch = {
                "inputs": self.all_inputs[idx],
            }

        W, H, batch = self.crop_batch(batch)
        batch["W"] = W
        batch["H"] = H

        return batch
