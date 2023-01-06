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


class Video3DTimeGroundTruthDataset(Base6DDataset):
    def __init__(self, cfg, split="train", **kwargs):
        self.use_reference = (
            cfg.dataset.use_reference if "use_reference" in cfg.dataset else False
        )
        self.correct_poses = (
            cfg.dataset.correct_poses if "correct_poses" in cfg.dataset else False
        )
        self.use_ndc = cfg.dataset.use_ndc if "use_ndc" in cfg.dataset else False
        self.num_keyframes = cfg.dataset.num_keyframes if "num_keyframes" in cfg.dataset else -1

        super().__init__(cfg, split, **kwargs)

    def read_meta(self):
        W, H = self.img_wh

        self.frame_paths = sorted(self.pmgr.ls(os.path.join(self.root_dir)))
        self.num_frames = len(self.frame_paths)
        if self.num_keyframes == -1:
            self.num_keyframes = self.num_frames
        self.keyframe_step = self.num_frames // self.num_keyframes

        ## Image and pose paths
        self.image_paths = []
        self.pose_paths = []
        self.depth_paths = []

        for frame_path in self.frame_paths:
            all_paths = sorted(self.pmgr.ls(os.path.join(self.root_dir, frame_path)))
            image_paths = [p for p in all_paths if p.endswith(".png")]
            pose_paths = [p for p in all_paths if p.endswith(".json")]
            depth_paths = [p for p in all_paths if p.endswith("_depth")]

            image_paths = [os.path.join(frame_path, p) for p in image_paths]
            pose_paths = [os.path.join(frame_path, p) for p in pose_paths]
            depth_paths = [os.path.join(frame_path, p) for p in depth_paths]

            self.image_paths += image_paths
            self.pose_paths += pose_paths
            self.depth_paths += depth_paths

        ## Load poses
        poses = []
        self.reference_matrix = []
        self.times = []
        self.frames = []

        for i, pose_path in enumerate(self.pose_paths):
            with self.pmgr.open(os.path.join(self.root_dir, pose_path), "r") as f:
                meta = json.load(f)

            if "frame" in meta:
                frame = meta["frame"]
            else:
                frame = int(pose_path.split("/")[-2].split("frame_")[-1])

            # Intrinsics
            if i == 0:
                self.meta = meta
                self.focal_x = self.meta["normalized_focal_length_x"]
                self.focal_y = self.meta["normalized_focal_length_y"]
                self.principal_point_x = self.meta["normalized_principal_point_x"]
                self.principal_point_y = self.meta["normalized_principal_point_y"]
                self.start_frame = frame
                self.end_frame = self.start_frame + self.num_frames - 1

            # Reference matrix
            if self.use_reference:
                self.reference_matrix.append(np.array(meta["world_to_camera"])[:3, :4])
            else:
                self.reference_matrix = np.eye(4)

        # Reference matrix
        if self.use_reference:
            self.reference_matrix = average_poses(np.stack(self.reference_matrix, 0))

        # Get all poses
        for i, pose_path in enumerate(self.pose_paths):
            with self.pmgr.open(os.path.join(self.root_dir, pose_path), "r") as f:
                meta = json.load(f)

            if "frame" in meta:
                frame = meta["frame"]
            else:
                frame = int(pose_path.split("/")[-2].split("frame_")[-1])

            frame_matrix = np.array(meta["camera_to_world"])
            pose = (self.reference_matrix @ frame_matrix)[:3, :4]
            poses += [pose]

            # Time
            if self.num_frames - 1 > 0:
                self.times.append(
                    (frame - self.start_frame) / (self.num_frames - 1)
                )
                self.frames.append(frame - self.start_frame)
            else:
                self.times.append(0.0)
                self.frames.append(0)

        poses = np.stack(poses, axis=0)
        self.times = np.array(self.times)

        ## Intrinsics
        self.K = np.eye(3)
        self.K[0, 0] = self.focal_x * W
        self.K[0, 2] = self.principal_point_x * W
        self.K[1, 1] = self.focal_y * H
        self.K[1, 2] = self.principal_point_x * H

        ## Bounds, common for all scenes
        # self.near = meta['near_clip']
        # self.far = meta['far_clip']
        self.near = 0.25
        self.far = 10.0
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
            rows = self.dataset_cfg.lightfield_rows
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
            self.depth_paths = [self.depth_paths[i] for i in val_indices]
            self.poses = self.poses[val_indices]
            self.times = self.times[val_indices]
            self.frames = [self.frames[i] for i in val_indices]
        elif self.split == "train":
            self.image_paths = [self.image_paths[i] for i in train_indices]
            self.depth_paths = [self.depth_paths[i] for i in train_indices]
            self.poses = self.poses[train_indices]
            self.times = self.times[train_indices]
            self.frames = [self.frames[i] for i in train_indices]

    def prepare_train_data(self):
        self.num_images = len(self.image_paths)

        ## Collect training data
        self.all_coords = []
        self.all_rgb = []
        self.all_depth = []
        self.all_pixel_flow = []
        self.all_flow = []

        for idx in range(len(self.image_paths)):
        #for idx in range(75, 76): # TODO: Remove
            # coords
            self.all_coords += [self.get_coords(idx)]

            # Color
            self.all_rgb += [self.get_rgb(idx)]

            # Depth
            self.all_depth += [self.get_depth(idx)]

            # Flow
            self.all_pixel_flow += [self.get_pixel_flow(idx)]
            self.all_flow += [self.get_flow(idx)]

        # Format / save loaded data
        self.update_all_data(
            torch.cat(self.all_coords, 0),
            torch.cat(self.all_rgb, 0),
            torch.cat(self.all_depth, 0),
            torch.cat(self.all_flow, 0),
        )

    def update_all_data(self, coords, rgb, depth, flow):
        self.all_coords = coords
        self.all_rgb = rgb
        self.all_depth = depth
        self.all_flow = flow
        self.all_weights = self.get_weights()

        ## Patches
        if self.use_patches or self.use_crop:
            self._all_coords = torch.clone(self.all_coords)
            self._all_rgb = torch.clone(self.all_rgb)
            self._all_depth = torch.clone(self.all_depth)
            self._all_flow = torch.clone(self.all_flow)

        ## All inputs
        self.all_inputs = torch.cat(
            [
                self.all_coords,
                self.all_rgb,
                self.all_depth,
                self.all_flow,
                self.all_weights,
            ],
            -1,
        )

    def format_batch(self, batch):
        batch["coords"] = batch["inputs"][..., : self.all_coords.shape[-1]]
        batch["rgb"] = batch["inputs"][
            ..., self.all_coords.shape[-1] : self.all_coords.shape[-1] + 3
        ]
        batch["depth"] = batch["inputs"][
            ..., self.all_coords.shape[-1] + 3 : self.all_coords.shape[-1] + 4
        ]
        batch["flow"] = batch["inputs"][
            ..., self.all_coords.shape[-1] + 4 : self.all_coords.shape[-1] + 7
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
            radii = np.percentile(np.abs(poses_one_frame[..., 3]), 80, axis=0)

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
        c2w = torch.FloatTensor(self.poses[idx])
        time = self.times[idx]
        print("Loading time:", np.round(time * (self.num_frames - 1)))
        rays_o, rays_d = get_rays(self.directions, c2w)

        if self.use_ndc:
            rays = self.to_ndc(torch.cat([rays_o, rays_d], dim=-1))
        else:
            rays = torch.cat([rays_o, rays_d], dim=-1)

        rays = torch.cat([rays, torch.ones_like(rays[..., :1]) * time], dim=-1)
        return rays

    def get_rgb(self, idx):
        image_path = self.image_paths[idx]

        with self.pmgr.open(os.path.join(self.root_dir, image_path), "rb") as im_file:
            img = Image.open(im_file).convert("RGBA")

        img = img.resize(self._img_wh, Image.LANCZOS)

        if self.img_wh[0] != self._img_wh[0] or self.img_wh[1] != self._img_wh[1]:
            img = img.resize(self.img_wh, Image.BOX)

        img = self.transform(img)
        img = img.view(4, -1).permute(1, 0)
        img = img[:, :3] * img[:, -1:] + (1 - img[:, -1:])

        return img

    def load_geometry(self, idx, prefix="depth", mode="exr"):
        gt_path = os.path.join(
            self.root_dir, self.depth_paths[idx].replace("depth", prefix)
        )
        gt_image_path = [p for p in self.pmgr.ls(gt_path) if p.endswith(mode)][0]

        depth_file = os.path.join(gt_path, gt_image_path)

        if mode == "exr":
            img = cv2.imread(depth_file, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        else:
            img = np.load(depth_file)

        # Resize
        img = cv2.resize(img, self._img_wh, interpolation=cv2.INTER_NEAREST)

        if self.img_wh[0] != self._img_wh[0] or self.img_wh[1] != self._img_wh[1]:
            img = cv2.resize(img, self.img_wh, interpolation=cv2.INTER_NEAREST)

        # Transform
        img = self.transform(np.copy(img))
        return img.view(img.shape[0], -1).permute(1, 0)

    def get_depth(self, idx, return_mask=False):
        depth = self.load_geometry(idx, "depth")[..., 0:1]

        directions = torch.nn.functional.normalize(self.directions, p=2.0, dim=-1).view(
            -1, 3
        )
        depth = depth / torch.abs(directions[..., 2:3])

        mask = (depth < self.near) | (depth > self.far)
        depth[depth < self.near] = self.near
        depth[depth > self.far] = self.far

        if return_mask:
            return depth, mask
        else:
            return depth

    def get_pixel_flow(self, idx):
        # Pixel flow
        pixel_flow = self.load_geometry(idx, "vector")[..., 1:3]
        pixel_flow = torch.flip(pixel_flow, [1])
        pixel_flow[..., 1] *= -1
        pixel_flow = pixel_flow * self.img_wh[0] / 800

        return pixel_flow

    def get_uv(self, idx):
        # Pixel flow
        uv = self.load_geometry(idx, "uv")[..., 1:3]
        uv = torch.flip(uv[1:3], [-1])

        return uv

    def get_flow(self, idx):
        # Flow
        return self.load_geometry(idx, "vector", mode='npy')

    def get_intrinsics(self):
        K = np.eye(3)
        K[0, 0] = self.focal_x * self.img_wh[0]
        K[0, 2] = self.principal_point_x * self.img_wh[0]
        K[1, 1] = self.focal_y * self.img_wh[1]
        K[1, 2] = self.principal_point_x * self.img_wh[1]

        return K

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
                "coords": self.get_coords(idx + 75),
                "rgb": self.get_rgb(idx + 75),
                "depth": self.get_depth(idx + 75),
                "flow": self.get_flow(idx + 75),
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
