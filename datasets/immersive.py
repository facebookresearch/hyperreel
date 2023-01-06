#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import json, csv
import os
import gc
import glob

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2

import numpy as np
import torch

from PIL import Image
from scipy.spatial.transform import Rotation

from utils.pose_utils import (
    average_poses,
    correct_poses_bounds,
    create_rotating_spiral_poses,
    create_spiral_poses,
    interpolate_poses,
    center_poses_with,
)

from utils.ray_utils import (
    get_ndc_rays_fx_fy,
    get_pixels_for_image,
    get_ray_directions_K,
    get_pixels_for_image,
    get_rays,
    sample_images_at_xy,
)

from .base import Base5DDataset, Base6DDataset


def perspective_to_fisheye(points, K, radial_distortion):
    return cv2.fisheye.undistortPoints(
        points[:, None],
        K,
        np.array([radial_distortion[0], radial_distortion[1], 0.0, 0.0]).astype(np.float32)
    )


class ImmersiveDataset(Base6DDataset):
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

        # Load meta
        with self.pmgr.open(os.path.join(self.root_dir, "models.json"), "r") as f:
            self.meta = json.load(f)

        # Populate vars
        self.video_paths = []
        self.intrinsics = []
        self.distortions = []
        self.poses = []

        for idx, camera in enumerate(self.meta):
            # Path
            self.video_paths.append(os.path.join(self.root_dir, camera["name"] + ".mp4"))

            # Intrinsics
            width_factor = self.img_wh[0] / 2560.0
            height_factor = self.img_wh[1] / 1920.0

            K = np.eye(3)
            K = np.array(
                [
                    [camera['focal_length'] * width_factor, 0.0, camera['principal_point'][0] * width_factor],
                    [0.0, camera['focal_length'] * height_factor, camera['principal_point'][1] * height_factor],
                    [0.0, 0.0, 1.0]
                ]
            )

            self.intrinsics.append(K)

            # Distortion
            radial_distortion = np.array(camera['radial_distortion'])
            self.distortions.append(radial_distortion[:2])

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

            if camera["name"] == "camera_0001":
                val_idx = idx
                center_pose = pose[None, :3, :4]

            self.poses.append(pose[:3, :4])

        self.images_per_frame = len(self.video_paths)
        self.total_num_views = len(self.video_paths)
        self.intrinsics = np.stack([self.intrinsics for i in range(self.num_frames)]).reshape(-1, 3, 3)
        self.distortions = np.stack([self.distortions for i in range(self.num_frames)]).reshape(-1, 2)
        self.poses = np.stack([self.poses for i in range(self.num_frames)]).reshape(-1, 3, 4)
        self.K = self.intrinsics[0]

        # Times
        self.times = np.tile(np.linspace(0, 1, self.num_frames)[..., None], (1, self.images_per_frame))
        self.times = self.times.reshape(-1)
        self.camera_ids = np.tile(np.linspace(0, self.images_per_frame - 1, self.images_per_frame)[None, :], (self.num_frames, 1))
        self.camera_ids = self.camera_ids.reshape(-1)

        ## Bounds, common for all scenes
        if self.dataset_cfg.collection in ['01_Welder']:
            self.near = 0.25
            self.far = 6.0

            self.bounds = np.array([self.near, self.far])
            self.depth_range = np.array([self.near * 2.0, self.far])

        if self.dataset_cfg.collection in ['02_Flames']:
            self.near = 1.0
            self.far = 10.0

            self.bounds = np.array([self.near, self.far])
            self.depth_range = np.array([self.near * 2.0, self.far])
        if self.dataset_cfg.collection in ['04_Truck']:
            self.near = 0.5
            self.far = 10.0

            self.bounds = np.array([self.near, self.far])
            self.depth_range = np.array([self.near * 2.0, self.far])
        elif self.dataset_cfg.collection in ['05_Horse']:
            self.near = 0.5
            self.far = 45.0

            self.bounds = np.array([self.near, self.far])
            self.depth_range = np.array([self.near * 2.0, self.far])
        elif self.dataset_cfg.collection in ['07_Car']:
            self.near = 0.5
            self.far = 50.0

            self.bounds = np.array([self.near, self.far])
            self.depth_range = np.array([self.near * 2.0, self.far])
        elif self.dataset_cfg.collection in ['09_Alexa_Meade_Exhibit']:
            self.near = 0.5
            self.far = 30.0

            self.bounds = np.array([self.near, self.far])
            self.depth_range = np.array([self.near * 2.0, self.far])
        elif self.dataset_cfg.collection in ['10_Alexa_Meade_Face_Paint_1']:
            self.near = 0.25
            self.far = 6.0

            self.bounds = np.array([self.near, self.far])
            #self.depth_range = np.array([self.near * 2.0, self.far])
            self.depth_range = np.array([0.5, self.far])
        elif self.dataset_cfg.collection in ['11_Alexa_Meade_Face_Paint_2']:
            self.near = 0.25
            self.far = 6.0

            self.bounds = np.array([self.near, self.far])
            #self.depth_range = np.array([self.near * 2.0, self.far])
            self.depth_range = np.array([0.5, self.far])
        elif self.dataset_cfg.collection in ['12_Cave']:
            self.near = 0.5
            self.far = 20.0

            self.bounds = np.array([self.near, self.far])
            self.depth_range = np.array([self.near * 2.0, self.far])
        else:
            self.near = 0.5
            self.far = 10.0

            self.bounds = np.array([self.near, self.far])
            self.depth_range = np.array([self.near * 2.0, self.far])

        ## Correct poses, bounds
        poses = np.copy(self.poses)

        if self.use_ndc or self.correct_poses:
            self.poses, self.poses_avg = center_poses_with(
                poses, center_pose
            )

        self.near = self.bounds.min() * 0.95
        self.far = self.bounds.max() * 1.05

        ## Holdout validation images
        val_indices = []

        if len(self.val_set) > 0:
            val_indices += [frame * self.images_per_frame + val_idx for frame in range(self.num_frames)]

        train_indices = [
            i for i in range(len(self.poses)) if i not in val_indices
        ]

        if self.val_all:
            val_indices = [i for i in train_indices]  # noqa

        if self.split == "val" or self.split == "test":
            if not self.val_all and len(self.val_set) > 0:
                self.video_paths = [self.video_paths[val_idx]]

            self.intrinsics = self.intrinsics[val_indices]
            self.camera_ids = self.camera_ids[val_indices]
            self.distortions = self.distortions[val_indices]
            self.poses = self.poses[val_indices]
            self.times = self.times[val_indices]
        elif self.split == "train":
            if not self.val_all and len(self.val_set) > 0:
                self.video_paths = [self.video_paths[i] for i in range(len(self.video_paths)) if i != val_idx]

            self.intrinsics = self.intrinsics[train_indices]
            self.camera_ids = self.camera_ids[train_indices]
            self.distortions = self.distortions[train_indices]
            self.poses = self.poses[train_indices]
            self.times = self.times[train_indices]

        self.num_images = len(self.poses)
        self.images_per_frame = len(self.video_paths)

    def random_subsample(self, coords, rgb, last_rgb, frame, fac=1.0):
        if (frame % self.load_full_step) == 0:
            return coords, rgb
        elif (frame % self.subsample_keyframe_step) == 0:
            num_take = int(np.round(coords.shape[0] * self.subsample_keyframe_frac * fac))
            perm = torch.tensor(
                np.random.permutation(coords.shape[0])
            )[:num_take]
        else:
            num_take = int(np.round(coords.shape[0] * self.subsample_frac * fac))
            perm = torch.tensor(
                np.random.permutation(coords.shape[0])
            )[:num_take]

        return coords[perm].view(-1, coords.shape[-1]), rgb[perm].view(-1, rgb.shape[-1])

    def regular_subsample(self, coords, rgb, last_rgb, frame, fac=1.0):
        if (frame % self.load_full_step) == 0:
            return coords, rgb
        elif (frame % self.subsample_keyframe_step) == 0:
            subsample_every = int(np.round(1.0 / (self.subsample_keyframe_frac * fac)))
            offset = self.keyframe_offset
            self.keyframe_offset += 1
        else:
            subsample_every = int(np.round(1.0 / (self.subsample_frac * fac)))
            offset = self.frame_offset
            self.frame_offset += 1

        pixels = get_pixels_for_image(
            self.img_wh[1], self.img_wh[0]
        ).reshape(-1, 2).long()
        mask = ((pixels[..., 0] + pixels[..., 1] + offset) % subsample_every) == 0.0

        return coords[mask].view(-1, coords.shape[-1]), rgb[mask].view(-1, rgb.shape[-1])

    def test_subsample(self, coords, rgb, last_rgb, frame):
        mask = coords[..., 5] < -0.25
        return coords[mask].view(-1, coords.shape[-1]), rgb[mask].view(-1, rgb.shape[-1])

    def importance_subsample(self, coords, rgb, last_rgb, frame, fac=1.0):
        if (frame % self.load_full_step) == 0:
            return coords, rgb

        diff = torch.abs(rgb - last_rgb).mean(-1)
        diff_sorted, _ = torch.sort(diff)

        if (frame % self.subsample_keyframe_step) == 0:
            num_take = int(np.round(coords.shape[0] * self.subsample_keyframe_frac * fac))
        else:
            num_take = int(np.round(coords.shape[0] * self.subsample_frac * fac))

        mask = diff > diff_sorted[-num_take]
        mask = mask & (coords[..., 5] < -0.05)

        return coords[mask].view(-1, coords.shape[-1]), rgb[mask].view(-1, rgb.shape[-1])

    def subsample(self, coords, rgb, last_rgb, frame):
        #coords, rgb = self.regular_subsample(coords, rgb, last_rgb, frame)
        #return coords, rgb

        if (frame % self.load_full_step) == 0:
            return coords, rgb
        else:
            coords, rgb = self.importance_subsample(coords, rgb, last_rgb, frame)

        return coords, rgb

    def prepare_train_data(self):
        ## Collect training data
        self.all_coords = []
        self.all_rgb = []
        num_pixels = 0
        last_rgb_full = None

        for video_idx in range(len(self.video_paths)):
            self.keyframe_offset = video_idx
            self.frame_offset = video_idx

            # Open video
            cam = cv2.VideoCapture(self.video_paths[video_idx])

            # Get coords
            video_coords = self.get_coords(video_idx)

            ctr = 0
            frame_idx = 0

            while ctr < self.start_frame + self.num_frames:
                _, frame = cam.read()

                if ctr < self.start_frame:
                    ctr += 1
                    continue
                else:
                    ctr += 1

                cur_time = self.times[frame_idx * self.images_per_frame + video_idx]
                cur_frame = int(np.round(self.times[frame_idx * self.images_per_frame + video_idx] * (self.num_frames - 1)))

                # Coords
                cur_coords = torch.cat(
                    [
                        video_coords[..., :-1],
                        torch.ones_like(video_coords[..., -1:]) * cur_time,
                    ],
                    -1
                )

                # Get RGB
                cur_rgb_full = self.get_rgb(frame)

                # Subsample
                if frame_idx == 0:
                    cur_rgb = cur_rgb_full
                else:
                    cur_coords, cur_rgb = self.subsample(cur_coords, cur_rgb_full, last_rgb_full, cur_frame)

                # Save for later
                last_rgb_full = cur_rgb_full

                # Coords
                self.all_coords += [cur_coords]

                # Color
                self.all_rgb += [cur_rgb]

                # Number of pixels
                num_pixels += cur_rgb.shape[0]

                print(f"Video {video_idx} frame {frame_idx}")
                print("Full res images loaded:", num_pixels / (self.img_wh[0] * self.img_wh[1]))

                # Increment frame idx
                frame_idx += 1

            cam.release()

        # Format / save loaded data
        self.all_coords = torch.cat(self.all_coords, 0)
        #self.all_coords = self.all_coords.view(
        #    -1, self.images_per_frame, self.num_frames, self.all_coords.shape[-1]
        #).permute(0, 2, 1, 3).reshape(-1, self.all_coords.shape[-1])
        self.all_rgb = torch.cat(self.all_rgb, 0)
        #self.all_rgb = self.all_rgb.view(
        #    -1, self.images_per_frame, self.num_frames, self.all_rgb.shape[-1]
        #).permute(0, 2, 1, 3).reshape(-1, self.all_rgb.shape[-1])
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
            radii = np.percentile(np.abs(poses_one_frame[..., 3]), 50, axis=0)
            radii[..., :-1] *= 1.0
            radii[..., -1] *= 0.05

            if self.num_frames > 1:
                poses = create_spiral_poses(
                    poses_one_frame,
                    radii,
                    focus_depth,
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
        if self.split != 'train' or self.split == 'render':
            camera_id = 1
        else:
            camera_id = self.camera_ids[idx]

        if self.split != 'render':
            K = torch.FloatTensor(self.intrinsics[idx])
            distortion = self.distortions[idx]
        else:
            K = torch.FloatTensor(self.intrinsics[0])
            K[0, 0] *= 0.75
            K[1, 1] *= 0.75
            distortion = None

        c2w = torch.FloatTensor(self.poses[idx])
        time = self.times[idx]

        print("Loading time:", np.round(time * (self.num_frames - 1)))

        # Undistort
        if distortion is not None:
            directions = get_ray_directions_K(
                self.img_wh[1], self.img_wh[0], K, centered_pixels=True
            ).view(-1, 3)
            directions = perspective_to_fisheye(
                np.array(directions[..., :2]).astype(np.float32),
                np.eye(3).astype(np.float32),
                distortion.astype(np.float32)
            )[:, 0]
            directions = np.concatenate(
                [
                    directions[..., 0:1],
                    directions[..., 1:2],
                    -np.ones_like(directions[..., -1:])
                ],
                -1
            )

            #directions = get_pixels_for_image(
            #    self.img_wh[1], self.img_wh[0]
            #).reshape(-1, 2)
            #directions = perspective_to_fisheye(
            #    np.array(directions[..., :2]).astype(np.float32),
            #    np.array(K).astype(np.float32),
            #    distortion.astype(np.float32)
            #)[:, 0]
            #directions = np.concatenate(
            #    [
            #        directions[..., 0:1],
            #        -directions[..., 1:2],
            #        -np.ones_like(directions[..., -1:])
            #    ],
            #    -1
            #)

            directions = torch.tensor(directions)
            directions = torch.nn.functional.normalize(directions, dim=-1)
        else:
            directions = get_ray_directions_K(
                self.img_wh[1], self.img_wh[0], K, centered_pixels=True
            ).view(-1, 3)

        # Convert to world space
        rays_o, rays_d = get_rays(directions, c2w)

        # Convert to NDC
        if self.use_ndc:
            rays = self.to_ndc(torch.cat([rays_o, rays_d], dim=-1))
        else:
            rays = torch.cat([rays_o, rays_d], dim=-1)

        # Add camera idx
        rays = torch.cat([rays, torch.ones_like(rays[..., :1]) * camera_id], dim=-1)

        # Add times
        rays = torch.cat([rays, torch.ones_like(rays[..., :1]) * time], dim=-1)

        # Return
        return rays

    def get_rgb(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if img.shape[0] != self._img_wh[0] or img.shape[1] != self._img_wh[1]:
            img = cv2.resize(img, self._img_wh, cv2.INTER_LANCZOS4)

        if self.img_wh[0] != self._img_wh[0] or self.img_wh[1] != self._img_wh[1]:
            img = cv2.resize(img, self.img_wh, cv2.INTER_AREA)

        img = self.transform(img)
        img = img.view(3, -1).permute(1, 0)

        return img

    def get_rgb_one(self, idx):
        # Open video
        cam = cv2.VideoCapture(self.video_paths[idx % self.images_per_frame])

        # Get RGB
        ctr = 0
        frame_idx = 0

        while ctr < self.start_frame + self.num_frames:
            _, frame = cam.read()

            if ctr < self.start_frame:
                ctr += 1
                continue
            else:
                ctr += 1

            if frame_idx != (idx // self.images_per_frame):
                frame_idx += 1
                continue
            else:
                rgb = self.get_rgb(frame)
                break

        cam.release()
        return rgb

    def get_intrinsics(self):
        return self.intrinsics

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
                "rgb": self.get_rgb_one(idx),
                "idx": idx,
            }

            batch["weight"] = torch.ones_like(batch["coords"][..., -1:])
        elif self.split == "val":
            batch = {
                "coords": self.get_coords(idx),
                "rgb": self.get_rgb_one(idx),
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
