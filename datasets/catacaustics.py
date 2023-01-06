#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import torch
import numpy as np
from PIL import Image
#import open3d as o3d

from .llff import LLFFDataset

from utils.pose_utils import (
    interpolate_poses,
    correct_poses_bounds,
    create_spiral_poses,
    center_poses_with
)

from utils.ray_utils import (
    get_ray_directions_K,
    get_ndc_rays_fx_fy,
    get_ray_directions_K,
    get_rays,
)

from utils.intersect_utils import intersect_axis_plane

import sys
from pathlib import Path


def readBundleFolder(cameras_folder, W, H, extension=".png", name_ints=8):
    poses = []
    intrinsics = []
    image_paths = []

    with open(os.path.join(cameras_folder, "bundle.out")) as bundle_file:
        # First line is a comment
        _ = bundle_file.readline()
        num_cameras, _ = [int(x) for x in bundle_file.readline().split()]

        for idx in range(num_cameras):
            cam_name = '{num:0{width}}'.format(num=idx, width=name_ints) + extension
            focal, dist0, dist1 = [float(x) for x in bundle_file.readline().split()]

            # Rotation
            R = []

            for i in range(3):
                R.append([float(x) for x in bundle_file.readline().split()])

            R = np.array(R).reshape(3, 3)

            # Translation
            T = [float(x) for x in bundle_file.readline().split()]
            T = np.array(T)

            # Pose
            pose = np.eye(4)
            pose[:3, :3] = R
            pose[:3, -1] = T
            #pose[:3, :3] = R.T
            #pose[:3, -1] = -R.T @ T.T
            pose = np.linalg.inv(pose)

            pose_pre = np.eye(4)
            #pose_pre[1, 1] *= -1
            #pose_pre[2, 2] *= -1

            pose = pose_pre @ pose @ pose_pre

            poses.append(pose[:3])

            # Intrinsics
            image_path = os.path.join(cameras_folder, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            K = np.eye(3)
            K[0, 0] = focal * W / float(image.size[0])
            K[0, 2] = W / 2.0
            K[1, 1] = focal * H / float(image.size[1])
            K[1, 2] = H / 2.0
            intrinsics.append(K)

            # TODO:
            # 1) Poses
            # 2) Intrinsics
            # 3) Model settings

            # Image
            image_path = os.path.join(cameras_folder, cam_name)
            image_paths.append(image_path)
    
    return np.stack(poses, 0), np.stack(intrinsics, 0), image_paths


class CatacausticsDataset(LLFFDataset):
    def __init__(
        self,
        cfg,
        split='train',
        **kwargs
        ):
        self.use_reference = (
            cfg.dataset.use_reference if "use_reference" in cfg.dataset else False
        )
        self.correct_poses = (
            cfg.dataset.correct_poses if "correct_poses" in cfg.dataset else False
        )
        self.use_ndc = cfg.dataset.use_ndc if "use_ndc" in cfg.dataset else False

        super().__init__(cfg, split, **kwargs)

    def read_meta(self):
        self.train_cameras_folder = os.path.join(self.root_dir, "cropped_train_cameras")
        self.validation_cameras_folder = os.path.join(self.root_dir, "validation_cameras")
        self.test_cameras_folder = os.path.join(self.root_dir, "test_path_cameras")

        train_poses, train_intrinsics, train_image_paths = readBundleFolder(
            self.train_cameras_folder, self.img_wh[0], self.img_wh[1]
        )
        validation_poses, validation_intrinsics, validation_image_paths = readBundleFolder(
            self.validation_cameras_folder, self.img_wh[0], self.img_wh[1], name_ints=5
        )
        test_poses, test_intrinsics, test_image_paths = readBundleFolder(
            self.test_cameras_folder, self.img_wh[0], self.img_wh[1], name_ints=5
        )

        self.poses_dict = {
            "train": train_poses,
            "render": test_poses,
            "val": validation_poses,
            "test": test_poses,
        }
        self.poses = np.stack(self.poses_dict[self.split], 0)

        self.intrinsics_dict = {
            "train": train_intrinsics,
            "render": test_intrinsics,
            "val": validation_intrinsics,
            "test": test_intrinsics,
        }
        self.intrinsics = np.stack(self.intrinsics_dict[self.split], 0)
        self.K = self.intrinsics_dict["train"][0]

        self.image_paths_dict = {
            "train": train_image_paths,
            "render": test_image_paths,
            "val": validation_image_paths,
            "test": test_image_paths,
        }
        self.image_paths = self.image_paths_dict[self.split]

        # Geometry
        print("Reading Point-Cloud...")

        pcd = o3d.io.read_point_cloud(os.path.join(self.root_dir, "meshes", "dense_point_cloud.ply"))
        self.bbox_center = np.array(pcd.get_center())
        points = np.array(pcd.points)

        min_dist = np.linalg.norm(points - self.bbox_center[None], axis=-1).min()
        max_dist = np.linalg.norm(points - self.bbox_center[None], axis=-1).max()
        fac = 8.0 / (min_dist + max_dist)

        min_dist = min_dist * fac
        max_dist = max_dist * fac
        self.bbox_center = self.bbox_center * fac
        self.bbox_min = np.array(pcd.get_min_bound()) * fac - self.bbox_center
        self.bbox_max = np.array(pcd.get_max_bound()) * fac - self.bbox_center

        self.depth_range = [min_dist, max_dist]

        # Change poses
        self.poses[..., -1] = self.poses[..., -1] * fac - self.bbox_center

        # Bounds
        self.near = min_dist
        self.far = max_dist
        self.bounds = np.array([self.near, self.far])

        self.near = self.bounds.min() * 0.95
        self.far = self.bounds.max() * 1.05

        ## Correct poses
        #poses = np.copy(self.poses)
        #train_poses = np.stack(self.poses_dict["train"])

        #if self.use_ndc or self.correct_poses:
        #    self.poses, self.poses_avg = center_poses_with(
        #        poses, np.stack(self.poses_dict["train"][:1])
        #    )
        #    train_poses, _ = center_poses_with(
        #        np.copy(train_poses), np.stack(self.poses_dict["train"][:1])
        #    )
        #    #self.poses, self.poses_avg = center_poses_with(
        #    #    poses, np.stack(self.poses_dict["train"])
        #    #)
        #    #train_poses, _ = center_poses_with(
        #    #    np.copy(train_poses), np.stack(self.poses_dict["train"])
        #    #)
        
        #sc = np.max(np.abs(train_poses[..., -1]))
        #self.poses[..., -1] /= sc

        #filter_idx = np.argwhere(self.poses[..., 2, 2] > 0.75).astype(np.int32).reshape(-1).tolist()
        #self.image_paths = [self.image_paths[i] for i in filter_idx]
        #self.poses = self.poses[filter_idx]
        #self.intrinsics = self.intrinsics[filter_idx]

    def prepare_render_data(self):
        self.prepare_test_data()
        
    def prepare_train_data(self):
        self.num_images = len(self.image_paths)

        ## Collect training data
        self.all_coords = []
        self.all_rgb = []

        for idx in range(len(self.image_paths)):
        #for idx in range(1):
            # coords
            self.all_coords += [self.get_coords(idx)]

            # Color
            self.all_rgb += [self.get_rgb(idx)]

        # Format / save loaded data
        self.update_all_data(
            torch.cat(self.all_coords, 0),
            torch.cat(self.all_rgb, 0),
        )

    def update_all_data(self, coords, rgb):
        self.all_coords = coords
        self.all_rgb = rgb
        self.all_weights = self.get_weights()

        ## Patches
        if self.use_patches or self.use_crop:
            self._all_coords = torch.clone(self.all_coords)
            self._all_rgb = torch.clone(self.all_rgb)

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

    def to_ndc(self, rays):
        return get_ndc_rays_fx_fy(
            self.img_wh[1], self.img_wh[0], self.K[0, 0], self.K[1, 1], self.near, rays
        )

    def get_coords(self, idx):
        K = torch.FloatTensor(self.intrinsics[idx])
        c2w = torch.FloatTensor(self.poses[idx])

        directions = get_ray_directions_K(
            self.img_wh[1], self.img_wh[0], K, centered_pixels=True
        )
        rays_o, rays_d = get_rays(directions, c2w)

        if self.use_ndc:
            rays = self.to_ndc(torch.cat([rays_o, rays_d], dim=-1))
        else:
            rays = torch.cat([rays_o, rays_d], dim=-1)

        return rays

    def get_rgb(self, idx):
        image_path = self.image_paths[idx]

        print(f"Loading image {idx}")

        with self.pmgr.open(os.path.join(self.root_dir, "images", image_path), "rb") as im_file:
            img = Image.open(im_file)
            img = img.convert("RGBA")

        img = img.resize(self._img_wh)

        if self.img_wh[0] != self._img_wh[0] or self.img_wh[1] != self._img_wh[1]:
            img = img.resize(self.img_wh, Image.BOX)
        
        img = self.transform(img)
        img = img.view(4, -1).permute(1, 0)
        img = img[:, :3] * img[:, -1:] + (1 - img[:, -1:])

        return img

    def get_intrinsics(self):
        return self.intrinsics

    def __getitem__(self, idx):
        if self.split == "render":
            batch = {
                "coords": self.get_coords(idx),
                "pose": self.poses[idx],
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