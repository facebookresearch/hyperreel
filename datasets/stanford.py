#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os

import torch
import numpy as np

from PIL import Image

from .lightfield import EPIDataset, LightfieldDataset
from .llff import LLFFDataset

from utils.pose_utils import (
    interpolate_poses,
    correct_poses_bounds,
    create_spiral_poses,
)

from utils.ray_utils import (
    get_rays,
    get_ray_directions_K,
    get_ndc_rays_fx_fy,
    get_lightfield_rays
)


class StanfordLightfieldDataset(LightfieldDataset):
    def __init__(
        self,
        cfg,
        split='train',
        **kwargs
        ):
        self.use_file_coords = cfg.dataset.lightfield.use_file_coords if 'use_file_coords' in cfg.dataset.lightfield else False

        super().__init__(cfg, split, **kwargs)

        if self.split == 'train' and self.use_file_coords:
            self.poses = []

            for (s_idx, t_idx) in self.all_st_idx:
                idx = t_idx * self.cols + s_idx
                coord = self.normalize_coord(self.camera_coords[idx])
                self.poses.append(coord)

    def read_meta(self):
        self.image_paths = sorted(
            self.pmgr.ls(self.root_dir)
        )

        if self.img_wh is None:
            image_path = self.image_paths[0]

            with self.pmgr.open(
                os.path.join(self.root_dir, image_path),
                'rb'
            ) as im_file:
                img = np.array(Image.open(im_file).convert('RGB'))

            self._img_wh = (img.shape[1] // self.downsample, img.shape[0] // self.downsample)
            self.img_wh = (img.shape[1] // self.downsample, img.shape[0] // self.downsample)
            self.aspect = (float(self.img_wh[0]) / self.img_wh[1])

        self.camera_coords = []

        if self.use_file_coords:
            for image_path in self.image_paths:
                if self.dataset_cfg.collection in ['beans', 'knights', 'tarot', 'tarot_small']:
                    yx = image_path.split('_')[-2:]
                    y = -float(yx[0])
                    x = float(yx[1].split('.png')[0])
                else:
                    yx = image_path.split('_')[-3:-1]
                    y, x = float(yx[0]), float(yx[1])

                self.camera_coords.append((x, y))

    def get_camera_range(self):
        xs = [coord[0] for coord in self.camera_coords]
        ys = [coord[1] for coord in self.camera_coords]

        min_x, max_x = np.min(xs), np.max(xs)
        min_y, max_y = np.min(ys), np.max(ys)

        return (min_x, max_x), (min_y, max_y)

    def get_camera_center(self):
        idx = (self.rows // 2) * self.cols + self.cols // 2
        return self.camera_coords[idx]

    def normalize_coord(self, coord):
        x_range, y_range = self.get_camera_range()

        #x_c, y_c = self.get_camera_center()
        #norm_x = 2 * (coord[0] - x_c) / (x_range[1] - x_range[0])
        #norm_y = 2 * (coord[1] - y_c) / (x_range[1] - x_range[0])

        aspect = (x_range[1] - x_range[0]) / (y_range[1] - y_range[0])
        norm_x = ((coord[0] - x_range[0]) / (x_range[1] - x_range[0])) * 2 - 1
        norm_y = (((coord[1] - y_range[0]) / (y_range[1] - y_range[0])) * 2 - 1) / aspect

        return (norm_x, norm_y)

    def get_coords(self, s_idx, t_idx):
        if not self.use_file_coords:
            return super().get_coords(s_idx, t_idx)

        idx = t_idx * self.cols + s_idx
        coord = self.normalize_coord(self.camera_coords[idx])

        if self.split == 'render':
            st_scale = self.vis_st_scale
        else:
            st_scale = self.st_scale

        return get_lightfield_rays(
            self.img_wh[0], self.img_wh[1],
            coord[0], coord[1],
            self.aspect,
            near=self.near_plane, far=self.far_plane,
            st_scale=st_scale,
            uv_scale=self.uv_scale
        )

    def get_rgb(self, s_idx, t_idx):
        idx = t_idx * self.cols + s_idx
        image_path = self.image_paths[idx]

        with self.pmgr.open(
            os.path.join(self.root_dir, image_path),
            'rb'
        ) as im_file:
            img = Image.open(im_file).convert('RGB')

        img = img.resize(self._img_wh, Image.LANCZOS)

        if self.img_wh[0] != self._img_wh[0] or self.img_wh[1] != self._img_wh[1]:
            img = img.resize(self.img_wh, Image.BOX)

        img = self.transform(img)
        img = img.view(3, -1).permute(1, 0)

        return img


class StanfordEPIDataset(EPIDataset):
    def __init__(
        self,
        cfg,
        split='train',
        **kwargs
        ):

        super().__init__(cfg, split, **kwargs)

    def read_meta(self):
        self.image_paths = sorted(
            self.pmgr.ls(self.root_dir)
        )

        if self.img_wh is None:
            image_path = self.image_paths[0]

            with self.pmgr.open(
                os.path.join(self.root_dir, image_path),
                'rb'
            ) as im_file:
                img = np.array(Image.open(im_file).convert('RGB'))

            self._img_wh = (img.shape[1] // self.downsample, img.shape[0] // self.downsample)
            self.img_wh = (img.shape[1] // self.downsample, img.shape[0] // self.downsample)
            self.aspect = (float(self.img_wh[0]) / self.img_wh[1])

    def get_coords(self):
        if self.dataset_cfg.collection in ['tarot_small', 'tarot', 'chess']:
            u = torch.linspace(-1, 1, self.img_wh[0], dtype=torch.float32)
            s = torch.linspace(1, -1, self.img_wh[1] * self.supersample, dtype=torch.float32) * self.st_scale
        else:
            u = torch.linspace(-1, 1, self.img_wh[0], dtype=torch.float32)
            s = torch.linspace(-1, 1, self.img_wh[1] * self.supersample, dtype=torch.float32) * self.st_scale

        su = list(torch.meshgrid([s, u]))
        return torch.stack(su, -1).view(-1, 2)

    def get_rgb(self):
        image_path = self.image_paths[0]

        with self.pmgr.open(
            os.path.join(self.root_dir, image_path),
            'rb'
        ) as im_file:
            img = Image.open(im_file).convert('RGB')

        img = img.resize(self._img_wh, Image.LANCZOS)

        if self.img_wh[0] != self._img_wh[0] or self.img_wh[1] != self._img_wh[1]:
            img = img.resize(self.img_wh, Image.BOX)

        img = self.transform(img)
        img = img.view(3, -1).permute(1, 0)

        return img


class StanfordEPIDataset(EPIDataset):
    def __init__(
        self,
        cfg,
        split='train',
        **kwargs
        ):

        super().__init__(cfg, split, **kwargs)

    def read_meta(self):
        self.image_paths = sorted(
            self.pmgr.ls(self.root_dir)
        )

        if self.img_wh is None:
            image_path = self.image_paths[0]

            with self.pmgr.open(
                os.path.join(self.root_dir, image_path),
                'rb'
            ) as im_file:
                img = np.array(Image.open(im_file).convert('RGB'))

            self._img_wh = (img.shape[1] // self.downsample, img.shape[0] // self.downsample)
            self.img_wh = (img.shape[1] // self.downsample, img.shape[0] // self.downsample)
            self.aspect = (float(self.img_wh[0]) / self.img_wh[1])

    def get_coords(self):
        if self.dataset_cfg.collection in ['tarot_small', 'tarot', 'chess']:
            u = torch.linspace(-1, 1, self.img_wh[0], dtype=torch.float32)
            s = torch.linspace(1, -1, self.img_wh[1] * self.supersample, dtype=torch.float32) * self.st_scale
        else:
            u = torch.linspace(-1, 1, self.img_wh[0], dtype=torch.float32)
            s = torch.linspace(-1, 1, self.img_wh[1] * self.supersample, dtype=torch.float32) * self.st_scale

        su = list(torch.meshgrid([s, u]))
        return torch.stack(su, -1).view(-1, 2)

    def get_rgb(self):
        image_path = self.image_paths[0]

        with self.pmgr.open(
            os.path.join(self.root_dir, image_path),
            'rb'
        ) as im_file:
            img = Image.open(im_file).convert('RGB')

        img = img.resize(self._img_wh, Image.LANCZOS)

        if self.img_wh[0] != self._img_wh[0] or self.img_wh[1] != self._img_wh[1]:
            img = img.resize(self.img_wh, Image.BOX)

        img = self.transform(img)
        img = img.view(3, -1).permute(1, 0)

        return img


class StanfordLLFFDataset(LLFFDataset):
    def __init__(
        self,
        cfg,
        split='train',
        **kwargs
        ):

        # Scale of ST plane relative to UV plane
        st_scale_dict = {
            'tarot': 0.125,
            'tarot_small': 0.125,
            'knights': 0.125,
            'bracelet': 0.125,
        }

        if 'st_scale' in cfg.dataset:
            self.st_scale = cfg.dataset.st_scale
        else:
            self.st_scale = st_scale_dict.get(cfg.dataset.collection, 1.0)

        # Near, far plane locations
        self.near_plane = cfg.dataset.near if 'near' in cfg.dataset else -1.0
        self.far_plane = cfg.dataset.far if 'far' in cfg.dataset else 0.0

        super().__init__(cfg, split, **kwargs)

    def read_meta(self):
        # Camera coords
        self.image_paths = sorted(
            self.pmgr.ls(self.root_dir)
        )

        # Get width, height
        if self.img_wh is None:
            image_path = self.image_paths[0]

            with self.pmgr.open(
                os.path.join(self.root_dir, image_path),
                'rb'
            ) as im_file:
                img = np.array(Image.open(im_file).convert('RGB'))

            self._img_wh = (img.shape[1] // self.downsample, img.shape[0] // self.downsample)
            self.img_wh = (img.shape[1] // self.downsample, img.shape[0] // self.downsample)
            self.aspect = (float(self.img_wh[0]) / self.img_wh[1])

        # Get camera coords
        self.camera_coords = []

        for image_path in self.image_paths:
            if self.dataset_cfg.collection in ['beans', 'knights', 'tarot', 'tarot_small']:
                yx = image_path.split('_')[-2:]
                y = -float(yx[0])
                x = float(yx[1].split('.png')[0])
            else:
                yx = image_path.split('_')[-3:-1]
                y, x = float(yx[0]), float(yx[1])

            self.camera_coords.append((x, y))
        
        self.camera_coords = np.array(self.camera_coords)
        self.camera_min = np.min(self.camera_coords, axis=0)
        self.camera_max = np.max(self.camera_coords, axis=0)

        self.camera_coords = (self.camera_coords - self.camera_min) / (self.camera_max - self.camera_min) * 2 - 1
        st_aspect = (self.camera_max[0] - self.camera_min[0]) / (self.camera_max[1] - self.camera_min[1])
        self.camera_coords[:, 1] /= st_aspect
        self.camera_coords *= self.st_scale

        # Set up poses
        self.poses = np.tile(np.eye(4, 4)[..., None], [1, 1, len(self.image_paths)])
        self.poses[:, 1:3, :] *= -1
        self.poses[:2, 3, :] = self.camera_coords.T
        self.poses[2, 3, :] = self.near_plane
        self.poses = self.poses.transpose(2, 0, 1)
        self.poses = self.poses[:, :3, :4]

        # Set up intrinsics
        focal = 1
        pixel_scale = self.img_wh[0] / 2

        self.intrinsics = np.tile(np.eye(3)[..., None], [1, 1, len(self.image_paths)])
        self.intrinsics[0, 0, :] = focal * pixel_scale
        self.intrinsics[1, 1, :] = focal * pixel_scale
        self.intrinsics[0, 2, :] = self.camera_coords.T[0] * focal * pixel_scale + self.img_wh[0] / 2
        self.intrinsics[1, 2, :] = -self.camera_coords.T[1] * focal * pixel_scale + self.img_wh[1] / 2
        self.intrinsics = self.intrinsics.transpose(2, 0, 1)

        self.K = np.eye(3)
        self.K[0, 0] = focal * pixel_scale
        self.K[1, 1] = focal * pixel_scale
        self.K[0, 2] = self.img_wh[0] / 2
        self.K[1, 2] = self.img_wh[1] / 2

        ## Correct poses, bounds
        self.bounds = np.array([0.25, 2.0])

        if self.use_ndc:
            self.poses, self.poses_avg, self.bounds = correct_poses_bounds(
                np.copy(self.poses), np.copy(self.bounds), flip=False, center=True
            )

        self.near = self.bounds.min() * 0.95
        self.far = self.bounds.max() * 1.05
        self.depth_range = np.array([self.near * 2.0, self.far])

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

                    if (row % step != 0 or col % step != 0 or ([row, col] in self.val_pairs)) and not self.val_all:
                        val_indices += [idx]

        elif len(self.val_set) > 0 or self.val_all:
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
            self.intrinsics = self.intrinsics[val_indices]
            self.poses = self.poses[val_indices]
        elif self.split == "train":
            self.image_paths = [self.image_paths[i] for i in train_indices]
            self.intrinsics = self.intrinsics[train_indices]
            self.poses = self.poses[train_indices]

    def prepare_train_data(self, reset=False):
        self.num_images = len(self.image_paths)

        ## Collect training data
        self.all_coords = []
        self.all_rgb = []
        num_pixels = 0

        for idx in range(len(self.image_paths)):
            cur_coords = self.get_coords(idx)
            cur_rgb = self.get_rgb(idx)

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

            radii = np.percentile(
                np.abs(self.poses[..., 3] - np.mean(self.poses[..., 3], axis=0)),
                50,
                axis=0
            )
            self.poses = create_spiral_poses(self.poses, radii, focus_depth * 4)

            self.poses = np.stack(self.poses, axis=0)
            self.poses[..., :3, 3] = self.poses[..., :3, 3] - 0.1 * close_depth * self.poses[..., :3, 2]
        else:
            self.poses = interpolate_poses(self.poses, self.render_supersample)

    def get_coords(self, idx):
        if self.split != 'train' and not self.val_all:
            cam_idx = 0
        else:
            cam_idx = idx

        if self.split != 'render':
            K = torch.FloatTensor(self.intrinsics[idx])
        else:
            K = torch.FloatTensor(np.copy(self.K))

        c2w = torch.FloatTensor(self.poses[idx])

        directions = get_ray_directions_K(
            self.img_wh[1], self.img_wh[0], K, centered_pixels=False, flipped=True
        )

        # Convert to world space / NDC
        rays_o, rays_d = get_rays(directions, c2w)

        if self.use_ndc:
            rays = self.to_ndc(torch.cat([rays_o, rays_d], dim=-1))
        else:
            rays = torch.cat([rays_o, rays_d], dim=-1)

        # Add camera idx
        rays = torch.cat([rays, torch.ones_like(rays[..., :1]) * cam_idx], dim=-1)

        # Return
        return rays

    def get_rgb(self, idx):
        image_path = self.image_paths[idx]

        with self.pmgr.open(os.path.join(self.root_dir, image_path), "rb") as im_file:
            img = Image.open(im_file)
            img = img.convert("RGB")

        if img.size[0] != self._img_wh[0] or img.size[1] != self._img_wh[1]:
            img = img.resize(self._img_wh, Image.LANCZOS)

        if self.img_wh[0] != self._img_wh[0] or self.img_wh[1] != self._img_wh[1]:
            img = img.resize(self.img_wh, Image.BOX)
        
        img = self.transform(img)
        img = img.view(3, -1).permute(1, 0)

        return img
