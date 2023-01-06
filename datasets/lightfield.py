#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import numpy as np

from .base import BaseDataset, Base5DDataset
from utils.ray_utils import (
    get_lightfield_rays
)


class LightfieldDataset(Base5DDataset):
    def __init__(
        self,
        cfg,
        split='train',
        **kwargs
        ):

        ## Dataset cfg
        self.cfg = cfg
        self.split = getattr(cfg.dataset, 'split', split)
        self.dataset_cfg = getattr(cfg.dataset, self.split, cfg.dataset)

        ## Param

        # Lightfield params
        self.rows = self.dataset_cfg.lightfield.rows
        self.cols = self.dataset_cfg.lightfield.cols
        self.step = self.dataset_cfg.lightfield.step

        self.start_row = self.dataset_cfg.lightfield.start_row if 'start_row' in self.dataset_cfg.lightfield else 0
        self.end_row = self.dataset_cfg.lightfield.end_row if 'end_row' in self.dataset_cfg.lightfield else self.rows

        self.start_col = self.dataset_cfg.lightfield.start_col if 'start_col' in self.dataset_cfg.lightfield else 0
        self.end_col = self.dataset_cfg.lightfield.end_col if 'end_col' in self.dataset_cfg.lightfield else self.cols

        self.st_scale = self.dataset_cfg.lightfield.st_scale if 'st_scale' in self.dataset_cfg.lightfield else 1.0
        self.uv_scale = self.dataset_cfg.lightfield.uv_scale if 'uv_scale' in self.dataset_cfg.lightfield else 1.0

        if self.step > 1:
            self.num_rows = ((self.end_row - self.start_row) // self.step + 1)
            self.num_cols = ((self.end_col - self.start_col) // self.step + 1)
        else:
            self.num_rows = (self.end_row - self.start_row) // self.step
            self.num_cols = (self.end_col - self.start_col) // self.step

        self.num_images = self.num_rows * self.num_cols

        self.near = 0
        self.far = 1
        self.near_plane = self.dataset_cfg.lightfield.near if 'near' in self.dataset_cfg.lightfield else -1.0
        self.far_plane = self.dataset_cfg.lightfield.far if 'far' in self.dataset_cfg.lightfield else 0.0

        # Validation and testing
        self.val_all = (self.dataset_cfg.val_all if 'val_all' in self.dataset_cfg else False) or self.step == 1
        self.val_pairs = self.dataset_cfg.val_pairs if 'val_pairs' in self.dataset_cfg else []

        if len(self.val_pairs) > 0:
            self.val_pairs = list(zip(self.val_pairs[::2], self.val_pairs[1::2]))
            self.num_test_images = len(self.val_pairs)
        elif self.val_all:
            self.num_test_images = (self.end_row - self.start_row) * (self.end_col - self.start_col)
        else:
            self.num_test_images = (self.end_row - self.start_row) * (self.end_col - self.start_col) - self.num_images

        # Render params
        self.disp_row = self.dataset_cfg.lightfield.disp_row
        self.supersample = self.dataset_cfg.lightfield.supersample
        self.keyframe_step = self.dataset_cfg.lightfield.keyframe_step if 'keyframe_step' in self.dataset_cfg.lightfield else -1
        self.keyframe_subsample = self.dataset_cfg.lightfield.keyframe_subsample if 'keyframe_subsample' in self.dataset_cfg.lightfield else 1

        self.render_spiral = self.dataset_cfg.render_params.spiral if 'spiral' in self.dataset_cfg.render_params else False
        self.render_far = self.dataset_cfg.render_params.far if 'far' in self.dataset_cfg.render_params else False

        self.spiral_rad = self.dataset_cfg.render_params.spiral_rad if 'spiral_rad' in self.dataset_cfg.render_params else 0.5
        self.uv_downscale = self.dataset_cfg.render_params.uv_downscale if 'uv_downscale' in self.dataset_cfg.render_params else 0.0

        if 'vis_st_scale' in self.dataset_cfg.lightfield:
            self.vis_st_scale = self.dataset_cfg.lightfield.vis_st_scale \
                if self.dataset_cfg.lightfield.vis_st_scale is not None else self.st_scale
        else:
            self.vis_st_scale = self.st_scale

        if 'vis_uv_scale' in self.dataset_cfg.lightfield:
            self.vis_uv_scale = self.dataset_cfg.lightfield.vis_uv_scale \
                if self.dataset_cfg.lightfield.vis_uv_scale is not None else self.uv_scale
        else:
            self.vis_uv_scale = self.uv_scale

        super().__init__(cfg, split, val_all=self.val_all, **kwargs)

        self.poses = [self.get_coord(st_idx) for st_idx in self.all_st_idx]

    def read_meta(self):
        pass

    def prepare_train_data(self):
        self.all_coords = []
        self.all_rgb = []
        self.all_st_idx = []

        for t_idx in range(self.start_row, self.end_row, self.step):
            for s_idx in range(self.start_col, self.end_col, self.step):
                if (s_idx, t_idx) in self.val_pairs:
                    continue

                # Rays
                self.all_coords += [self.get_coords(s_idx, t_idx)]

                idx = t_idx * self.cols + s_idx
                image_path = self.image_paths[idx]

                print(image_path)
                print(self.all_coords[0][0])
                exit()

                # Color
                self.all_rgb += [self.get_rgb(s_idx, t_idx)]

                # Random subsample for frames that are not keyframes
                # TODO: Re-do every N iterations
                if self.keyframe_step != -1 and self.keyframe_subsample != 1:
                    num_take = self.all_coords[-1].shape[0] // self.keyframe_subsample

                    if (s_idx % self.keyframe_step != 0) or (t_idx % self.keyframe_step != 0):
                        perm = torch.tensor(
                            np.random.permutation(self.all_coords[-1].shape[0])
                        )[:num_take]

                        self.all_coords[-1] = self.all_coords[-1][perm].view(-1, 6)
                        self.all_rgb[-1] = self.all_rgb[-1][perm].view(-1, 3)

                # Pose
                self.all_st_idx.append((s_idx, t_idx))

        self.all_coords = torch.cat(self.all_coords, 0)
        self.all_rgb = torch.cat(self.all_rgb, 0)
        self.all_weights = self.get_weights()
        self.all_inputs = torch.cat(
            [self.all_coords, self.all_rgb, self.all_weights], -1
        )

    def prepare_val_data(self):
        self.prepare_test_data()

    def prepare_test_data(self):
        self.all_st_idx = []

        for t_idx in range(self.start_row, self.end_row, 1):
            for s_idx in range(self.start_col, self.end_col, 1):
                if len(self.val_pairs) == 0:
                    if (t_idx % self.step) == 0 and (s_idx % self.step) == 0 \
                        and not self.val_all:
                        continue
                elif (s_idx, t_idx) not in self.val_pairs:
                    continue

                self.all_st_idx.append((s_idx, t_idx))

    def prepare_render_data(self):
        if not self.render_spiral:
            self.all_st_idx = []
            t_idx = self.disp_row

            for s_idx in range(self.cols * self.supersample):
                self.all_st_idx.append((s_idx / self.supersample, t_idx))
        else:
            N = 120
            rots = 2
            scale = self.spiral_rad

            self.all_st_idx = []

            for theta in np.linspace(0., 2. * np.pi * rots, N+1)[:-1]:
                s = (np.cos(theta) * scale + 1) / 2.0 * (self.cols - 1)
                t = -np.sin(theta) * scale / 2.0 * (self.rows - 1) + ((self.rows - 1) - self.disp_row)

                self.all_st_idx.append((s, t))

    def get_coord(self, st_idx):
        s = (st_idx[0] / (self.cols - 1)) * 2 - 1 \
            if self.cols > 1 else 0
        t = -(((st_idx[1] / (self.rows - 1)) * 2 - 1) \
            if self.rows > 1 else 0)

        return (s, t)

    def get_coords(self, s_idx, t_idx):
        if self.split == 'render':
            st_scale = self.vis_st_scale
            uv_scale = self.vis_uv_scale
        else:
            st_scale = self.st_scale
            uv_scale = self.uv_scale

        s, t = self.get_coord((s_idx, t_idx))

        if self.render_spiral or self.render_far:
            return get_lightfield_rays(
                self.img_wh[0], self.img_wh[1], s, t, self.aspect,
                st_scale=st_scale,
                uv_scale=uv_scale,
                near=self.near_plane, far=self.far_plane,
                use_inf=True, center_u=-s*self.uv_downscale, center_v=-t*self.uv_downscale
            )
        else:
            return get_lightfield_rays(
                self.img_wh[0], self.img_wh[1], s, t, self.aspect,
                near=self.near_plane, far=self.far_plane,
                st_scale=st_scale,
                uv_scale=uv_scale,
            )

    def get_rgb(self, s_idx, t_idx):
        pass

    def get_closest_rgb(self, query_st):
        W = self.img_wh[0]
        H = self.img_wh[1]

        images = self.all_rgb.view(self.num_images, H, W, -1)
        dists = np.linalg.norm(
            np.array(self.poses) - np.array(query_st)[None], axis=-1
        )
        return images[list(np.argsort(dists))[0]]

    def __len__(self):
        if self.split == 'train':
            return len(self.all_coords)
        elif self.split == 'val':
            return min(self.val_num, self.num_test_images)
        elif self.split == 'render':
            if not self.render_spiral:
                return self.supersample * self.cols
            else:
                return 120
        else:
            return self.num_test_images

    def __getitem__(self, idx):
        if self.split == 'render':
            s_idx, t_idx = self.all_st_idx[idx]

            batch = {
                'coords': LightfieldDataset.get_coords(self, s_idx, t_idx),
                'pose': self.poses[idx],
                'idx': idx,
                's_idx': s_idx,
                't_idx': t_idx,
            }

        elif self.split == 'val' or self.split == 'test':
            s_idx, t_idx = self.all_st_idx[idx]

            batch = {
                'coords': self.get_coords(s_idx, t_idx),
                'rgb': self.get_rgb(s_idx, t_idx),
                'idx': idx,
                's_idx': s_idx,
                't_idx': t_idx,
            }
        else:
            batch = {
                'inputs': self.all_inputs[idx],
            }

        W, H, batch = self.crop_batch(batch)
        batch['W'] = W
        batch['H'] = H

        return batch


class EPIDataset(BaseDataset):
    def __init__(
        self,
        cfg,
        split='train',
        **kwargs
        ):

        ## Dataset cfg
        self.cfg = cfg
        self.split = getattr(cfg.dataset, 'split', split)
        self.dataset_cfg = getattr(cfg.dataset, self.split, cfg.dataset)

        # Lightfield params
        self.st_scale = self.dataset_cfg.lightfield.st_scale if 'st_scale' in self.dataset_cfg.lightfield else 1.0
        self.supersample = self.dataset_cfg.lightfield.supersample if self.split == 'render' else 1

        super().__init__(cfg, split, **kwargs)

    def read_meta(self):
        pass

    def prepare_train_data(self):
        self.all_coords = []
        self.all_rgb = []

        # Rays
        self.all_coords += [self.get_coords()]

        # Color
        self.all_rgb += [self.get_rgb()]

        # Stack
        self.all_coords = torch.cat(self.all_coords, 0)
        self.all_rgb = torch.cat(self.all_rgb, 0)
        self.all_weights = self.get_weights()

        self.all_inputs = torch.cat(
            [self.all_coords, self.all_rgb, self.all_weights], -1
        )

    def prepare_val_data(self):
        self.prepare_test_data()

    def prepare_test_data(self):
        self.prepare_train_data()

    def prepare_render_data(self):
        self.all_coords = []
        self.all_rgb = []

        # Rays
        self.all_coords += [self.get_coords()]

        # Color
        self.all_rgb += [self.get_rgb()]

        # Stack
        self.all_coords = torch.cat(self.all_coords, 0)
        self.all_rgb = torch.cat(self.all_rgb, 0)
        self.all_weights = self.get_weights()

    def get_coords(self):
        u = torch.linspace(-1, 1, self.img_wh[0], dtype=torch.float32)
        s = torch.linspace(-1, 1, self.img_wh[1] * self.supersample, dtype=torch.float32) * self.st_scale
        su = list(torch.meshgrid([s, u]))
        return torch.stack(su, -1).view(-1, 2)

    def get_rgb(self):
        # TODO: return single image
        pass

    def get_closest_rgb(self, query_st):
        pass

    def __len__(self):
        if self.split == 'train':
            return len(self.all_coords)
        elif self.split == 'val':
            return 1
        elif self.split == 'render':
            return 1
        else:
            return 1

    def __getitem__(self, idx):
        if self.split == 'render':
            batch = {
                'coords': self.get_coords(),
            }

        elif self.split == 'val' or self.split == 'test':
            batch = {
                'coords': self.get_coords(),
                'rgb': self.get_rgb(),
            }
        else:
            batch = {
                'inputs': self.all_inputs[idx],
            }

        W, H, batch = self.crop_batch(batch)
        batch['W'] = W
        batch['H'] = H

        if self.split == 'render':
            batch['H'] *= self.supersample

        return batch
