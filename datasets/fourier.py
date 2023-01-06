#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import numpy as np
from torch.utils.data import Dataset

from utils.ray_utils import get_lightfield_rays


def fft_rgb(rgb):
    return torch.stack(
        [
            torch.fft.fft2(rgb[..., 0], norm='ortho'),
            torch.fft.fft2(rgb[..., 1], norm='ortho'),
            torch.fft.fft2(rgb[..., 2], norm='ortho'),
        ],
        dim=-1,
    )


class FourierDataset(Dataset):
    def __init__(
        self,
        cfg,
        train_dataset=None,
        **kwargs
    ):
        super().__init__()

        self.cfg = cfg
        self.img_wh = train_dataset.img_wh
        self.width, self.height = self.img_wh[0], self.img_wh[1]
        self.aspect = train_dataset.aspect
        self.num_images = train_dataset.num_images
        self.batch_size = cfg.batch_size

        self.all_rays = torch.clone(train_dataset.all_rays)
        self.all_rgb = torch.clone(train_dataset.all_rgb)

        # Prepare
        self.compute_stats()
        self.prepare_data()
        self.shuffle()

    def compute_stats(self):
        all_rays = self.all_rays.view(
            self.num_images, self.img_wh[1] * self.img_wh[0], -1
        )
        ray_dim = all_rays.shape[-1] // 2

        ## Per view statistics
        self.all_means = []
        self.all_stds = []

        for idx in range(self.num_images):
            cur_rays = all_rays[idx]
            self.all_means += [cur_rays.mean(0)]
            self.all_stds += [cur_rays.std(0)]

        self.all_means = torch.stack(self.all_means, 0)
        self.all_stds = torch.stack(self.all_stds, 0)

        ## Full dataset statistics
        self.pos_mean = self.all_rays[..., :ray_dim].mean(0)
        self.pos_std = self.all_rays[..., :ray_dim].std(0)

        self.dir_mean = self.all_rays[..., ray_dim:].mean(0)
        self.dir_std = self.all_rays[..., ray_dim:].std(0)

    def prepare_data(self):
        self.all_rays = self.all_rays.view(
            self.num_images, self.img_wh[1], self.img_wh[0], -1
        )
        self.all_rgb = self.all_rgb.view(
            self.num_images, self.img_wh[1], self.img_wh[0], -1
        )
        self.all_rgb_fft = fft_rgb(self.all_rgb)
        self.rgb_fft_mean = self.all_rgb_fft.mean(0)

    def shuffle(self):
        idx = list(np.random.choice(
            np.arange(0, self.num_images),
            size=self.num_images,
            replace=False
        ))

        self.all_rays = self.all_rays[idx]
        self.all_rgb = self.all_rgb[idx]
        #self.all_rgb_fft = torch.abs(self.all_rgb_fft[idx])
        self.all_rgb_fft = self.all_rgb_fft[idx]

    def __len__(self):
        return len(self.all_rays)

    def __getitem__(self, idx):
        return {
            'rays': self.all_rays[idx],
            'rgb': self.all_rgb[idx],
            'mean_fft': self.rgb_fft_mean,
        }

    def get_random_rays(self, ray_range):
        pos_rand = (torch.rand(
            (1, 1, 3,)
        ) * 2 - 1) * ray_range.pos
        pos_rand[..., 2] = 0

        dir_rand = (torch.rand(
            (self.height, self.width, 3,)
        ) * 2 - 1) * ray_range.dir
        dir_rand[..., 2] = -1
        dir_rand = torch.nn.functional.normalize(dir_rand, p=2.0, dim=-1)

        pos_rand = pos_rand.repeat(self.height, self.width, 1)

        return torch.cat([pos_rand, dir_rand], -1)


class FourierLightfieldDataset(Dataset):
    def __init__(
        self,
        cfg,
        train_dataset=None,
        **kwargs
    ):
        super().__init__()

        self.cfg = cfg
        self.img_wh = train_dataset.img_wh
        self.width, self.height = self.img_wh[0], self.img_wh[1]
        self.aspect = train_dataset.aspect
        self.num_images = train_dataset.num_images
        self.batch_size = cfg.batch_size

        self.all_rays = torch.clone(train_dataset.all_rays)
        self.all_rgb = torch.clone(train_dataset.all_rgb)

        # Prepare
        self.compute_stats()
        self.prepare_data()
        self.shuffle()

    def compute_stats(self):
        all_rays = self.all_rays.view(
            self.num_images, self.img_wh[1] * self.img_wh[0], -1
        )
        ray_dim = all_rays.shape[-1] // 2

        ## Per view statistics
        self.all_means = []
        self.all_stds = []

        for idx in range(self.num_images):
            cur_rays = all_rays[idx]
            self.all_means += [cur_rays.mean(0)]
            self.all_stds += [cur_rays.std(0)]

        self.all_means = torch.stack(self.all_means, 0)
        self.all_stds = torch.stack(self.all_stds, 0)

        ## Full dataset statistics
        self.pos_mean = self.all_rays[..., :ray_dim].mean(0)
        self.pos_std = self.all_rays[..., :ray_dim].std(0)

        self.dir_mean = self.all_rays[..., ray_dim:].mean(0)
        self.dir_std = self.all_rays[..., ray_dim:].std(0)

    def prepare_data(self):
        self.all_rays = self.all_rays.view(
            self.num_images, self.img_wh[1], self.img_wh[0], -1
        )
        self.all_rgb = self.all_rgb.view(
            self.num_images, self.img_wh[1], self.img_wh[0], -1
        )
        self.all_rgb_fft = fft_rgb(self.all_rgb)
        self.rgb_fft_mean = self.all_rgb_fft.mean(0)

    def shuffle(self):
        idx = list(np.random.choice(
            np.arange(0, self.num_images),
            size=self.num_images,
            replace=False
        ))

        self.all_rays = self.all_rays[idx]
        self.all_rgb = self.all_rgb[idx]
        #self.all_rgb_fft = torch.abs(self.all_rgb_fft[idx])
        self.all_rgb_fft = self.all_rgb_fft[idx]

    def __len__(self):
        return len(self.all_rays)

    def __getitem__(self, idx):
        return {
            'rays': self.all_rays[idx],
            'rgb': self.all_rgb[idx],
            'mean_fft': self.rgb_fft_mean,
        }

    def get_random_rays(self, ray_range):
        pos_rand = (torch.rand(
            (2,)
        ) * 2 - 1) * ray_range.pos

        return get_lightfield_rays(
            self.width, self.height,
            pos_rand[0], pos_rand[1],
            self.aspect
        )
