#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from utils.config_utils import replace_config, lambda_config

from utils.intersect_utils import (
    intersect_axis_plane,
    intersect_sphere,
)

from .embedding import embedding_dict


class Subdivision(nn.Module):
    def __init__(
        self,
        system,
        cfg
        ):
        super().__init__()

        self.cfg = cfg
        self.no_reparam = False

        ## (Hack) Prevent from storing system variables
        self.systems = [system]

        self.update_every = cfg.update_every if 'update_every' in cfg \
            else float('inf')

        if self.update_every == 'inf':
            self.update_every = float('inf')

    def get_system(self):
        return self.systems[0]

    def get_dataset(self):
        return self.systems[0].trainer.datamodule.train_dataset

    def process_intersect(self, rays, pts, idx):
        pass

    def intersect(self, rays):
        pass

    def forward(self, rays):
        with torch.no_grad():
            isect_pts, isect_depth, isect_idx = self.intersect(rays)

        isect_rays, isect_centers = self.process_intersect(
            rays, isect_pts, isect_idx
        )

        return torch.cat(
            [isect_rays, isect_centers], -1
        ), isect_depth, isect_idx, isect_idx.eq(-1)

    def validation(self, rays, results):
        system = self.get_system()
        W = system.cur_wh[0]
        H = system.cur_wh[1]

        depth = results['depth'].view(H, W, 1).cpu().numpy()
        depth = depth.transpose(2, 0, 1)

        disp = 1 / depth
        disp[depth == 0] = 0

        disp = (disp - disp.min()) / (disp.max() - disp.min())
        depth = (depth - depth.min()) / (depth.max() - depth.min())

        accum = results['accum'].view(H, W, 1).cpu().numpy()
        accum = accum.transpose(2, 0, 1)

        return {
            'depth': depth,
            'disp': disp,
            'accum': accum,
        }

    def validation_video(self, rays, results):
        outputs = self.validation(rays, results)

        return {
            'videos/subdivision_depth': outputs['depth'],
            'videos/subdivision_disp': outputs['disp'],
            'videos/subdivision_accum': outputs['accum'],
        }

    def validation_image(self, batch, batch_idx, results):
        outputs = self.validation(batch['coords'], results)

        return {
            'images/subdivision_depth': outputs['depth'],
            'images/subdivision_disp': outputs['disp'],
            'images/subdivision_accum': outputs['accum'],
        }

    def update(self):
        pass


class DepthSubdivision(Subdivision):
    def __init__(
        self,
        system,
        cfg
        ):
        super().__init__(system, cfg)

        self.near = cfg.near
        self.far = cfg.far

        self.grid_depth = cfg.grid_depth + 1
        self.reparam = cfg.reparam if 'reparam' in cfg else True
        self.voxel_size = torch.tensor((self.far - self.near) / (self.grid_depth - 1))

    def process_intersect(self, rays, pts, idx):
        depths = torch.linspace(
            self.near, self.far, self.grid_depth, device=rays.device
        ).float()

        # Reparametrize rays
        rays = rays[..., :6].unsqueeze(1)
        rays = rays * rays.new_ones(1, self.grid_depth, 1)

        if self.reparam:
            pts[..., 2] = pts[..., 2] - depths[None]

        rays = torch.cat(
            [
                pts,
                rays[..., 3:6]
            ],
            -1
        )

        centers = torch.ones_like(rays[..., 0:1]) * depths[None, ..., None]

        return rays, centers

    def intersect(self, rays):
        rays = rays[..., :6].unsqueeze(1)
        rays = rays * rays.new_ones(1, self.grid_depth, 1)

        depths = torch.linspace(
            self.near, self.far, self.grid_depth, device=rays.device
        ).float()
        depths = depths.view(1, self.grid_depth) * depths.new_ones(rays.shape[0], 1)

        isect_pts, isect_depth = intersect_axis_plane(
            rays, depths, -1
        )
        isect_depth = isect_depth[..., -1]
        isect_idx = (isect_depth >= 0).long()
        isect_idx[isect_depth < 0] = -1
        isect_idx[..., -1] = -1

        return isect_pts, isect_depth, isect_idx


class DepthEmbeddingSubdivision(Subdivision):
    def __init__(
        self,
        system,
        cfg,
        **kwargs
        ):

        super().__init__(system, cfg)

        self.group = cfg.group if 'group' in cfg else (kwargs['group'] if 'group' in kwargs else 'embedding')

        def set_z_channels(cfg, key):
            cfg[key] = cfg[key] * self.grid_depth

        # Net
        self.embedding_cfg = cfg.embedding
        lambda_config(self.embedding_cfg, 'z_channels', set_z_channels)
        self.net = embedding_dict[self.embedding_cfg.type](
            6,
            self.embedding_cfg,
            latent_dim=0,
            net_in_channels=None,
            group=self.group
        )

        # Subdivision
        self.near = cfg.near
        self.far = cfg.far
        self.grid_depth = cfg.grid_depth
        self.voxel_size = torch.tensor((self.far - self.near) / self.grid_depth)

        # Correct bounds
        self.near += self.voxel_size / 2
        self.far -= self.voxel_size / 2

    def forward(self, rays):
        # Embedded points and indices
        embed_rays = self.net(rays)
        embed_rays = embed_rays.view(
            rays.shape[0],
            self.grid_depth + 1,
            -1
        )

        # Points, primitive indices, primitive codes
        pts = embed_rays[..., :3]
        idx = torch.ones_like(pts[..., -1]).long()
        codes = torch.linspace(
            -1, 1, self.grid_depth + 1, device=rays.device
        ).float()[None] * torch.ones_like(pts[..., -1])

        # Depth for sorting
        depth = torch.norm(pts - rays[..., None, :3], dim=-1)
        sort_idx = torch.argsort(depth, dim=-1)
        sort_idx_pts = torch.stack([sort_idx, sort_idx, sort_idx], dim=1)

        # Sort all tensors
        depth = torch.gather(depth, -1, sort_idx)
        idx = torch.gather(idx, -1, sort_idx)
        codes = torch.gather(codes, -1, sort_idx)

        pts = pts.permute(0, 2, 1)
        pts = torch.gather(pts, -1, sort_idx_pts)
        pts = pts.permute(0, 2, 1)

        # Get embed rays
        pts = torch.cat([pts, pts[..., :1, :, :]], 1)
        embed_rays = torch.cat([pts[..., :-1, :], pts[..., 1:, :]], -1)

        return torch.cat(
            [embed_rays, codes], -1
        ), depth, idx, idx.eq(-1)


class VoxelEmbeddingSubdivision(Subdivision):
    def __init__(
        self,
        system,
        cfg,
        **kwargs
        ):
        super().__init__(system, cfg)

        self.group = cfg.group if 'group' in cfg else (kwargs['group'] if 'group' in kwargs else 'embedding')

        self.min_point = torch.tensor(cfg.min_point).float().cuda()
        self.max_point = torch.tensor(cfg.max_point).float().cuda()

        # Subdivision in depth dimension
        self.continuous_code = cfg.continuous_code if 'continuous_code' in cfg else False
        self.no_reparam = cfg.no_reparam if 'no_reparam' in cfg else True
        self.no_voxel = cfg.no_voxel if 'no_voxel' in cfg else True
        self.no_voxel_reparam = cfg.no_voxel_reparam if 'no_voxel_reparam' in cfg else self.no_voxel
        self.grid_depth = cfg.grid_depth
        self.depth_step = torch.tensor((self.max_point[-1] - self.min_point[-1]) / self.grid_depth).cuda()

        # Lateral subdivision
        self.grid_width = cfg.grid_width
        self.lat_step = torch.tensor((self.max_point[0] - self.min_point[0]) / self.grid_width)
        self.voxel_size = self.depth_step
        self.voxel_size_ = torch.stack([self.lat_step, self.lat_step, self.depth_step]).cuda()

        # Correct bounds
        self.offset = torch.tensor([0.0, 0.0, 0.5]).float().cuda()
        self.min_point += self.voxel_size_ * self.offset
        self.max_point -= self.voxel_size_ * self.offset
        self.near, self.far = self.min_point[-1], self.max_point[-1]

        # Embedding
        def set_z_channels(cfg, key):
            cfg[key] = cfg[key] * self.grid_depth

        self.embedding_cfg = cfg.embedding
        lambda_config(self.embedding_cfg, 'z_channels', set_z_channels)
        self.net = embedding_dict[self.embedding_cfg.type](
            6,
            self.embedding_cfg,
            latent_dim=0,
            net_in_channels=None,
            group=self.group
        )

        # Post embedding
        self.post_embedding_cfg = cfg.post_embedding if 'post_embedding' in cfg else None

        if self.post_embedding_cfg is not None:
            lambda_config(self.post_embedding_cfg, 'z_channels', set_z_channels)
            self.post_in_channels = self.net.out_channels * 2

            self.post_net = embedding_dict[self.post_embedding_cfg.type](
                self.post_in_channels,
                self.post_embedding_cfg,
                latent_dim=0,
                net_in_channels=None,
                group=self.group
            )
        else:
            self.post_net = None

    def forward(self, rays):
        # Intersect layers
        plane_depths = torch.linspace(
            self.near, self.far, self.grid_depth, device=rays.device
        ).float()
        plane_depths = torch.cat(
            [
                plane_depths,
                10000 * torch.ones_like(plane_depths[..., -1:])
            ],
            -1
        )
        plane_depths = plane_depths.view(1, self.grid_depth + 1) \
            * plane_depths.new_ones(rays.shape[0], 1)

        isect_pts, isect_depth = intersect_axis_plane(
            rays[..., None, :], plane_depths, -1
        )
        isect_depth = isect_depth[..., -1]
        isect_idx = (isect_depth >= 0).long()
        isect_idx[isect_depth < 0] = -1
        isect_idx[..., -1] = -1

        # Voxel codes
        voxel_centers = (torch.round(
            (isect_pts - self.min_point[None, None]) / self.voxel_size_[None, None]
        ) + self.offset[None, None]) * self.voxel_size_[None, None] + self.min_point[None, None]

        if self.no_voxel_reparam:
            voxel_centers[..., :2] = 0

        # Embedded points
        embed_rays = self.net(rays)
        embed_pts = embed_rays.view(
            rays.shape[0],
            self.grid_depth,
            -1,
            3,
        )

        # Local light field parameterization
        pts = embed_pts[..., :3]
        depths = torch.norm(
            pts - rays[..., None, None, :3],
            dim=-1
        ).mean(-1)

        if self.no_reparam:
            embed_rays = torch.cat(
                [
                    pts,
                    rays[..., None, None, 3:6] * torch.ones_like(pts)
                ],
                dim=-1
            )
        else:
            embed_rays = torch.cat(
                [
                    pts - voxel_centers[..., :-1, None, :],
                    rays[..., None, None, 3:6] * torch.ones_like(pts)
                ],
                dim=-1
            )

        # Post embedding
        if self.post_net is not None:
            embed_rays = embed_rays.view(rays.shape[0], -1)
            embed_rays = self.post_net(embed_rays)

        # Reshape and concat
        embed_rays = embed_rays.view(
            rays.shape[0],
            self.grid_depth,
            -1,
        )
        embed_rays = torch.cat([embed_rays, embed_rays[..., :1, :]], 1)
        depths = torch.cat([depths, depths[..., :1]], 1)

        # Center subdivisions at predicted point, use full point as "latent code"
        if self.continuous_code:
            voxel_centers[..., -1:] = embed_rays[..., 2:3].clone()

        if self.no_voxel:
            voxel_centers = voxel_centers[..., -1:]

        embed_rays[..., 2] = 0
        embed_rays = torch.cat([embed_rays, voxel_centers], -1)
        return embed_rays, depths, isect_idx, isect_idx.eq(-1)


class NeRFSubdivision(Subdivision):
    def __init__(
        self,
        system,
        cfg
        ):
        super().__init__(system, cfg)

        self.min_point = torch.tensor(cfg.min_point).float().cuda()
        self.max_point = torch.tensor(cfg.max_point).float().cuda()

        # Subdivision in depth dimension
        self.no_voxel = cfg.no_voxel if 'no_voxel' in cfg else True
        self.grid_depth = cfg.grid_depth
        self.steps = cfg.steps
        self.depth_step = torch.tensor((self.max_point[-1] - self.min_point[-1]) / self.grid_depth).cuda()

        # Lateral subdivision
        self.grid_width = cfg.grid_width
        self.lat_step = torch.tensor((self.max_point[0] - self.min_point[0]) / self.grid_width)
        self.voxel_size = self.depth_step
        self.voxel_size_ = torch.stack([self.lat_step, self.lat_step, self.depth_step]).cuda()

        # Correct bounds
        self.offset = torch.tensor([0.0, 0.0, 0.5]).float().cuda()
        self.min_point += self.voxel_size_ * self.offset
        self.max_point -= self.voxel_size_ * self.offset
        self.near, self.far = self.min_point[-1], self.max_point[-1]

        # Embedding

    def forward(self, rays):
        # Intersect layers
        plane_depths = torch.linspace(
            self.near, self.far, self.steps, device=rays.device
        ).float()
        plane_depths = torch.cat(
            [
                plane_depths,
                10000 * torch.ones_like(plane_depths[..., -1:])
            ],
            -1
        )
        plane_depths = plane_depths.view(1, self.steps + 1) \
            * plane_depths.new_ones(rays.shape[0], 1)

        isect_pts, isect_depth = intersect_axis_plane(
            rays[..., None, :], plane_depths, -1
        )
        isect_depth = isect_depth[..., -1]
        isect_idx = (isect_depth >= 0).long()
        isect_idx[isect_depth < 0] = -1
        isect_idx[..., -1] = -1

        # Voxel codes
        voxel_centers = (torch.round(
            (isect_pts - self.min_point[None, None]) / self.voxel_size_[None, None]
        ) + self.offset[None, None]) * self.voxel_size_[None, None] + self.min_point[None, None]

        # Embedded points
        embed_pts = isect_pts[..., :-1, :]

        # Local light field parameterization
        depths = torch.norm(
            embed_pts - rays[..., None, :3],
            dim=-1
        )

        # Use full point as "latent code"
        if self.no_voxel:
            voxel_centers[..., :2] = 0.0

        embed_rays = torch.cat(
            [
                embed_pts - voxel_centers[..., :-1, :],
                rays[..., None, 3:6] * torch.ones_like(embed_pts)
            ],
            dim=-1
        )

        # Reshape and concat
        embed_rays = embed_rays.view(
            rays.shape[0],
            self.steps,
            -1,
        )
        embed_rays = torch.cat([embed_rays, embed_rays[..., :1, :]], 1)
        depths = torch.cat([depths, depths[..., :1]], 1)

        embed_rays = torch.cat([embed_rays, isect_pts], -1)
        return embed_rays, depths, isect_idx, isect_idx.eq(-1)


class RadialSubdivision(Subdivision):
    def __init__(
        self,
        system,
        cfg
        ):
        super().__init__(system, cfg)

        self.near = cfg.near
        self.far = cfg.far

        if cfg.voxel_size is not None:
            self.voxel_size = torch.tensor(cfg.voxel_size)
            self.num_slices = int(np.round((self.far - self.near) / self.voxel_size)) + 1
            self.grid_depth = self.num_slices
        else:
            self.grid_depth = cfg.grid_depth
            self.num_slices = cfg.grid_depth
            self.voxel_size = torch.tensor((self.far - self.near) / (self.num_slices - 1))

        # Depths
        self.radii = torch.linspace(
            self.near, self.far, self.num_slices
        )

    def process_intersect(self, rays, pts, idx):
        mask = idx.eq(-1)
        idx[mask] = 0

        radii = self.radii.view(1, self.num_slices).to(rays).repeat(rays.shape[0], 1)
        radii = torch.gather(radii, -1, idx)

        idx[mask] = -1

        isect_pts = rays[..., None, 0:3] / radii
        pts = torch.where(
            mask.unsqueeze(-1) * mask.new_ones(1, 1, 3),
            pts,
            isect_pts
        )
        rays = torch.cat([pts, rays[..., 3:6]], -1)

        return rays, radii

    def intersect(self, rays):
        # Reparametrize rays
        rays = rays[..., :6].unsqueeze(1).repeat(1, self.num_slices, 1)

        radii = self.radii.view(1, self.num_slices).to(rays.device).repeat(rays.shape[0], 1)
        isect_pts = intersect_sphere(rays, radii)
        isect_depth = torch.norm(
            rays[..., :3] - isect_pts, dim=-1
        )

        # Sort
        sort_idx = torch.argsort(isect_depth, dim=-1)
        sort_idx_pts = torch.stack(
            [sort_idx, sort_idx, sort_idx], dim=1
        )

        isect_depth = torch.gather(isect_depth, -1, sort_idx)
        isect_pts = isect_pts.permute(0, 2, 1)
        isect_pts = torch.gather(isect_pts, -1, sort_idx_pts)
        isect_pts = isect_pts.permute(0, 2, 1)

        isect_idx = torch.ones_like(isect_depth).long()
        isect_idx[isect_depth < 0] = -1

        return isect_pts, isect_depth, isect_idx


def voxels_from_bb(min_point, max_point, voxel_size):
    steps = ((max_point - min_point) / voxel_size).round().astype('int64') + 1
    x, y, z = [
        c.reshape(-1).astype('float32') for c in np.meshgrid(
            np.arange(steps[0]),
            np.arange(steps[1]),
            np.arange(steps[2])
        )
    ]
    x = x * voxel_size + min_point[0]
    y = y * voxel_size + min_point[1]
    z = z * voxel_size + min_point[2]

    return np.stack([x, y, z]).T.astype('float32')



subdivision_dict = {
    'depth': DepthSubdivision,
    'depth_embed': DepthEmbeddingSubdivision,
    'voxel_embed': VoxelEmbeddingSubdivision,
    'nerf': NeRFSubdivision,
    'radial': RadialSubdivision,
}
