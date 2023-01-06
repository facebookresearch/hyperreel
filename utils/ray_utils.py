#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn.functional as F
import numpy as np

from kornia import create_meshgrid


def get_lightfield_rays(
    U, V, s, t, aspect, st_scale=1.0, uv_scale=1.0, near=-1, far=0,
    use_inf=False, center_u=0.0, center_v=0.0,
    ):
    u = torch.linspace(-1, 1, U, dtype=torch.float32)
    v = torch.linspace(1, -1, V, dtype=torch.float32) / aspect

    vu = list(torch.meshgrid([v, u]))
    u = vu[1] * uv_scale
    v = vu[0] * uv_scale
    s = torch.ones_like(vu[1]) * s * st_scale
    t = torch.ones_like(vu[0]) * t * st_scale

    rays = torch.stack(
        [
            s,
            t,
            near * torch.ones_like(s),
            u - s,
            v - t,
            (far - near) * torch.ones_like(s),
        ],
        axis=-1
    ).view(-1, 6)

    return torch.cat(
        [
            rays[..., 0:3],
            torch.nn.functional.normalize(rays[..., 3:6], p=2.0, dim=-1)
        ],
        -1
    )

def get_epi_rays(
    U, v, S, t, aspect, st_scale=1.0, uv_scale=1.0, near=-1, far=0,
    use_inf=False, center_u=0.0, center_v=0.0,
    ):
    u = torch.linspace(-1, 1, U, dtype=torch.float32)
    s = torch.linspace(-1, 1, S, dtype=torch.float32) / aspect

    su = list(torch.meshgrid([s, u]))
    u = su[1] * uv_scale
    v = torch.ones_like(su[0]) * v * uv_scale
    s = su[0] * st_scale
    t = torch.ones_like(su[0]) * t * st_scale

    rays = torch.stack(
        [
            s,
            t,
            near * torch.ones_like(s),
            u - s,
            v - t,
            (far - near) * torch.ones_like(s),
        ],
        axis=-1
    ).view(-1, 6)

    return torch.cat(
        [
            rays[..., 0:3],
            torch.nn.functional.normalize(rays[..., 3:6], p=2.0, dim=-1)
        ],
        -1
    )

def get_pixels_for_image(
    H, W, device='cpu'
):
    grid = create_meshgrid(H, W, normalized_coordinates=False, device=device)[0]

    return grid

def get_random_pixels(
    n_pixels, H, W, device='cpu'
):
    grid = torch.rand(n_pixels, 2, device=device)

    i, j = grid.unbind(-1)
    grid[..., 0] = grid[..., 0] * (W - 1)
    grid[..., 1] = grid[..., 1] * (H - 1)

    return grid

def get_ray_directions_from_pixels_K(
    grid, K, centered_pixels=False, flipped=False
):
    i, j = grid.unbind(-1)

    offset_x = 0.5 if centered_pixels else 0.0
    offset_y = 0.5 if centered_pixels else 0.0

    directions = torch.stack(
        [
            (i - K[0, 2] + offset_x) / K[0, 0],
            (-(j - K[1, 2] + offset_y) / K[1, 1]) if not flipped else (j - K[1, 2] + offset_y) / K[1, 1],
            -torch.ones_like(i)
        ],
        -1
    )

    return directions

def get_ray_directions_K(H, W, K, centered_pixels=False, flipped=False, device='cpu'):
    grid = create_meshgrid(H, W, normalized_coordinates=False, device=device)[0]
    return get_ray_directions_from_pixels_K(grid, K, centered_pixels, flipped=flipped)

def get_rays(directions, c2w, normalize=True):
    # Implementation: https://github.com/kwea123/nerf_pl

    # Rotate ray directions from camera coordinate to the world coordinate
    rays_d = directions @ c2w[:, :3].T # (H, W, 3)
    if normalize:
        rays_d = torch.nn.functional.normalize(rays_d, p=2.0, dim=-1)

    # The origin of all rays is the camera origin in world coordinate
    rays_o = c2w[:, 3].expand(rays_d.shape) # (H, W, 3)

    rays_d = rays_d.view(-1, 3)
    rays_o = rays_o.view(-1, 3)

    return rays_o, rays_d

def get_ndc_rays_fx_fy(H, W, fx, fy, near, rays):
    rays_o, rays_d = rays[..., 0:3], rays[..., 3:6]

    # Shift ray origins to near plane
    t = -(near + rays_o[...,2]) / rays_d[...,2]
    rays_o = rays_o + t[...,None] * rays_d

    # o_z = -near
    # (o_z / (1 - t') - o_z) / d_z

    # Store some intermediate homogeneous results
    ox_oz = rays_o[...,0] / rays_o[...,2]
    oy_oz = rays_o[...,1] / rays_o[...,2]

    # Projection
    o0 = -1./(W/(2.*fx)) * ox_oz
    o1 = -1./(H/(2.*fy)) * oy_oz
    o2 = 1. + 2. * near / rays_o[...,2]

    d0 = -1./(W/(2.*fx)) * (rays_d[...,0]/rays_d[...,2] - ox_oz)
    d1 = -1./(H/(2.*fy)) * (rays_d[...,1]/rays_d[...,2] - oy_oz)
    d2 = 1 - o2

    rays_o = torch.stack([o0, o1, o2], -1) # (B, 3)
    rays_d = torch.stack([d0, d1, d2], -1) # (B, 3)
    #rays_d = torch.nn.functional.normalize(rays_d, p=2.0, dim=-1)

    return torch.cat([rays_o, rays_d], -1)

def sample_images_at_xy(
    images,
    xy_grid,
    H, W,
    mode="bilinear",
    padding_mode="border"
):
    batch_size = images.shape[0]
    spatial_size = images.shape[1:-1]

    xy_grid = torch.clone(xy_grid.reshape(batch_size, -1, 1, 2))
    xy_grid[..., 0] = (xy_grid[..., 0] / (W - 1)) * 2 - 1
    xy_grid[..., 1] = (xy_grid[..., 1] / (H - 1)) * 2 - 1

    images_sampled = torch.nn.functional.grid_sample(
        images.permute(0, 3, 1, 2),
        xy_grid,
        align_corners=False,
        mode=mode,
        padding_mode=padding_mode,
    )

    return images_sampled.permute(0, 2, 3, 1).view(-1, images.shape[-1])

def dot(a, b, axis=-1, keepdim=False):
    return torch.sum(a * b, dim=axis, keepdim=keepdim)

def reflect(dirs, normal):
    dir_dot_normal = dot(-dirs, normal, keepdim=True) * normal
    return 2 * dir_dot_normal + dirs

def get_stats(rays):
    return (rays.mean(0), rays.std(0))

def get_weight_map(
    rays,
    jitter_rays,
    cfg,
    weights=None,
    softmax=True
):
    ray_dim = rays.shape[-1] // 2

    # Angles
    angles = torch.acos(
        torch.clip(
            dot(rays[..., ray_dim:], jitter_rays[..., ray_dim:]),
            -1 + 1e-8, 1 - 1e-8
        )
    ).detach()

    # Distances
    dists = torch.linalg.norm(
        rays[..., :ray_dim] - jitter_rays[..., :ray_dim],
        dim=-1
    ).detach()

    # Weights
    if weights is None:
        weights = torch.zeros_like(angles)

    if softmax:
        weights = torch.nn.functional.softmax(
            0.5 * -(torch.square(angles / cfg.angle_std) + torch.square(dists / cfg.dist_std)) + weights, dim=0
        )[..., None]
    else:
        #print("Angle:", angles.max(), angles.mean(), cfg.angle_std)
        #print("Dist:", dists.max(), dists.mean(), cfg.dist_std)

        weights = torch.exp(
            0.5 * -(torch.square(angles / cfg.angle_std) + torch.square(dists / cfg.dist_std)) + weights
        )[..., None]

    # Normalization constant
    constant = np.power(2 * np.pi * cfg.angle_std * cfg.angle_std, -1.0 / 2.0) \
        * np.power(2 * np.pi * cfg.dist_std * cfg.dist_std, -1.0 / 2.0)

    return weights / constant

def compute_sigma_angle(
    query_ray,
    rays,
    angle_std=-1
):
    # Angles
    angles = torch.acos(
        torch.clip(
            dot(rays, query_ray),
            -1 + 1e-8, 1 - 1e-8
        )
    )

    # Calculate angle std
    if angle_std < 0:
        mean_ray = torch.nn.functional.normalize(rays.mean(1).unsqueeze(1), dim=-1)
        mean_angles = torch.acos(
            torch.clip(
                dot(mean_ray, query_ray),
                -1 + 1e-8, 1 - 1e-8
            )
        )

        angle_std, _ = torch.median(torch.abs(mean_angles), dim=1, keepdim=True)
        print(angle_std[0])
        c = torch.pow(2 * np.pi * angle_std * angle_std, -1.0 / 2.0)
    else:
        c = np.power(2 * np.pi * angle_std * angle_std, -1.0 / 2.0)

    # Weights
    weights = torch.exp(
        0.5 * -(torch.square(angles / angle_std))
    )[..., None]
    weights = c * weights.mean(1)

    return weights * c

def compute_sigma_dot(
    query_ray,
    rays,
    dot_std=-1
):
    # Dots
    dots = torch.clip(
        dot(rays, query_ray),
        -1 + 1e-8,
        1 - 1e-8
    )

    # Calculate dot std
    if dot_std < 0:
        mean_ray = torch.nn.functional.normalize(rays.mean(1).unsqueeze(1), dim=-1)
        mean_dots = torch.clip(
            dot(mean_ray, query_ray),
            -1 + 1e-8, 1 - 1e-8
        )

        dot_std, _ = torch.median(torch.abs(1 - mean_dots), dim=1, keepdim=True)
        print(dot_std[0])

        c = torch.pow(2 * np.pi * dot_std * dot_std, -1.0 / 2.0)
    else:
        c = np.power(2 * np.pi * dot_std * dot_std, -1.0 / 2.0)

    # Weights
    weights = torch.exp(
        0.5 * -(torch.square((1 - dots) / dot_std))
    )[..., None]
    weights = c * weights.mean(1)

    return weights * c


def weighted_stats(rgb, weights):
    weights_sum = weights.sum(0)
    rgb_mean = ((rgb * weights).sum(0) / weights_sum)
    rgb_mean = torch.where(
        weights_sum == 0,
        torch.zeros_like(rgb_mean),
        rgb_mean
    )

    diff = rgb - rgb_mean.unsqueeze(0)
    rgb_var = (diff * diff * weights).sum(0) / weights_sum
    rgb_var = torch.where(
        weights_sum == 0,
        torch.zeros_like(rgb_var),
        rgb_var
    )

    return rgb_mean, rgb_var

def jitter_ray_origins(rays, jitter):
    ray_dim = 3

    pos_rand = torch.randn(
        (rays.shape[0], jitter.bundle_size, ray_dim), device=rays.device
    ) * jitter.pos

    rays = rays.view(rays.shape[0], -1, rays.shape[-1])

    if rays.shape[1] == 1:
        rays = rays.repeat(1, jitter.bundle_size, 1)

    rays_o = rays[..., :ray_dim] + pos_rand.type_as(rays)

    return torch.cat([rays_o, rays[..., ray_dim:]], -1)

def jitter_ray_directions(rays, jitter):
    ray_dim = 3

    dir_rand = torch.randn(
        (rays.shape[0], jitter.bundle_size, ray_dim), device=rays.device
    ) * jitter.dir

    rays = rays.view(rays.shape[0], -1, rays.shape[-1])

    if rays.shape[1] == 1:
        rays = rays.repeat(1, jitter.bundle_size, 1)

    rays_d = rays[..., ray_dim:2*ray_dim] + dir_rand.type_as(rays)
    rays_d = F.normalize(rays_d, dim=-1)

    return torch.cat([rays[..., :ray_dim], rays_d], -1)


def from_ndc(t_p, rays, near):
    t = (near / (1 - t_p) - near) / rays[..., 5, None]
    t = t + (near - rays[..., None, 2]) / rays[..., None, 5]
    return t


def get_ray_density(sigma, ease_iters, cur_iter):
    if cur_iter >= ease_iters:
        return sigma
    else:
        w = min(max(float(ease_iters) / cur_iter, 0.0), 1.0)
        return sigma * w + (1 - w)
