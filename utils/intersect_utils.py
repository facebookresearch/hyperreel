#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn.functional as F
import numpy as np


def sort_z(z_vals, dim: int, descending: bool):
    sort_idx = torch.argsort(z_vals, dim=dim, descending=descending)
    z_vals = torch.gather(z_vals, -1, sort_idx)

    return z_vals, sort_idx

def sort_with(sort_idx, points):
    points = points.permute(0, 2, 1)
    sort_idx = sort_idx.unsqueeze(1).repeat(1, points.shape[1], 1)
    points = torch.gather(points, -1, sort_idx)
    return points.permute(0, 2, 1)

def dot(a, b, axis=-1):
    return torch.sum(a * b, dim=axis)

def min_sphere_radius(rays, origin):
    rays_o, rays_d = rays[..., :3] - origin, rays[..., 3:6]
    rays_d = torch.nn.functional.normalize(rays_d, p=2.0, dim=-1)

    m = torch.cross(rays_o, rays_d, dim=-1)
    rays_o = torch.cross(rays_d, m, dim=-1)
    return torch.linalg.norm(rays_o, dim=-1)

def min_cylinder_radius(rays, origin):
    rays_o, rays_d = rays[..., 0:3] - origin, rays[..., 3:6]
    rays_o = torch.cat([rays_o[..., 0:1], torch.zeros_like(rays[..., 1:2]), rays_o[..., 2:3]], -1)
    rays_d = torch.cat([rays_d[..., 0:1], torch.zeros_like(rays[..., 1:2]), rays_d[..., 2:3]], -1)
    rays_d = torch.nn.functional.normalize(rays_d, p=2.0, dim=-1)

    m = torch.cross(rays_o, rays_d, dim=-1)
    rays_o = torch.cross(rays_d, m, dim=-1)
    return torch.linalg.norm(rays_o, dim=-1)

def intersect_sphere(rays, origin, radius, continuous=False):
    rays_o, rays_d = rays[..., 0:3] - origin, rays[..., 3:6]
    o = rays_o
    d = rays_d

    dot_o_o = dot(o, o)
    dot_d_d = dot(d, d)
    dot_o_d = dot(o, d)

    a = dot_d_d
    b = 2 * dot_o_d
    c = dot_o_o - radius * radius
    disc = b * b - 4 * a * c

    if continuous:
        disc = torch.abs(disc)
    else:
        disc = torch.where(disc < 0, torch.zeros_like(disc), disc)

    t1 = (-b + torch.sqrt(disc + 1e-8)) / (2 * a)
    t2 = (-b - torch.sqrt(disc + 1e-8)) / (2 * a)

    t1 = torch.where(
        disc <= 0,
        torch.zeros_like(t1),
        t1
    )
    t2 = torch.where(
        disc <= 0,
        torch.zeros_like(t2),
        t2
    )

    t = torch.where(
        (t2 < 0) | (radius < 0),
        t1,
        t2
    )

    return t

def intersect_cylinder(rays, origin, radius, continuous=False):
    rays_o, rays_d = rays[..., 0:3] - origin, rays[..., 3:6]
    o = torch.cat([rays_o[..., 0:1], rays_o[..., 2:3]], -1)
    d = torch.cat([rays_d[..., 0:1], rays_d[..., 2:3]], -1)

    dot_o_o = dot(o, o)
    dot_d_d = dot(d, d)
    dot_o_d = dot(o, d)

    a = dot_d_d
    b = 2 * dot_o_d
    c = dot_o_o - radius * radius
    disc = b * b - 4 * a * c

    if continuous:
        disc = torch.abs(disc)
    else:
        disc = torch.where(disc < 0, torch.zeros_like(disc), disc)

    t1 = (-b + torch.sqrt(disc + 1e-8)) / (2 * a)
    t2 = (-b - torch.sqrt(disc + 1e-8)) / (2 * a)

    t1 = torch.where(
        disc <= 0,
        torch.zeros_like(t1),
        t1
    )
    t2 = torch.where(
        disc <= 0,
        torch.zeros_like(t2),
        t2
    )

    t = torch.where(
        (t2 < 0) | (radius < 0), # TODO: Maybe change
        t1,
        t2
    )

    return t

def intersect_axis_plane(
    rays,
    val,
    dim,
    exclude=False,
):

    # Calculate intersection
    rays_o, rays_d = rays[..., :3], rays[..., 3:6]
    rays_d = torch.where(
        torch.abs(rays_d) < 1e-5,
        torch.ones_like(rays_d) * 1e12,
        rays_d
    )

    t = (val - rays_o[..., dim]) / rays_d[..., dim]
    #t = torch.where(
    #    t < 1e-5,
    #    torch.zeros_like(t),
    #    t
    #)

    # Return
    return t

def intersect_voxel_grid(
    rays,
    origin,
    val,
):

    rays_o, rays_d = rays[..., :3] - origin, rays[..., 3:6]

    # Mask out invalid
    rays_d = torch.where(
        torch.abs(rays_d) < 1e-5,
        torch.ones_like(rays_d) * 1e12,
        rays_d
    )

    # Calculate intersection
    t = (val - rays_o) / rays_d
    #t = torch.where(
    #    (t < 1e-5),
    #    torch.zeros_like(t),
    #    t
    #)

    # Reshape
    t = t.view(t.shape[0], -1)

    # Return
    return t

def intersect_max_axis_plane(
    rays,
    max_dir,
    origin,
    val,
):

    # Calculate intersection
    rays_o, rays_d = rays[..., :3] - origin, rays[..., 3:6]
    rays_d = torch.where(
        torch.abs(rays_d) < 1e-5,
        torch.ones_like(rays_d) * 1e12,
        rays_d
    )

    t = (val - rays_o) / rays_d
    t = torch.where(
        t < 1e-8,
        torch.zeros_like(t),
        t
    )
    t = torch.gather(t, 2, max_dir).reshape(t.shape[0], -1)

    # Reshape
    t = t.view(t.shape[0], -1)

    # Return
    return t

def intersect_plane(
    rays,
    normal,
    distance,
):
    # Calculate intersection
    rays_o, rays_d = rays[..., :3], rays[..., 3:6]
    o_dot_n = dot(rays_o, normal)
    d_dot_n = dot(rays_d, normal)
    d_dot_n = torch.where(
        torch.abs(d_dot_n) < 1e-5,
        torch.ones_like(d_dot_n) * 1e12,
        d_dot_n
    )

    t = (distance - o_dot_n) / (d_dot_n)
    #t = torch.where(
    #    t < 1e-5,
    #    torch.zeros_like(t),
    #    t
    #)

    # Reshape
    t = t.view(t.shape[0], -1)

    # Return
    return t
