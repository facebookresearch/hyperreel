#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import scipy.linalg


def normalize(v):
    return v / np.linalg.norm(v)

def average_poses(poses):
    # 1. Compute the center
    center = poses[..., 3].mean(0) # (3)

    # 2. Compute the z axis
    z = normalize(poses[..., 2].mean(0)) # (3)

    # 3. Compute axis y' (no need to normalize as it's not the final output)
    y_ = poses[..., 1].mean(0) # (3)

    # 4. Compute the x axis
    x = normalize(np.cross(y_, z)) # (3)

    # 5. Compute the y axis (as z and x are normalized, y is already of norm 1)
    y = np.cross(z, x) # (3)

    R = np.stack([x, y, z], 1)

    center = center[..., None]
    #center = (-R @ center[..., None])

    pose_avg = np.concatenate([R, center], 1) # (3, 4)

    return pose_avg

def viewmatrix(z, up, pos):
    vec2 = normalize(z)
    vec1_avg = up
    vec0 = normalize(np.cross(vec1_avg, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.stack([vec0, vec1, vec2, pos], 1)
    return m


def center_poses(poses):
    pose_avg = average_poses(poses) # (3, 4)
    pose_avg_homo = np.eye(4)
    pose_avg_homo[:3] = pose_avg
    last_row = np.tile(np.array([0, 0, 0, 1]), (len(poses), 1, 1)) # (N_images, 1, 4)
    poses_homo = \
        np.concatenate([poses, last_row], 1) # (N_images, 4, 4) homogeneous coordinate

    poses_centered = np.linalg.inv(pose_avg_homo) @ poses_homo # (N_images, 4, 4)
    poses_centered = poses_centered[:, :3] # (N_images, 3, 4)

    return poses_centered, np.linalg.inv(pose_avg_homo)


def center_poses_with(poses, train_poses, avg_pose=None):
    if avg_pose is None:
        pose_avg = average_poses(train_poses) # (3, 4)
        pose_avg_homo = np.eye(4)
        pose_avg_homo[:3] = pose_avg
        inv_pose = np.linalg.inv(pose_avg_homo)
    else:
        inv_pose = np.copy(avg_pose)

    last_row = np.tile(np.array([0, 0, 0, 1]), (len(poses), 1, 1)) # (N_images, 1, 4)
    poses_homo = \
        np.concatenate([poses, last_row], 1) # (N_images, 4, 4) homogeneous coordinate
    poses_centered = inv_pose @ poses_homo # (N_images, 4, 4)
    poses_centered = poses_centered[:, :3] # (N_images, 3, 4)

    return poses_centered, inv_pose


def center_poses_with_rotation_only(poses, train_poses):
    pose_avg = average_poses(train_poses) # (3, 4)
    pose_avg_homo = np.eye(4)
    pose_avg_homo[:3, :3] = pose_avg[:3, :3]
    last_row = np.tile(np.array([0, 0, 0, 1]), (len(poses), 1, 1)) # (N_images, 1, 4)
    poses_homo = \
        np.concatenate([poses, last_row], 1) # (N_images, 4, 4) homogeneous coordinate

    poses_centered = np.linalg.inv(pose_avg_homo) @ poses_homo # (N_images, 4, 4)
    poses_centered = poses_centered[:, :3] # (N_images, 3, 4)

    return poses_centered, np.linalg.inv(pose_avg_homo)


def center_poses_reference(poses):
    pose_avg = average_poses(poses) # (3, 4)
    pose_avg_homo = np.eye(4)

    pose_avg_homo[:3] = pose_avg

    last_row = np.tile(np.array([0, 0, 0, 1]), (len(poses), 1, 1)) # (N_images, 1, 4)
    poses_homo = \
        np.concatenate([poses, last_row], 1) # (N_images, 4, 4) homogeneous coordinate

    # Get reference
    dists = np.sum(np.square(pose_avg[:3, 3] - poses[:, :3, 3]), -1)
    reference_view_id = np.argmin(dists)
    pose_avg_homo = poses_homo[reference_view_id]

    poses_centered = np.linalg.inv(pose_avg_homo) @ poses_homo # (N_images, 4, 4)
    poses_centered = poses_centered[:, :3] # (N_images, 3, 4)

    return poses_centered, np.linalg.inv(pose_avg_homo)

def create_rotating_spiral_poses(camera_offset, poses, pose_rad, spiral_rads, focal, theta_range, N=240, rots=4, flip=False):
    # Camera offset and up
    camera_offset = np.array(camera_offset)
    up = normalize(poses[:, :3, 1].sum(0))

    # Radii in X, Y, Z
    render_poses = []
    spiral_rads = np.array(list(spiral_rads) + [1.0])

    # Pose, spiral angle
    pose_thetas = np.linspace(
        np.pi * theta_range[0],
        np.pi * theta_range[1],
        N,
        endpoint=False
    )

    spiral_thetas = np.linspace(
        0,
        2 * np.pi * rots,
        N,
        endpoint=False
    )

    # Create poses
    for pose_theta, spiral_theta in zip(pose_thetas, spiral_thetas):
        # Central cylindrical pose
        pose_x, pose_z = (
            np.sin(pose_theta) * pose_rad,
            -np.cos(pose_theta) * pose_rad,
        )
        pose_y = 0

        pose_center = np.array([pose_x, pose_y, pose_z]) + camera_offset
        pose_forward = np.array([-pose_x, -pose_y, -pose_z])
        c2w = viewmatrix(pose_forward, up, pose_center)

        # Spiral pose
        c = np.dot(c2w[:3,:4], np.array(
            [np.cos(spiral_theta), -np.sin(spiral_theta), -np.sin(spiral_theta * 0.5), 1.]
            ) * spiral_rads
        )

        z = normalize(c - np.dot(c2w[:3,:4], np.array([0, 0, -focal, 1.])))
        render_poses.append(viewmatrix(z, up, c))

    return render_poses

def create_spiral_poses(poses, rads, focal, N=120, flip=False):
    c2w = average_poses(poses)
    up = normalize(poses[:, :3, 1].sum(0))
    rots = 2

    render_poses = []
    rads = np.array(list(rads) + [1.])

    for theta in np.linspace(0., 2. * np.pi * rots, N+1)[:-1]:
        c = np.dot(c2w[:3,:4], np.array(
            [np.cos(theta), -np.sin(theta), -np.sin(theta*0.5), 1.]
            ) * rads
        )

        if flip:
            z = normalize(np.dot(c2w[:3,:4], np.array([0, 0, focal, 1.])) - c)
        else:
            z = normalize(c - np.dot(c2w[:3,:4], np.array([0, 0, -focal, 1.])))

        render_poses.append(viewmatrix(z, up, c))

    return render_poses

def create_spherical_poses(radius, n_poses=120):
    def spherical_pose(theta, phi, radius):
        def trans_t(t):
            return np.array(
                [
                    [1,0,0,0],
                    [0,1,0,-0.9*t],
                    [0,0,1,t],
                    [0,0,0,1],
                ]
            )

        def rot_phi(phi):
            return np.array(
                [
                    [1,0,0,0],
                    [0,np.cos(phi),-np.sin(phi),0],
                    [0,np.sin(phi), np.cos(phi),0],
                    [0,0,0,1],
                ]
            )

        def rot_theta(th):
            return np.array(
                [
                    [np.cos(th),0,-np.sin(th),0],
                    [0,1,0,0],
                    [np.sin(th),0, np.cos(th),0],
                    [0,0,0,1],
                ]
            )

        c2w = rot_theta(theta) @ rot_phi(phi) @ trans_t(radius)
        c2w = np.array(
            [[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]]
        ) @ c2w
        return c2w[:3]

    spherical_poses = []

    for th in np.linspace(0, 2*np.pi, n_poses+1)[:-1]:
        spherical_poses += [spherical_pose(th, -np.pi/5, radius)] # 36 degree view downwards

    return np.stack(spherical_poses, 0)

def correct_poses_bounds(poses, bounds, flip=True, use_train_pose=False, center=True, train_poses=None):
    # Original poses has rotation in form "down right back", change to "right up back"
    # See https://github.com/bmild/nerf/issues/34
    if flip:
        poses = np.concatenate([poses[..., 1:2], -poses[..., :1], poses[..., 2:4]], -1)

    # See https://github.com/bmild/nerf/issues/34
    if train_poses is None:
        near_original = bounds.min()
        scale_factor = near_original * 0.75 # 0.75 is the default parameter
        bounds /= scale_factor
        poses[..., :3, 3] /= scale_factor

    # Recenter
    if center:
        if use_train_pose:
            if train_poses is not None:
                poses, ref_pose = center_poses_with(poses, train_poses)
            else:
                poses, ref_pose = center_poses_reference(poses)
        else:
            poses, ref_pose = center_poses(poses)
    else:
        ref_pose = poses[0]

    return poses, ref_pose, bounds

# Assumes centered poses
def get_bounding_sphere(poses):
    dists = np.linalg.norm(poses[:, :3, -1], axis=-1)
    return dists.max()

def get_bounding_box(poses):
    min_x, max_x = poses[:, 0, -1].min(), poses[:, 0, -1].max()
    min_y, max_y = poses[:, 1, -1].min(), poses[:, 1, -1].max()
    min_z, max_z = poses[:, 2, -1].min(), poses[:, 2, -1].max()

    return [min_x, min_y, min_z, max_x, max_y, max_z]

def p34_to_44(p):
    return np.concatenate(
        [p, np.tile(np.reshape(np.eye(4)[-1,:], [1,1,4]), [p.shape[0], 1,1])], 1
    )

def poses_to_twists(poses):
    twists = []

    for i in range(poses.shape[0]):
        M = scipy.linalg.logm(poses[i])
        twist = np.stack(
            [
                M[..., 2, 1],
                M[..., 0, 2],
                M[..., 1, 0],
                M[..., 0, 3],
                M[..., 1, 3],
                M[..., 2, 3],
            ],
            axis=-1
        )
        twists.append(twist)

    return np.stack(twists, 0)

def twists_to_poses(twists):
    poses = []

    for i in range(twists.shape[0]):
        twist = twists[i]
        null = np.zeros_like(twist[..., 0])

        M = np.stack(
            [
                np.stack(
                    [
                    null,
                    twist[..., 2],
                    -twist[..., 1],
                    null
                    ],
                    axis=-1
                ),
                np.stack(
                    [
                    -twist[..., 2],
                    null,
                    twist[..., 0],
                    null
                    ],
                    axis=-1
                ),
                np.stack(
                    [
                    twist[..., 1],
                    -twist[..., 0],
                    null,
                    null
                    ],
                    axis=-1
                ),
                np.stack(
                    [
                    twist[..., 3],
                    twist[..., 4],
                    twist[..., 5],
                    null
                    ],
                    axis=-1
                ),
            ],
            axis=-1
        )

        poses.append(scipy.linalg.expm(M))

    return np.stack(poses, 0)

def interpolate_poses(poses, supersample):
    t = np.linspace(0, 1, supersample, endpoint=False).reshape(1, supersample, 1)
    twists = poses_to_twists(p34_to_44(poses))

    interp_twists = twists.reshape(-1, 1, twists.shape[-1])
    interp_twists = (1 - t) * interp_twists[:-1] + t * interp_twists[1:]
    interp_twists = interp_twists.reshape(-1, twists.shape[-1])
    interp_twists = np.concatenate([interp_twists, np.tile(twists[-1:], [supersample, 1])], 0)

    return twists_to_poses(interp_twists)[:, :3, :4]
