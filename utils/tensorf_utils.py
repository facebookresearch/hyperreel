#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright (c) 2022 Anpei Chen
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import cv2
import torch
import numpy as np
from PIL import Image

import torchvision.transforms as T
import torch.nn.functional as F
import scipy.signal

from utils.sh_utils import eval_sh_bases
from utils.ray_utils import dot

mse2psnr = lambda x : -10. * torch.log(x) / torch.log(torch.Tensor([10.]))


def visualize_depth_numpy(depth, minmax=None, cmap=cv2.COLORMAP_JET):
    """
    depth: (H, W)
    """

    x = np.nan_to_num(depth) # change nan to 0
    if minmax is None:
        mi = np.min(x[x>0]) # get minimum positive depth (ignore background)
        ma = np.max(x)
    else:
        mi,ma = minmax

    x = (x-mi)/(ma-mi+1e-8) # normalize to 0~1
    x = (255*x).astype(np.uint8)
    x_ = cv2.applyColorMap(x, cmap)
    return x_, [mi,ma]

def init_log(log, keys):
    for key in keys:
        log[key] = torch.tensor([0.0], dtype=float)
    return log

def visualize_depth(depth, minmax=None, cmap=cv2.COLORMAP_JET):
    """
    depth: (H, W)
    """
    if type(depth) is not np.ndarray:
        depth = depth.cpu().numpy()

    x = np.nan_to_num(depth) # change nan to 0
    if minmax is None:
        mi = np.min(x[x>0]) # get minimum positive depth (ignore background)
        ma = np.max(x)
    else:
        mi,ma = minmax

    x = (x-mi)/(ma-mi+1e-8) # normalize to 0~1
    x = (255*x).astype(np.uint8)
    x_ = Image.fromarray(cv2.applyColorMap(x, cmap))
    x_ = T.ToTensor()(x_)  # (3, H, W)
    return x_, [mi,ma]

def N_to_reso(n_voxels, bbox):
    xyz_min, xyz_max = bbox
    voxel_size = ((xyz_max - xyz_min).prod() / n_voxels).pow(1 / 3)
    return ((xyz_max - xyz_min) / voxel_size).long().tolist()

def cal_n_samples(reso, step_ratio=0.5):
    return int(np.linalg.norm(reso)/step_ratio)




__LPIPS__ = {}
def init_lpips(net_name, device):
    assert net_name in ['alex', 'vgg']
    import lpips
    print(f'init_lpips: lpips_{net_name}')
    return lpips.LPIPS(net=net_name, version='0.1').eval().to(device)

def rgb_lpips(np_gt, np_im, net_name, device):
    if net_name not in __LPIPS__:
        __LPIPS__[net_name] = init_lpips(net_name, device)
    gt = torch.from_numpy(np_gt).permute([2, 0, 1]).contiguous().to(device)
    im = torch.from_numpy(np_im).permute([2, 0, 1]).contiguous().to(device)
    return __LPIPS__[net_name](gt, im, normalize=True).item()


def findItem(items, target):
    for one in items:
        if one[:len(target)]==target:
            return one
    return None


''' Evaluation metrics (ssim, lpips)
'''
def rgb_ssim(img0, img1, max_val,
             filter_size=11,
             filter_sigma=1.5,
             k1=0.01,
             k2=0.03,
             return_map=False):
    # Modified from https://github.com/google/mipnerf/blob/16e73dfdb52044dcceb47cda5243a686391a6e0f/internal/math.py#L58
    assert len(img0.shape) == 3
    assert img0.shape[-1] == 3
    assert img0.shape == img1.shape

    # Construct a 1D Gaussian blur filter.
    hw = filter_size // 2
    shift = (2 * hw - filter_size + 1) / 2
    f_i = ((np.arange(filter_size) - hw + shift) / filter_sigma)**2
    filt = np.exp(-0.5 * f_i)
    filt /= np.sum(filt)

    # Blur in x and y (faster than the 2D convolution).
    def convolve2d(z, f):
        return scipy.signal.convolve2d(z, f, mode='valid')

    filt_fn = lambda z: np.stack([
        convolve2d(convolve2d(z[...,i], filt[:, None]), filt[None, :])
        for i in range(z.shape[-1])], -1)
    mu0 = filt_fn(img0)
    mu1 = filt_fn(img1)
    mu00 = mu0 * mu0
    mu11 = mu1 * mu1
    mu01 = mu0 * mu1
    sigma00 = filt_fn(img0**2) - mu00
    sigma11 = filt_fn(img1**2) - mu11
    sigma01 = filt_fn(img0 * img1) - mu01

    # Clip the variances and covariances to valid values.
    # Variance must be non-negative:
    sigma00 = np.maximum(0., sigma00)
    sigma11 = np.maximum(0., sigma11)
    sigma01 = np.sign(sigma01) * np.minimum(
        np.sqrt(sigma00 * sigma11), np.abs(sigma01))
    c1 = (k1 * max_val)**2
    c2 = (k2 * max_val)**2
    numer = (2 * mu01 + c1) * (2 * sigma01 + c2)
    denom = (mu00 + mu11 + c1) * (sigma00 + sigma11 + c2)
    ssim_map = numer / denom
    ssim = np.mean(ssim_map)
    return ssim_map if return_map else ssim


import torch.nn as nn
class TVLoss(nn.Module):
    def __init__(self,TVLoss_weight=1):
        super(TVLoss,self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self,x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:,:,1:,:])
        count_w = self._tensor_size(x[:,:,:,1:])
        h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()
        w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()
        return self.TVLoss_weight*2*(h_tv/count_h+w_tv/count_w)/batch_size

    def _tensor_size(self,t):
        return t.size()[1]*t.size()[2]*t.size()[3]

import plyfile
import skimage.measure
def convert_sdf_samples_to_ply(
    pytorch_3d_sdf_tensor,
    ply_fp,
    bbox,
    level=0.5,
    offset=None,
    scale=None,
):
    """
    Convert sdf samples to .ply

    :param pytorch_3d_sdf_tensor: a torch.FloatTensor of shape (n,n,n)
    :voxel_grid_origin: a list of three floats: the bottom, left, down origin of the voxel grid
    :voxel_size: float, the size of the voxels
    :ply_fp: file object to save ply to

    This function adapted from: https://github.com/RobotLocomotion/spartan
    """

    numpy_3d_sdf_tensor = pytorch_3d_sdf_tensor.numpy()
    voxel_size = list((bbox[1]-bbox[0]) / np.array(pytorch_3d_sdf_tensor.shape))

    verts, faces, normals, values = skimage.measure.marching_cubes(
        numpy_3d_sdf_tensor, level=level, spacing=voxel_size
    )
    faces = faces[...,::-1] # inverse face orientation

    # transform from voxel coordinates to camera coordinates
    # note x and y are flipped in the output of marching_cubes
    mesh_points = np.zeros_like(verts)
    mesh_points[:, 0] = bbox[0,0] + verts[:, 0]
    mesh_points[:, 1] = bbox[0,1] + verts[:, 1]
    mesh_points[:, 2] = bbox[0,2] + verts[:, 2]

    # apply additional offset and scale
    if scale is not None:
        mesh_points = mesh_points / scale
    if offset is not None:
        mesh_points = mesh_points - offset

    # try writing to the ply file

    num_verts = verts.shape[0]
    num_faces = faces.shape[0]

    verts_tuple = np.zeros((num_verts,), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])

    for i in range(0, num_verts):
        verts_tuple[i] = tuple(mesh_points[i, :])

    faces_building = []
    for i in range(0, num_faces):
        faces_building.append(((faces[i, :].tolist(),)))
    faces_tuple = np.array(faces_building, dtype=[("vertex_indices", "i4", (3,))])

    el_verts = plyfile.PlyElement.describe(verts_tuple, "vertex")
    el_faces = plyfile.PlyElement.describe(faces_tuple, "face")

    ply_data = plyfile.PlyData([el_verts, el_faces])
    ply_data.write(ply_fp)


def positional_encoding(positions, freqs):

    freq_bands = (2 ** torch.arange(freqs).float()).to(positions.device)  # (F,)
    pts = (positions[..., None] * freq_bands).reshape(
        positions.shape[:-1] + (freqs * positions.shape[-1],)
    )  # (..., DF)
    pts = torch.cat([torch.sin(pts), torch.cos(pts)], dim=-1)
    return pts


def raw2alpha(sigma, dist):
    alpha = 1.0 - torch.exp(-sigma * dist)

    T = torch.cumprod(
        torch.cat(
            [torch.ones(alpha.shape[0], 1).to(alpha.device), 1.0 - alpha + 1e-10], -1
        ),
        -1,
    )

    weights = alpha * T[:, :-1]  # [N_rays, N_samples]
    return alpha, weights, T[:, -1:]


def alpha2weights(alpha):
    T = torch.cumprod(
        torch.cat(
            [torch.ones(alpha.shape[0], 1).to(alpha.device), 1.0 - alpha + 1e-10], -1
        ),
        -1,
    )

    weights = alpha * T[:, :-1]  # [N_rays, N_samples]
    return weights

def scale_shift_color_all(rgb, color_scale, color_shift):
    color_scale = color_scale.view(*rgb.shape)
    color_shift = color_shift.view(*rgb.shape)

    #print(color_shift.mean((0, 1)))

    return rgb * (color_scale + 1.0) + color_shift

def scale_shift_color_one(rgb, rgb_map, x):
    color_scale = x['color_scale_global'].view(*rgb.shape)[:, 0, :]
    color_shift = x['color_shift_global'].view(*rgb.shape)[:, 0, :]

    #print(color_shift.mean(0))

    return rgb_map * (color_scale + 1.0) + color_shift

def transform_color_all(rgb, color_transform, color_shift):
    color_transform = color_transform.view(rgb.shape[0], 3, 3)
    color_shift = color_shift.view(*rgb.shape)

    rgb = torch.stack(
        [
            rgb[..., 0] + dot(rgb, color_transform[..., 0, :]),
            rgb[..., 1] + dot(rgb, color_transform[..., 1, :]),
            rgb[..., 2] + dot(rgb, color_transform[..., 2, :]),
        ],
        -1
    )
    #rgb = torch.stack(
    #    [
    #        dot(rgb, color_transform[..., 0, :]),
    #        dot(rgb, color_transform[..., 1, :]),
    #        dot(rgb, color_transform[..., 2, :]),
    #    ],
    #    -1
    #)

    #print(color_transform.mean(0), color_shift.mean(0))

    return rgb + color_shift

def transform_color_one(rgb, rgb_map, x):
    color_transform = x['color_transform_global'].view(rgb.shape[0], -1, 3, 3)[:, 0, :, :]
    color_shift = x['color_shift_global'].view(rgb.shape[0], -1, 3)[:, 0, :]

    rgb_map = torch.stack(
        [
            rgb_map[..., 0] + dot(rgb_map, color_transform[..., 0, :]),
            rgb_map[..., 1] + dot(rgb_map, color_transform[..., 1, :]),
            rgb_map[..., 2] + dot(rgb_map, color_transform[..., 2, :]),
        ],
        -1
    )
    #rgb_map = torch.stack(
    #    [
    #        dot(rgb_map, color_transform[..., 0, :]),
    #        dot(rgb_map, color_transform[..., 1, :]),
    #        dot(rgb_map, color_transform[..., 2, :]),
    #    ],
    #    -1
    #)

    #print(color_transform.mean(0), color_shift.mean(0))

    return rgb_map + color_shift


def SHRender(xyz_sampled, viewdirs, features, kwargs):
    sh_mult = eval_sh_bases(2, viewdirs[..., :3])[:, None]
    rgb_sh = features.view(-1, 3, sh_mult.shape[-1])
    rgb = torch.relu(torch.sum(sh_mult * rgb_sh, dim=-1) + 0.5)
    return rgb


def RGBRender(xyz_sampled, viewdirs, features, kwargs):
    rgb = features
    return torch.sigmoid(rgb)


def RGBIdentityRender(xyz_sampled, viewdirs, features, kwargs):
    rgb = features
    return torch.abs(rgb + 0.5)


def RGBtLinearRender(xyz_sampled, viewdirs, features, kwargs):
    # Coefficients
    coeffs = features.view(-1, 3, 2)

    # Basis functions
    t = kwargs["times"].view(-1, 1)

    basis = torch.cat(
        [
            torch.ones_like(t),
            t,
        ],
        dim=-1,
    )

    # RGB
    rgb = torch.relu(torch.sum(basis.unsqueeze(1) * coeffs, dim=-1) + 0.5)

    return rgb


def RGBtFourierRender(xyz_sampled, viewdirs, features, kwargs):
    frames_per_keyframe = kwargs["frames_per_keyframe"]
    num_keyframes = kwargs["num_keyframes"]
    total_num_frames = kwargs["total_num_frames"]
    time_scale_factor = num_keyframes * (total_num_frames - 1) / (total_num_frames)

    # Coefficients
    coeffs = features.view(-1, 3, frames_per_keyframe * 2 + 1)

    # Basis functions
    time_offset = kwargs["time_offset"].view(-1, 1) * time_scale_factor
    t = kwargs["times"].view(-1, 1)

    freqs = torch.linspace(
        0, frames_per_keyframe - 1, frames_per_keyframe, device=time_offset.device
    )[None]
    basis = torch.cat(
        [
            t,
            torch.cos(time_offset * freqs * 2 * np.pi),
            torch.sin(time_offset * freqs * 2 * np.pi),
        ],
        dim=-1,
    )

    # RGB
    rgb = torch.relu(torch.sum(basis.unsqueeze(1) * coeffs, dim=-1) + 0.5)

    return rgb


def DensityRender(density_features, kwargs):
    return density_features[..., 0]


def DensityLinearRender(density_features, kwargs):
    # Coefficients
    coeffs = density_features.view(-1, 1, 2)

    # Basis functions
    t = kwargs["times"].view(-1, 1)

    basis = torch.cat(
        [
            torch.ones_like(t),
            t,
        ],
        dim=-1,
    )

    # Density
    density = torch.sum(basis.unsqueeze(1) * coeffs, dim=-1)[..., 0]

    return density


def DensityFourierRender(density_features, kwargs):
    frames_per_keyframe = kwargs["frames_per_keyframe"]
    num_keyframes = kwargs["num_keyframes"]
    total_num_frames = kwargs["total_num_frames"]
    time_scale_factor = num_keyframes * (total_num_frames - 1) / (total_num_frames)

    # Coefficients
    coeffs = density_features.view(-1, 1, frames_per_keyframe * 2 + 1)

    # Basis functions
    time_offset = kwargs["time_offset"].view(-1, 1) * time_scale_factor
    t = kwargs["times"].view(-1, 1)

    freqs = torch.linspace(
        0, frames_per_keyframe - 1, frames_per_keyframe, device=time_offset.device
    )[None]
    basis = torch.cat(
        [
            t,
            torch.cos(time_offset * freqs * 2 * np.pi),
            torch.sin(time_offset * freqs * 2 * np.pi),
        ],
        dim=-1,
    )

    # Density
    density = torch.sum(basis.unsqueeze(1) * coeffs, dim=-1)[..., 0]

    return density


class AlphaGridMask(torch.nn.Module):
    def __init__(self, device, aabb, alpha_volume):
        super().__init__()

        self.opt_group = "color"
        self.device = device

        self.register_buffer('alpha_aabb', aabb.to(self.device))
        self.register_buffer('alpha_volume', alpha_volume.view(1, 1, *alpha_volume.shape[-3:]))

        self.aabbSize = self.alpha_aabb[1] - self.alpha_aabb[0]
        self.invgridSize = 1.0 / self.aabbSize * 2
        self.gridSize = torch.LongTensor(
            [alpha_volume.shape[-1], alpha_volume.shape[-2], alpha_volume.shape[-3]]
        ).to(self.device)

    def sample_alpha(self, xyz_sampled):
        xyz_sampled = self.normalize_coord(xyz_sampled)
        alpha_vals = F.grid_sample(
            self.alpha_volume, xyz_sampled.view(1, -1, 1, 1, 3), align_corners=True
        ).view(-1)

        return alpha_vals

    def normalize_coord(self, xyz_sampled):
        return (xyz_sampled - self.alpha_aabb[0]) * self.invgridSize - 1