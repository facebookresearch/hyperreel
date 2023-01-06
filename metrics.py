#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import numpy as np
from kornia.losses import ssim as dssim
import lpips

from skimage.metrics import structural_similarity, peak_signal_noise_ratio # @manual=fbsource//third-party/pythonlibs/native/scikit-image:scikit-image


def mse(image_pred, image_gt, valid_mask=None, reduction='mean'):
    value = (image_pred-image_gt)**2

    if valid_mask is not None:
        value = value[valid_mask]
    if reduction == 'mean':
        return torch.mean(value)

    return value

def psnr(image_pred, image_gt):
    #image_gt = np.array(image_gt)
    #return peak_signal_noise_ratio(np.array(image_pred), image_gt, data_range=(image_gt.max() - image_gt.min()))
    #return peak_signal_noise_ratio(np.round(np.array(image_pred) * 255), np.round(np.array(image_gt) * 255), data_range=255.0)

    return peak_signal_noise_ratio(np.array(image_pred), np.array(image_gt), data_range=1.0)

    #return np.array(-10 * torch.log10(mse(torch.tensor(image_pred), torch.tensor(image_gt), torch.tensor((image_gt != 1.0).any(-1)).unsqueeze(-1).repeat((1, 1, 3)), 'mean')))

def ssim(image0, image1):
    return structural_similarity(np.array(image1), np.array(image0), win_size=11, multichannel=True, gaussian_weights=True, data_range=1.0)

def psnr_gpu(image_pred, image_gt, valid_mask=None, reduction='mean'):
    #return 10*torch.log10(torch.square(image_gt.max() - image_gt.min()) / mse(image_pred, image_gt, valid_mask, reduction))
    #return -10*torch.log10(mse(image_pred, image_gt, valid_mask, reduction))

    #image_pred = torch.round(image_pred * 255).float()
    #image_gt = torch.round(image_gt * 255).float()
    #return -10*torch.log10(mse(image_pred, image_gt, valid_mask, reduction) / (255 * 255))

    return -10*torch.log10(mse(image_pred, image_gt, valid_mask, reduction))

def ssim_gpu(image_pred, image_gt, reduction='mean'):
    """
    image_pred and image_gt: (1, 3, H, W)
    """
    dssim_ = dssim(image_pred, image_gt, 11, reduction) # dissimilarity in [0, 1]
    return 1-2*dssim_ # in [-1, 1]

def compute_lpips(image0, image1, lpips_model):
    gt_lpips = torch.tensor(image0).clone().cpu() * 2.0 - 1.0
    predict_image_lpips = torch.tensor(image1).clone().detach().cpu() * 2.0 - 1.0
    lpips_result = lpips_model.forward(predict_image_lpips, gt_lpips).cpu().detach().numpy()
    return np.squeeze(lpips_result)

def get_mean_outputs(outputs, cpu=False):
    # Stack
    stacked = {}

    for x in outputs:
        for key, val in x.items():
            if key not in stacked:
                stacked[key] = []

            stacked[key].append(val)

    # Mean
    mean = {}

    for key in stacked:
        if cpu:
            mean_val = np.stack(stacked[key]).mean()
        else:
            mean_val = torch.stack(stacked[key]).mean()

        mean[key] = mean_val
    
    #if cpu:
    #    if 'val/loss' in mean:
    #        mean['val/psnr'] = -10*np.log10(mean['val/loss'])
    #    elif 'train/loss' in mean:
    #        mean['train/psnr'] = -10*np.log10(mean['train/loss'])
    #else:
    #    if 'val/loss' in mean:
    #        mean['val/psnr'] = -10*torch.log10(mean['val/loss'])
    #    elif 'train/loss' in mean:
    #        mean['train/psnr'] = -10*torch.log10(mean['train/loss'])

    return mean
