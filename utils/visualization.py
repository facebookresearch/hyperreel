#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import numpy as np


def get_warp_dimensions(
    embedding,
    W,
    H,
    k=3,
    sort=False,
    **kwargs
):
    if sort:
        embedding_std = torch.std(embedding, 0, True)
        return list(torch.argsort(-embedding_std, axis=-1)[:k].cpu().numpy())
    else:
        return list(range(0, embedding.shape[-1]))

def visualize_warp(
    embedding,
    warp_dims,
    use_abs=False,
    bounds=None,
    normalize=False,
    **kwargs
):

    if embedding.shape[-1] > 1:
        warp_vis = embedding[..., warp_dims]
    else:
        warp_vis = embedding

    if use_abs:
        warp_vis = torch.abs(warp_vis)

    if bounds is not None and len(bounds) > 0:
        bounds_min = torch.tensor(bounds[0], device=warp_vis.device).view(1, -1)
        bounds_max = torch.tensor(bounds[1], device=warp_vis.device).view(1, -1)
        warp_vis = (warp_vis - bounds_min) / (bounds_max - bounds_min)

    if normalize:
        bounds_min = torch.min(warp_vis, dim=0)[0].view(1, -1)
        bounds_max = torch.max(warp_vis, dim=0)[0].view(1, -1)
        warp_vis = (warp_vis - bounds_min) / (bounds_max - bounds_min)

    return warp_vis.clamp(0, 1)
