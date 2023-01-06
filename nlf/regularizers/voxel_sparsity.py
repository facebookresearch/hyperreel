#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch

from .base import BaseRegularizer


class VoxelSparsityRegularizer(BaseRegularizer):
    def __init__(
        self,
        system,
        cfg
    ):
        super().__init__(system, cfg)

    def get_subdivision(self):
        system = self.get_system()
        return system.subdivision

    def loss(self, batch, batch_results, batch_idx):
        system = self.get_system()
        pos_model = system.render_fn.model.pos_model
        chunk = system.cfg.training.ray_chunk
        subdivision = self.get_subdivision()

        points = subdivision.voxel_centers[0, :subdivision.num_voxels].cuda()
        sampled_xyz = points.unsqueeze(1)

        sh = sampled_xyz.shape[:-1] # noqa
        sampled_idx = torch.arange(
            points.size(0), device=points.device
        )[:, None].expand(*sampled_xyz.size()[:2])
        sampled_xyz, sampled_idx = sampled_xyz.reshape(-1, 3), sampled_idx.reshape(-1)

        ## Evaluate
        B = sampled_xyz.shape[0]
        out_chunks = []

        for i in range(0, B, chunk):
            # Get points, idx
            cur_pts = sampled_xyz[i:i+chunk].unsqueeze(1)
            cur_idx = sampled_idx[i:i+chunk].unsqueeze(1)

            cur_mask = cur_idx.eq(-1)
            cur_idx[cur_mask] = 0

            # Get codes
            cur_codes = subdivision.get_vertex_codes(cur_pts, cur_idx, cur_mask)

            # Combine inputs
            cur_inps = torch.cat([cur_pts, cur_codes], -1)
            cur_inps = cur_inps.view(-1, cur_inps.shape[-1])
            out_chunks += [pos_model.pos_forward(cur_inps)]

        out = torch.cat(out_chunks, 0)
        out = out[..., -1].view(-1)

        loss = -(out * torch.log(out + 1e-8) + (1 - out) * torch.log(1 - out + 1e-8)).mean()
        print(out.max(), loss)
        return loss
