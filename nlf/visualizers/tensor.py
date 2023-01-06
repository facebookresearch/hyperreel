#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch

from .base import BaseVisualizer


class TensorVisualizer(BaseVisualizer):
    def __init__(
        self,
        system,
        cfg
    ):
        super().__init__(system, cfg)

    def validation(self, batch, batch_idx):
        if batch_idx > 0:
            return

        system = self.get_system()
        W, H = batch['W'], batch['H']

        outputs = {}

        # Render tensors (unwarped)
        if system.is_subdivided:
            tensors = system.render_fn.model.ray_model.color_model.net.tensors[0].tensors[0].tensor[:, ..., 0]
        else:
            tensors = system.render_fn.model.color_model.net.tensors[0].tensors[0].tensor[:, ..., 0]

        for i in range(tensors.shape[0]):
            if len(tensors[i].shape) == 2:
                rgb = torch.sigmoid(tensors[i].permute(1, 0)[None, ..., :3]).repeat(128, 1, 1).cpu()
                rgb = rgb.permute(2, 0, 1)
                outputs[f'rgb_unwarped_{i:03d}'] = rgb
            elif len(tensors[i].shape) == 3:
                rgb = torch.sigmoid(tensors[i].permute(1, 2, 0)[..., :3]).cpu()
                rgb = rgb.permute(2, 0, 1)
                outputs[f'rgb_unwarped_{i:03d}'] = rgb

        # Render tensors (layers)
        if system.is_subdivided:
            num_partitions = system.render_fn.model.ray_model.color_model.net.num_partitions
        else:
            num_partitions = system.render_fn.model.color_model.net.num_partitions

        for i in range(num_partitions):
            coords, rgb, = batch['coords'], batch['rgb']
            coords = torch.clone(coords.view(-1, coords.shape[-1]))

            results = system(coords, keep_tensor_partitions=[i])
            rgb = results['rgb'].view(H, W, 3).cpu().numpy()
            rgb = rgb.transpose(2, 0, 1)
            outputs[f'rgb_warped_{i:03d}'] = rgb

        return outputs

    def validation_image(self, batch, batch_idx):
        if batch_idx > 0:
            return {}

        # Outputs
        temp_outputs = self.validation(batch, batch_idx)
        outputs = {}

        for key in temp_outputs.keys():
            outputs[f'images/tensor_{key}'] = temp_outputs[key]

        return outputs
