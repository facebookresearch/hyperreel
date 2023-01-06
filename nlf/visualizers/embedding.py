#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch

from .base import BaseVisualizer

from utils.ray_utils import get_epi_rays
from utils.visualization import (
    get_warp_dimensions,
    visualize_warp
)
from datasets.lightfield import LightfieldDataset
from datasets.base import Base6DDataset


class EmbeddingVisualizer(BaseVisualizer):
    def __init__(
        self,
        system,
        cfg
    ):
        super().__init__(system, cfg)

        # Variables to visualize
        self.fields = cfg.fields
        self.data_fields = list(cfg.data_fields) if 'data_fields' in cfg else []
        self.no_over_fields = list(cfg.no_over_fields) if 'no_over_fields' in cfg else []
        self.pred_weights_fields = list(cfg.pred_weights_fields) if 'pred_weights_fields' in cfg else []

        # Vis dims
        self.vis_dims = {}

    def validation(self, batch, batch_idx):
        system = self.get_system()
        W = system.cur_wh[0]
        H = system.cur_wh[1]

        # Get coords
        coords = batch['coords']
        coords = torch.clone(coords.view(-1, coords.shape[-1]))

        # Render fields
        outputs = system.render(
            'forward_multiple',
            coords,
            fields=self.fields.keys(),
            no_over_fields=self.no_over_fields,
            pred_weights_fields=self.pred_weights_fields
        )

        # Data outputs
        data_outputs = {}

        for key in self.data_fields:
            data_outputs[key] = outputs[key].view(H, W, outputs[key].shape[-1]).cpu().numpy()
            data_outputs[key] = data_outputs[key].transpose(2, 0, 1)

        # Visualize outputs
        vis_outputs = {}

        for key in self.fields:
            vis_outputs[key] = outputs[key].view(H * W, outputs[key].shape[-1])

            # Get dimensions to visualize
            if batch_idx == 0:
                self.vis_dims[key] = get_warp_dimensions(
                    vis_outputs[key],
                    W,
                    H,
                    k=min(vis_outputs[key].shape[-1], 3),
                    **dict(self.fields[key])
                )

            # Visualize
            vis_outputs[key] = visualize_warp(
                vis_outputs[key],
                self.vis_dims[key],
                **dict(self.fields[key])
            )

            # Convert to numpy array
            vis_outputs[key] = vis_outputs[key].view(H, W, vis_outputs[key].shape[-1]).cpu().numpy()
            vis_outputs[key] = vis_outputs[key].transpose(2, 0, 1)

        # Return
        return data_outputs, vis_outputs

    def validation_image(self, batch, batch_idx):
        data_outputs, vis_outputs = self.validation(batch, batch_idx)
        outputs = {}

        for key in data_outputs.keys():
            outputs[f'data/{key}'] = data_outputs[key]

        for key in vis_outputs.keys():
            outputs[f'images/embedding_{key}'] = vis_outputs[key]

        return outputs

    def validation_video(self, batch, batch_idx):
        data_outputs, vis_outputs = self.validation(batch, batch_idx)
        outputs = {}

        for key in vis_outputs.keys():
            outputs[f'videos/embedding_{key}'] = vis_outputs[key]

        return outputs
