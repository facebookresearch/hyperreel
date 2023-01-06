#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np # noqa

from .base import BaseVisualizer


class ClosestViewVisualizer(BaseVisualizer):
    def __init__(
        self,
        system,
        cfg
    ):
        super().__init__(system, cfg)

    def validation(self, batch):
        system = self.get_system()

        if 'pose' not in batch:
            return {}

        if 'lightfield' not in system.trainer.datamodule.train_dataset.dataset_cfg \
            or system.trainer.datamodule.train_dataset.keyframe_step == -1:

            if 'time' in batch:
                rgb = system.trainer.datamodule.train_dataset.get_closest_rgb(batch['pose'], batch['time']).cpu()
            else:
                rgb = system.trainer.datamodule.train_dataset.get_closest_rgb(batch['pose']).cpu()

            rgb = rgb.permute(2, 0, 1)

            return {
                'rgb': rgb
            }
        else:
            return {}

    def validation_video(self, batch, batch_idx):
        temp_outputs = self.validation(batch)
        outputs = {}

        for key in temp_outputs.keys():
            outputs[f'videos/closest_{key}'] = temp_outputs[key]

        return outputs

    def validation_image(self, batch, batch_idx):
        return {}
