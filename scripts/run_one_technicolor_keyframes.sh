#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

CUDA_VISIBLE_DEVICES=$1 python main.py experiment/dataset=technicolor \
    experiment/training=technicolor_tensorf \
    experiment.training.val_every=5 \
    experiment.training.test_every=20 \
    experiment.training.render_every=40 \
    experiment/model=technicolor_z_plane \
    experiment.params.print_loss=True \
    experiment.dataset.collection=$2 \
    +experiment/regularizers/tensorf=tv_4000_technicolor \
    experiment.training.num_epochs=40 \
    experiment.dataset.keyframe_step=$3 \
    experiment.params.name=technicolor_$2_keyframe_step_$3

