#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

CUDA_VISIBLE_DEVICES=$1 python main.py experiment/dataset=neural_3d \
    experiment/training=neural_3d_tensorf \
    experiment.training.val_every=5 \
    experiment.training.test_every=30 \
    experiment.training.ckpt_every=10 \
    experiment.training.render_every=30 \
    experiment.training.num_epochs=30 \
    experiment/model=neural_3d_z_plane \
    experiment.params.print_loss=True \
    experiment.dataset.collection=$2 \
    +experiment/regularizers/tensorf=tv_4000 \
    experiment.dataset.start_frame=$3 \
    experiment.params.name=neural_3d_$2_start_$3 \
    experiment.dataset.val_all=True \
    experiment.dataset.val_set=[]
