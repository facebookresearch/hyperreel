#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

CUDA_VISIBLE_DEVICES=$1 python main.py experiment/dataset=llff_large \
    experiment/training=llff_tensorf \
    experiment.training.val_every=5 \
    experiment.training.render_every=10 \
    experiment.training.test_every=10 \
    experiment/model=llff_z_plane \
    experiment.params.print_loss=True \
    experiment.dataset.collection=$2 \
    +experiment/regularizers/tensorf=tv_4000_large

