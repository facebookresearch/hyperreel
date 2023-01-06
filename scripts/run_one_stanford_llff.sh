#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

CUDA_VISIBLE_DEVICES=$1 python main.py experiment/dataset=stanford_llff \
    experiment/training=stanford_tensorf \
    experiment.training.val_every=5 \
    experiment.training.render_every=5 \
    experiment.training.ckpt_every=20 \
    experiment.training.test_every=20 \
    ++experiment.training.num_epochs=100 \
    experiment/model=stanford_z_plane$4 \
    experiment.dataset.use_ndc=False \
    experiment.params.print_loss=True \
    experiment.dataset.collection=$2 \
    experiment.dataset.lightfield_step=$3 \
    experiment.params.name=stanford_llff_$2_step_$3$4_no_ndc \
    +experiment/regularizers/tensorf=tv_4000 \

