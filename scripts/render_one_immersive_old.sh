#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

CUDA_VISIBLE_DEVICES=$1 python main.py experiment/dataset=immersive_old \
    experiment/training=immersive_tensorf \
    experiment.training.val_every=5 \
    experiment.training.render_every=10 \
    experiment.training.test_every=10 \
    experiment.training.ckpt_every=1000 \
    experiment.training.test_every=100 \
    experiment.training.num_epochs=1000 \
    experiment/model=immersive_cylinder_pe \
    experiment.params.print_loss=True \
    experiment.dataset.collection=$2 \
    experiment.dataset.val_set=[0] \
    experiment.dataset.val_all=False \
    +experiment/regularizers/tensorf=tv_4000 \
    experiment.dataset.start_frame=$3 \
    experiment.params.name=immersive_$2_start_$3 \
    experiment.params.save_results=True \
    experiment.training.num_iters=100 \
    experiment.params.render_only=True
