#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

CUDA_VISIBLE_DEVICES=$1 python main.py experiment/dataset=stanford_large \
    experiment/training=stanford_tensorf \
    experiment.training.val_every=1 \
    experiment.training.test_every=1 \
    experiment.training.render_every=1 \
    experiment.training.num_epochs=1000 \
    experiment/model=stanford_z_plane_small \
    experiment.params.print_loss=True \
    experiment.dataset.collection=$2 \
    experiment.dataset.lightfield.step=1 \
    experiment.params.name=stanford_$2_step_1_small \
    experiment.dataset.img_wh=[512,512] \
    experiment.training.render_ray_chunk=1048576 \
    experiment.training.ray_chunk=1048576 \
    experiment.training.net_chunk=1048576 \
    experiment.params.save_results=False \
    experiment.training.num_iters=100 \
    experiment.params.render_only=True \
    +experiment.params.interact_only=True



