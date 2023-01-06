#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

CUDA_VISIBLE_DEVICES=$1 python main.py experiment/dataset=technicolor \
    experiment/training=technicolor_tensorf \
    experiment.training.val_every=10 \
    experiment.training.test_every=20 \
    experiment.training.ckpt_every=10 \
    experiment.training.render_every=20 \
    experiment.training.num_epochs=80 \
    experiment/model=technicolor_z_plane \
    experiment.params.print_loss=True \
    experiment.dataset.collection=$2 \
    +experiment/regularizers/tensorf=tv_4000 \
    experiment.dataset.start_frame=$3 \
    experiment.params.name=technicolor_$2_start_$3$4
    experiment.dataset.img_wh=[512,512] \
    experiment.training.render_ray_chunk=1048576 \
    experiment.training.ray_chunk=1048576 \
    experiment.training.net_chunk=1048576 \
    experiment.params.save_results=False \
    experiment.training.num_iters=100 \
    experiment.training.num_epochs=1000 \
    experiment.params.render_only=True \
    +experiment.params.interact_only=True
