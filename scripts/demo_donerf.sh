#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

CUDA_VISIBLE_DEVICES=$1 python main.py experiment/dataset=donerf_large \
    experiment/training=donerf_tensorf \
    experiment.training.val_every=10 \
    experiment.training.test_every=20 \
    experiment.training.render_every=50 \
    experiment.training.ckpt_every=80 \
    experiment/model=donerf_sphere \
    experiment.params.print_loss=True \
    experiment.dataset.collection=$2 \
    +experiment/regularizers/tensorf=tv_4000 \
    experiment.params.name=donerf_$2_sphere \
    experiment.dataset.img_wh=[512,512] \
    experiment.training.render_ray_chunk=1048576 \
    experiment.training.ray_chunk=1048576 \
    experiment.training.net_chunk=1048576 \
    experiment.params.save_results=False \
    experiment.training.num_iters=100 \
    experiment.training.num_epochs=1000 \
    experiment.params.render_only=True \
    +experiment.params.interact_only=True


