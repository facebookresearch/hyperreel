#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

CUDA_VISIBLE_DEVICES=$1 python main.py experiment/dataset=shiny_dense_large \
    experiment/training=shiny_tensorf \
    experiment.training.val_every=10 \
    experiment.training.ckpt_every=10 \
    experiment.training.test_every=20 \
    experiment.training.render_every=50 \
    ++experiment.training.num_epochs=100 \
    experiment/model=shiny_z_plane$3 \
    experiment.params.print_loss=True \
    experiment.dataset.collection=$2 \
    +experiment/regularizers/tensorf=tv_4000 \
    experiment.params.name=shiny_$2$3 \
    +experiment/visualizers/embedding=default \
    experiment.dataset.img_wh=[512,512] \
    experiment.training.render_ray_chunk=1048576 \
    experiment.training.ray_chunk=1048576 \
    experiment.training.net_chunk=1048576 \
    experiment.params.save_results=False \
    experiment.training.num_iters=100 \
    ++experiment.training.num_epochs=1000 \
    experiment.params.render_only=True \
    +experiment.params.interact_only=True
