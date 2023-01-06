#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

CUDA_VISIBLE_DEVICES=$1 python main.py experiment/dataset=donerf \
    experiment/training=donerf_tensorf_small \
    experiment.training.val_every=1 \
    experiment.training.test_every=20 \
    experiment.training.ckpt_every=1 \
    experiment.training.render_every=80 \
    experiment.training.num_epochs=80 \
    experiment/model=donerf_cylinder_small \
    experiment.params.print_loss=True \
    experiment.dataset.collection=$2 \
    +experiment/regularizers/tensorf=tv_4000_small \
    experiment.params.name=donerf_$2_small \
    experiment.training.num_iters=100


