#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

CUDA_VISIBLE_DEVICES=$1 python main.py experiment/dataset=donerf_large \
    experiment/training=donerf_tensorf \
    experiment.params.render_only=True \
    experiment.training.val_every=1 \
    experiment.training.render_every=1 \
    experiment.training.test_every=1 \
    experiment.training.num_epochs=1000 \
    experiment/model=donerf_cylinder \
    experiment.params.print_loss=True \
    experiment.dataset.collection=$2 \
    experiment.params.save_results=False \
    experiment.training.num_iters=100 \
    experiment.params.render_only=True



