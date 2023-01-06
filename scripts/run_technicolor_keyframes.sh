#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

bash scripts/run_one_technicolor_keyframes.sh $1 $2 1
bash scripts/run_one_technicolor_keyframes.sh $1 $2 4
bash scripts/run_one_technicolor_keyframes.sh $1 $2 16
bash scripts/run_one_technicolor_keyframes.sh $1 $2 50