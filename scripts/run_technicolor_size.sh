#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

bash scripts/run_one_technicolor_ablate.sh $1 $2 technicolor_z_plane_tiny technicolor_tiny_$2
bash scripts/run_one_technicolor_ablate.sh $1 $2 technicolor_z_plane_small technicolor_small_$2
bash scripts/run_one_technicolor_ablate.sh $1 $2 technicolor_z_plane_large technicolor_large_$2
