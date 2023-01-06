#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

bash scripts/run_one_technicolor_ablate.sh $1 birthday technicolor_z_plane_no_sample technicolor_no_sample_birthday
bash scripts/run_one_technicolor_ablate.sh $1 fabien technicolor_z_plane_no_sample technicolor_no_sample_fabien
bash scripts/run_one_technicolor_ablate.sh $1 painter technicolor_z_plane_no_sample technicolor_no_sample_painter
bash scripts/run_one_technicolor_ablate.sh $1 theater technicolor_z_plane_no_sample technicolor_no_sample_theater
bash scripts/run_one_technicolor_ablate.sh $1 trains technicolor_z_plane_no_sample technicolor_no_sample_trains
