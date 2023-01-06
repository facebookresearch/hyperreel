#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

intersect_dict = {}

# Add primitives
from .primitive import primitive_intersect_dict
for k, v in primitive_intersect_dict.items(): intersect_dict[k] = v

# Add voxel
from .voxel import voxel_intersect_dict
for k, v in voxel_intersect_dict.items(): intersect_dict[k] = v

# Add z
from .z import z_intersect_dict
for k, v in z_intersect_dict.items(): intersect_dict[k] = v
