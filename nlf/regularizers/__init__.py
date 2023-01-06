#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from .coarse import CoarseRegularizer
from .fourier import FourierRegularizer
from .geometry import FlowRegularizer, GeometryRegularizer, GeometryFeedbackRegularizer, RenderWeightRegularizer

from .point import PointRegularizer
from .ray_density import RayDensityRegularizer, SimpleRayDensityRegularizer
from .teacher import BlurryTeacherRegularizer, TeacherRegularizer, TeacherModelRegularizer
from .tensor import TensorTV
from .tensorf import TensoRF
from .voxel_sparsity import VoxelSparsityRegularizer
from .warp import WarpLevelSetRegularizer, WarpRegularizer

regularizer_dict = {
    "fourier": FourierRegularizer,
    "coarse": CoarseRegularizer,
    "teacher": TeacherRegularizer,
    "teacher_model": TeacherModelRegularizer,
    "blurry_teacher": BlurryTeacherRegularizer,
    "voxel_sparsity": VoxelSparsityRegularizer,
    "warp": WarpRegularizer,
    "warp_level": WarpLevelSetRegularizer,
    "point": PointRegularizer,
    "geometry": GeometryRegularizer,
    "geometry_feedback": GeometryFeedbackRegularizer,
    "flow": FlowRegularizer,
    "render_weight": RenderWeightRegularizer,
    "tensor_tv": TensorTV,
    "tensorf": TensoRF,
    "ray_density": RayDensityRegularizer,
    "simple_ray_density": SimpleRayDensityRegularizer,
}
