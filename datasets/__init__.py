#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from .blender import BlenderDataset, BlenderLightfieldDataset, DenseBlenderDataset
from .donerf import DONeRFDataset
from .fourier import FourierDataset, FourierLightfieldDataset
from .llff import DenseLLFFDataset, LLFFDataset
from .eikonal import EikonalDataset
from .random import (
    RandomPixelDataset,
    RandomRayDataset,
    RandomRayLightfieldDataset,
    RandomViewSubsetDataset,
)
from .shiny import DenseShinyDataset, ShinyDataset
from .stanford import StanfordEPIDataset, StanfordLightfieldDataset, StanfordLLFFDataset
from .video3d_static import Video3DDataset
from .video3d_time import Video3DTimeDataset
from .video3d_ground_truth import Video3DTimeGroundTruthDataset
from .technicolor import TechnicolorDataset
from .neural_3d import Neural3DVideoDataset
from .catacaustics import CatacausticsDataset
from .immersive import ImmersiveDataset
from .spaces import SpacesDataset

dataset_dict = {
    "fourier": FourierDataset,
    "fourier_lightfield": FourierLightfieldDataset,
    "random_ray": RandomRayDataset,
    "random_pixel": RandomPixelDataset,
    "random_lightfield": RandomRayLightfieldDataset,
    "random_view": RandomViewSubsetDataset,
    "donerf": DONeRFDataset,
    "blender": BlenderDataset,
    "dense_blender": DenseBlenderDataset,
    "llff": LLFFDataset,
    "eikonal": EikonalDataset,
    "dense_llff": DenseLLFFDataset,
    "dense_shiny": DenseShinyDataset,
    "shiny": ShinyDataset,
    "blender_lightfield": BlenderLightfieldDataset,
    "stanford": StanfordLightfieldDataset,
    "stanford_llff": StanfordLLFFDataset,
    "stanford_epi": StanfordEPIDataset,
    "video3d": Video3DDataset,
    "video3d_time": Video3DTimeDataset,
    "video3d_time_ground_truth": Video3DTimeGroundTruthDataset,
    "technicolor": TechnicolorDataset,
    "neural_3d": Neural3DVideoDataset,
    "catacaustics": CatacausticsDataset,
    "immersive": ImmersiveDataset,
    "spaces": SpacesDataset,
}
