#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import torch
from omegaconf import DictConfig, OmegaConf

from kornia.filters import gaussian_blur2d

from .base import BaseRegularizer
from losses import loss_dict

from utils.tensorf_utils import AlphaGridMask
from nlf.models import model_dict
from nlf.rendering import (
    render_chunked,
    render_fn_dict
)


class TeacherRegularizer(BaseRegularizer):
    def __init__(
        self,
        system,
        cfg
    ):
        super().__init__(system, cfg)

        self.loss_fn = loss_dict[self.cfg.loss.type]()
        self.use_inp_freq = 'inf'

    def _loss(self, train_batch, batch_results, batch_idx):
        system = self.get_system()

        if self.cur_iter >= self.cfg.weight.stop_iters:
            return 0.0

        # Get inputs
        dataset = self.get_dataset()
        batch = dataset.get_batch(batch_idx, self.batch_size)

        rays = batch['coords'].type_as(train_batch['coords'])
        rgb = batch['rgb'].type_as(train_batch['rgb'])

        # Loss
        results = system(rays)
        pred_rgb = results['rgb']

        loss = self.loss_fn(
            pred_rgb,
            rgb
        )

        return loss


class BlurryTeacherRegularizer(BaseRegularizer):
    def __init__(
        self,
        system,
        cfg
    ):
        super().__init__(system, cfg)

        self.loss_fn = loss_dict[self.cfg.loss.type]()
        self.use_inp_freq = 'inf'

        self.patch_width = self.cfg.dataset.patch_width
        self.blur_radius = self.cfg.blur_radius
        self.batch_size = self.patch_width * self.patch_width

    def _loss(self, train_batch, batch_idx):
        system = self.get_system()

        if self.cur_iter >= self.cfg.weight.stop_iters:
            return 0.0

        # Get inputs
        dataset = self.get_dataset()
        batch = dataset.get_batch(batch_idx, self.batch_size)

        rays = batch['coords'].type_as(train_batch['coords'])
        rgb = batch['rgb'].type_as(train_batch['rgb'])

        # Run forward and blur
        pred_rgb = system(rays)['rgb']
        pred_rgb = pred_rgb.view(-1, self.patch_width, self.patch_width, 3).permute(0, 3, 1, 2)
        rgb = rgb.view(-1, self.patch_width, self.patch_width, 3).permute(0, 3, 1, 2)

        if self.blur_radius > 0:
            blur_rgb = gaussian_blur2d(
                pred_rgb,
                (self.blur_radius * 2 + 1, self.blur_radius * 2 + 1),
                (self.blur_radius / 3.0, self.blur_radius / 3.0)
            )
            blur_rgb = blur_rgb[
                ...,
                self.blur_radius:-self.blur_radius,
                self.blur_radius:-self.blur_radius
            ]
            rgb = rgb[
                ...,
                self.blur_radius:-self.blur_radius,
                self.blur_radius:-self.blur_radius
            ]
        else:
            blur_rgb = pred_rgb

        # Loss
        return self.loss_fn(
            blur_rgb,
            rgb
        )


class TeacherModelRegularizer(BaseRegularizer):
    def __init__(
        self,
        system,
        cfg
    ):
        super().__init__(system, cfg)

        self.cfg = cfg

        self.model_ckpt_dir = os.path.expanduser(system.cfg.params.ckpt_dir)
        self.model_ckpt_path = os.path.join(self.model_ckpt_dir, cfg.model_ckpt_path)

        # Create model
        self.model_config = self.cfg.model
        self.model_start_epoch = cfg.model_start_epoch

        model = model_dict[self.model_config.type](
            self.model_config, system=system
        )

        # Load from checkpoint
        model = model.cuda().eval()
        model_state_dict = torch.load(self.model_ckpt_path)['state_dict']
        self.load_state_dict_for_model(model, model_state_dict)

        # Set iteration
        model.set_iter(system.cfg.training.iters_per_epoch * self.model_start_epoch)

        # List of models (do not save in subsequent checkpoints)
        self.models = [model]

        # TODO: 
        #   - incorporate dataset info
        #   - better ray generation
        #   - some debugging

        # Random ray generation
        self.origin_range = torch.tensor(cfg.origin_range if 'origin_range' in cfg else [[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]]).cuda()
        self.direction_range = torch.tensor(cfg.direction_range if 'direction_range' in cfg else [[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]]).cuda()
        self.extra_range = torch.tensor(cfg.extra_range if 'extra_range' in cfg else [[0.0], [0.0]]).cuda()

        self.use_ndc = cfg.use_ndc if 'use_ndc' in cfg else False
        self.convert_ndc = cfg.convert_ndc if 'convert_ndc' in cfg else False
    
    def generate_random_rays(self, coords):
        batch_size = coords.shape[0]

        origins = torch.rand((batch_size, 3), device=coords.device) \
            * (self.origin_range[1:2] - self.origin_range[0:1]) \
            + self.origin_range[0:1]

        directions = torch.rand((batch_size, 3), device=coords.device) \
            * (self.direction_range[1:2] - self.direction_range[0:1]) \
            + self.direction_range[0:1]
        
        if self.use_ndc:
            #directions = (directions / directions[..., -1:]) * 2.0
            directions = torch.nn.functional.normalize(directions, dim=-1)
        else:
            directions = torch.nn.functional.normalize(directions, dim=-1)

        extras = torch.rand((batch_size, self.extra_range.shape[-1]), device=coords.device) \
            * (self.extra_range[1:2] - self.extra_range[0:1]) \
            + self.extra_range[0:1]

        return torch.cat([origins, directions, extras], dim=-1)

    def generate_random_rays_convex(self, coords):
        batch_size = coords.shape[0]

        # Extras
        rays = coords[..., :6]
        extras = coords[..., 6:]

        # Collect
        num_convex = 4

        rand_idx = torch.randint(
            low=0,
            high=batch_size,
            size=(batch_size * (num_convex - 1), 1),
            device=rays.device
        ).repeat(1, 6)

        rand_rays = torch.gather(rays, 0, rand_idx)
        rand_rays = rand_rays.reshape(batch_size, num_convex - 1, 6)
        rand_rays = torch.cat([rays.unsqueeze(1), rand_rays], dim=1)

        # Convex combination
        weights = torch.rand(
            (rays.shape[0], num_convex), device=rays.device
        )
        weights = weights / (weights.sum(1).unsqueeze(1) + 1e-8)

        # Valid rays
        rays = (rays.unsqueeze(1) * weights.unsqueeze(-1)).sum(1)
        rays_o = rays[..., 0:3]
        rays_d = rays[..., 3:6]

        # NOTE: normalization affects distances in over-composite
        if self.use_ndc:
            rays_d = (rays_d / rays_d[..., -1:]) * 2.0
        else:
            rays_d = torch.nn.functional.normalize(rays_d, dim=-1)

        return torch.cat([rays_o, rays_d, extras], -1)

    def _loss(self, batch, outputs, batch_idx):
        system = self.get_system()

        # Teacher results
        with torch.no_grad():
            # Generate random rays
            coords = batch['coords']

            #new_coords = self.generate_random_rays(coords)
            new_coords = self.generate_random_rays_convex(coords)
            #new_coords = coords
            
            # Run forward
            teacher_results = self.models[0](new_coords, {})

        # Model results
        model_results = system(new_coords, **system.regularizer_render_kwargs)
        weight = (teacher_results['rgb'] != 0.0).any(dim=-1, keepdim=True)
        weight = torch.ones_like(weight)

        # Loss
        image_loss = system.loss(model_results['rgb'] * weight, teacher_results['rgb'] * weight, **batch)
        return image_loss

    def load_state_dict_for_model(self, model, state_dict, strict=False):
        new_state_dict = {}

        # For loading subdivision variables (voxel grid, voxel size, etc.) #
        alpha_aabb = None
        alpha_volume = None

        for key in state_dict.keys():
            new_key = key.split('render_fn.model.')[-1]
            new_state_dict[new_key] = state_dict[key]
            
            # Update size of tensor components
            if 'alpha_aabb' in key:
                alpha_aabb = state_dict[key]
            elif 'alpha_volume' in key:
                alpha_volume = state_dict[key]
            elif 'gridSize' in key:
                model.color_model.net.gridSize = state_dict[key]

                model.color_model.net.init_svd_volume(
                    model.color_model.net.gridSize[0],
                    model.color_model.net.device
                )

        model.load_state_dict(new_state_dict, strict=False)

        # Update other grid-size-dependent variables
        model.color_model.net.update_stepSize(
            model.color_model.net.gridSize.cpu()
        )

        # Update alpha mask
        if alpha_volume is not None:
            device = model.color_model.net.device
            model.color_model.net.alphaMask = AlphaGridMask(
                device,
                alpha_aabb.to(device),
                alpha_volume.to(device)
            )