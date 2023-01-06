#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch

from .base import BaseRegularizer

from nlf.param import ray_param_dict, ray_param_pos_dict
from losses import loss_dict

import copy
from omegaconf import OmegaConf # @manual //github/third-party/omry/omegaconf:omegaconf


class WarpRegularizer(BaseRegularizer):
    def __init__(
        self,
        system,
        cfg
    ):
        super().__init__(system, cfg)

        # Setup parametrization
        param_cfg = copy.deepcopy(cfg.param)
        OmegaConf.set_struct(param_cfg, False)

        if system.is_subdivided:
            system_param_cfg = system.cfg.model.ray.param

            for key in system_param_cfg.keys():
                param_cfg.__dict__[key] = system_param_cfg[key]
                setattr(param_cfg, key, system_param_cfg[key])
        else:
            system_param_cfg = system.cfg.model.param

            for key in system_param_cfg.keys():
                param_cfg.__dict__[key] = system_param_cfg[key]
                setattr(param_cfg, key, system_param_cfg[key])

        self.ray_param_fn = ray_param_dict[param_cfg.fn](param_cfg)
        self.ray_param_pos_fn = ray_param_pos_dict[param_cfg.fn](param_cfg)
        self.param_channels = self.cfg.param.n_dims

        # Setup losses
        self.loss_fn = loss_dict[self.cfg.loss.type]()
        self.use_inp_freq = cfg.use_inp_freq

    def _loss(self, train_batch, batch_results, batch_idx):
        #### Prepare ####
        system = self.get_system()

        ## Batch
        batch = self.get_batch(train_batch, batch_idx)
        rays = batch['coords']

        ## tform constraints
        raw = system.render(
            'embed_params',
            rays
        )['value']
        out_channels = (raw.shape[-1] // (self.param_channels + 1))
        tform = raw[..., :out_channels].reshape(
            -1, out_channels, self.param_channels
        )

        _, S, _ = torch.svd(tform)

        loss = self.loss_fn(
            S[..., 2:],
            torch.zeros_like(S[..., 2:])
        )

        return loss


class WarpLevelSetRegularizer(BaseRegularizer):
    def __init__(
        self,
        system,
        cfg
    ):
        super().__init__(system, cfg)

        # Setup parametrization
        param_cfg = copy.deepcopy(cfg.param)
        OmegaConf.set_struct(param_cfg, False)

        if system.is_subdivided:
            system_param_cfg = system.cfg.model.ray.param

            for key in system_param_cfg.keys():
                param_cfg.__dict__[key] = system_param_cfg[key]
                setattr(param_cfg, key, system_param_cfg[key])
        else:
            system_param_cfg = system.cfg.model.param

            for key in system_param_cfg.keys():
                param_cfg.__dict__[key] = system_param_cfg[key]
                setattr(param_cfg, key, system_param_cfg[key])

        self.ray_param_fn = ray_param_dict[param_cfg.fn](param_cfg)
        self.ray_param_pos_fn = ray_param_pos_dict[param_cfg.fn](param_cfg)
        self.param_channels = self.cfg.param.n_dims

        # Setup losses
        self.svd_loss_fn = loss_dict[self.cfg.svd_loss.type]()
        self.level_loss_fn = loss_dict[self.cfg.level_loss.type]()
        self.use_inp_freq = cfg.use_inp_freq

    def _loss(self, train_batch, batch_results, batch_idx):
        #### Prepare ####
        system = self.get_system()

        ## Batch
        batch = self.get_batch(train_batch, batch_idx)
        rays = batch['coords']

        rgb_channels = 4 if system.is_subdivided else 3

        ## Get params
        outputs = system.render(
            'forward_all', rays, embed_params=True
        )
        params = outputs['value'][..., rgb_channels:].view(
            -1, outputs['value'].shape[-1] - rgb_channels
        )
        rgba = outputs['value'][..., :rgb_channels].view(
            -1, rgb_channels
        )

        out_channels = (params.shape[-1] // (self.param_channels + 1))
        tform = params[..., :-out_channels].reshape(
            -1, out_channels, self.param_channels
        )
        bias = params[..., -out_channels:]

        ## Decompose params
        U, S, V = torch.linalg.svd(tform)

        ## Jitter directions
        if not system.is_subdivided:
            param_rays = self.ray_param_fn(rays)
        else:
            num_slices = outputs['isect_inps'].shape[1]
            param_rays = self.ray_param_fn(
                outputs['isect_inps'][..., :6]
            )
            param_rays = param_rays.view(-1, param_rays.shape[-1])

        jitter = self.cfg.jitter

        jitter_dirs = torch.randn(
            (V.shape[0], jitter.bundle_size, V.shape[-2] - 2, 1), device=V.device
        ) * jitter.pos
        jitter_dirs = (jitter_dirs * V[..., 2:, :].unsqueeze(1)).mean(-2)
        jitter_dirs = jitter_dirs.view(
            -1, jitter_dirs.shape[-1]
        )

        ## Reshape all for comparison
        param_rays = param_rays.unsqueeze(1).repeat(
            1, jitter.bundle_size, 1
        ).view(-1, param_rays.shape[-1])

        rays = rays.unsqueeze(1).repeat(
            1, jitter.bundle_size, 1
        ).view(-1, rays.shape[-1])

        tform = tform.reshape(tform.shape[0], -1)
        tform = tform.unsqueeze(1).repeat(
            1, jitter.bundle_size, 1
        ).view(-1, tform.shape[-1])

        bias = bias.unsqueeze(1).repeat(
            1, jitter.bundle_size, 1
        ).view(-1, bias.shape[-1])

        ## Reshape for forward
        if system.is_subdivided:
            isect_codes = outputs['isect_inps'][..., 6:]
            isect_codes = isect_codes.view(-1, isect_codes.shape[-1])
            isect_codes = isect_codes.unsqueeze(1).repeat(
                1, jitter.bundle_size, 1
            ).view(-1, isect_codes.shape[-1])

            isect_mask = outputs['isect_mask']
            isect_mask = isect_mask.view(-1)
            isect_mask = isect_mask.unsqueeze(1).repeat(
                1, jitter.bundle_size
            ).view(-1)

        ## Forward
        jitter_rays = param_rays + jitter_dirs

        if not system.is_subdivided:
            jitter_outputs = system.render(
                'forward_all', jitter_rays, apply_ndc=False, no_param=True, embed_params=True
            )

            jitter_params = jitter_outputs['value'][..., rgb_channels:].view(
                -1, jitter_outputs['value'].shape[-1] - rgb_channels
            )
            jitter_rgba = jitter_outputs['value'][..., :rgb_channels].view(
                -1, rgb_channels
            )
        else:
            isect_inps = torch.cat(
                [
                    jitter_rays.view(
                        -1, num_slices, jitter.bundle_size, jitter_rays.shape[-1]
                    ),
                    isect_codes.view(
                        -1, num_slices, jitter.bundle_size, isect_codes.shape[-1]
                    )
                ],
                -1
            )
            isect_inps = isect_inps.permute(0, 2, 1, 3).reshape(
                -1, num_slices, isect_inps.shape[-1]
            )

            isect_mask = isect_mask.view(
                -1, num_slices, jitter.bundle_size
            )
            isect_mask = isect_mask.permute(0, 2, 1).reshape(
                -1, num_slices
            )

            jitter_outputs = system.render_fn.render(
                'forward_all',
                None,
                no_param=True,
                isect_inps=isect_inps,
                isect_mask=isect_mask,
                embed_params=True
            )

            jitter_params = jitter_outputs['value'][..., rgb_channels:].view(
                -1, jitter.bundle_size, num_slices, jitter_outputs['value'].shape[-1] - rgb_channels
            )
            jitter_params = jitter_params.permute(0, 2, 1, 3).reshape(
                -1, jitter_params.shape[-1]
            )

            jitter_rgba = jitter_outputs['value'][..., :rgb_channels].view(
                -1, jitter.bundle_size, num_slices, rgb_channels
            )
            jitter_rgba = jitter_rgba.permute(0, 2, 1, 3).reshape(
                -1, jitter_rgba.shape[-1]
            )

        jitter_tform = jitter_params[..., :-out_channels]
        jitter_bias = jitter_params[..., -out_channels:]

        #### Losses ####

        all_losses = {
            loss: 0.0 for loss in self.loss_fns.keys()
        }

        if self._do_loss('color_loss'):
            all_losses['color_loss'] += self._loss_fn(
                'color_loss',
                rgba,
                jitter_rgba
            )

        if self._do_loss('svd_loss'):
            all_losses['svd_loss'] += self._loss_fn(
                'svd_loss',
                S[..., 2:],
                torch.zeros_like(S[..., 2:])
            )

        if self._do_loss('level_loss'):
            all_losses['level_loss'] += self._loss_fn(
                'level_loss',
                jitter_tform,
                tform
            )

            all_losses['level_loss'] += self._loss_fn(
                'level_loss',
                jitter_bias,
                bias
        )

        ## Total loss
        total_loss = 0.0

        for name in all_losses.keys():
            if batch_idx == 0:
                print(name + ':', all_losses[name])

            total_loss += all_losses[name]

        return total_loss
