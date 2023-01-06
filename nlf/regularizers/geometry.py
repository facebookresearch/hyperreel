#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn

from .base import BaseRegularizer
from nlf.contract import contract_dict

from utils.ray_utils import (
    dot,
    from_ndc,
    reflect
)


class GeometryRegularizer(BaseRegularizer):
    def __init__(
        self,
        system,
        cfg
    ):
        super().__init__(system, cfg)

        # Setup losses
        self.cfg = cfg

        # Variables
        self.fields = list(cfg.fields)

        # Origin
        self.origin = torch.tensor(cfg.origin if 'origin' in cfg else [0.0, 0.0, 0.0], device='cuda')

        # Contract function
        if 'contract' in cfg:
            self.contract_fn = contract_dict[cfg.contract.type](
                cfg.contract, system=system
            )
        else:
            self.contract_fn = contract_dict['identity']({})

        # How many points to use
        self.num_points = cfg.num_points if 'num_points' in cfg else -1

    def _loss(self, batch, outputs, batch_idx):
        # Get coords
        rays = batch['coords']
        rays = torch.clone(rays.view(-1, rays.shape[-1]))

        # Render points
        pred_points = outputs[self.fields[0]]
        pred_points = pred_points.view(pred_points.shape[0], -1, 3)

        pred_distance = outputs[self.fields[1]]
        pred_distance = pred_distance.view(pred_points.shape[0], -1)

        # Get ground truth points
        gt_depth = batch['depth']

        rays_o, rays_d = rays[..., :3] - self.origin[None], rays[..., 3:6]
        rays_d = torch.nn.functional.normalize(rays_d, p=2.0, dim=-1)
        gt_points = self.contract_fn.contract_points(rays_o + gt_depth * rays_d)

        # Compute mask
        mask = (gt_depth != 0.0) \
            & (pred_distance != 0.0)

        # Loss
        diff = torch.norm(pred_points - gt_points.unsqueeze(1), dim=-1) * mask.float()

        if self.num_points > 0:
            diff = torch.sort(diff, dim=-1)[0][..., :self.num_points]

        loss = torch.mean(diff)
        return loss

    @property
    def render_kwargs(self):
        return {
            'fields': self.fields,
            'no_over_fields': self.fields,
        }


class GeometryFeedbackRegularizer(BaseRegularizer):
    def __init__(
        self,
        system,
        cfg
    ):
        super().__init__(system, cfg)

        # Setup losses
        self.cfg = cfg

        # Variables
        self.student_fields = list(cfg.student_fields)
        self.teacher_fields = list(cfg.teacher_fields)
        self.sizes = list(cfg.sizes) if 'sizes' in cfg else [3 for s in self.student_fields]
        self.weights = list(cfg.weights) if 'weights' in cfg else [1.0 for s in self.student_fields]

        # How many points to use
        self.num_points = cfg.num_points if 'num_points' in cfg else -1

    def _loss(self, batch, outputs, batch_idx):
        # Get coords
        rays = batch['coords']
        rays = torch.clone(rays.view(-1, rays.shape[-1]))

        # Weights
        render_weights = outputs['render_weights']

        # Total loss
        total_loss = 0.0

        for idx, loss_weight in enumerate(self.weights):
            # Student outputs
            student_points = outputs[self.student_fields[idx]]
            student_points = student_points.view(student_points.shape[0], -1, 1, self.sizes[-1])

            # Teacher outputs
            teacher_points = outputs[self.teacher_fields[idx]]

            if not self.teacher_fields[idx] == 'render_normal':
                teacher_points = teacher_points.detach()
            else:
                pass

            teacher_points = teacher_points.view(teacher_points.shape[0], student_points.shape[1], -1, self.sizes[-1])

            # Loss
            if not self.teacher_fields[idx] == 'render_normal':
                cur_render_weights = render_weights.detach()
            else:
                cur_render_weights = render_weights.detach()
                #cur_render_weights = render_weights

            cur_render_weights = cur_render_weights.view(cur_render_weights.shape[0], student_points.shape[1], -1)

            # Special case normal
            if self.teacher_fields[idx] == 'render_normal':
                viewdirs = outputs['viewdirs']

                diff = 1.0 - dot(
                    student_points,
                    teacher_points
                )
                loss_match = (
                    diff * cur_render_weights
                ).sum((-2, -1)).mean()

                dot_dirs_normal = dot(
                    student_points.view(student_points.shape[0], -1, 3),
                    viewdirs.view(student_points.shape[0], -1, 3),
                    keepdim=True
                )
                loss_penalty = (
                    torch.square(
                        torch.maximum(dot_dirs_normal, torch.zeros_like(dot_dirs_normal))
                    ) * cur_render_weights
                ).sum((-2, -1)).mean()

                loss = loss_match * loss_weight[0] + loss_penalty * loss_weight[1]
            else:
                diff = torch.square(student_points - teacher_points).sum(-1)
                #diff = torch.norm(student_points - teacher_points, dim=-1)
                #diff = torch.abs(student_points - teacher_points).sum(-1)
                diff = (diff * cur_render_weights).sum((-2, -1))

                loss = torch.mean(diff) * loss_weight

            total_loss = total_loss + loss

        return total_loss

    @property
    def render_kwargs(self):
        return {
            'fields': self.student_fields + self.teacher_fields + ['render_weights', 'viewdirs'],
            'no_over_fields': self.student_fields + self.teacher_fields + ['render_weights', 'viewdirs'],
        }


class FlowRegularizer(BaseRegularizer):
    def __init__(
        self,
        system,
        cfg
    ):
        super().__init__(system, cfg)

        # Setup losses
        self.cfg = cfg

        # Variables
        self.fields = list(cfg.fields)

        # Origin
        self.origin = torch.tensor(cfg.origin if 'origin' in cfg else [0.0, 0.0, 0.0], device='cuda')

        # Contract function
        if 'contract' in cfg:
            self.contract_fn = contract_dict[cfg.contract.type](
                cfg.contract
            )
        else:
            self.contract_fn = contract_dict['identity']({})

        # How many points to use
        self.num_points = cfg.num_points if 'num_points' in cfg else -1

    def _loss(self, batch, outputs, batch_idx):
        # Get coords
        rays = batch['coords']
        rays = torch.clone(rays.view(-1, rays.shape[-1]))

        # Render points and distance
        pred_points_start = outputs[self.fields[0]]
        pred_points_start = pred_points_start.view(pred_points_start.shape[0], -1, 3)

        pred_points_end = outputs[self.fields[1]]
        pred_points_end = pred_points_end.view(pred_points_end.shape[0], -1, 3)
        pred_points = torch.cat([pred_points_start, pred_points_end], -1)

        pred_distance = outputs[self.fields[2]]
        pred_distance = pred_distance.view(pred_points_start.shape[0], -1)

        # Get ground truth flow
        gt_flow = batch['flow']
        gt_depth = batch['depth']

        rays_o, rays_d = rays[..., :3] - self.origin[None], rays[..., 3:6]
        rays_d = torch.nn.functional.normalize(rays_d, p=2.0, dim=-1)
        gt_world_points = rays_o + gt_depth * rays_d

        gt_points_start = self.contract_fn.contract_points(gt_world_points)
        gt_points_end = self.contract_fn.contract_points(gt_world_points + gt_flow)
        gt_points = torch.cat([gt_points_start, gt_points_end], -1)

        # Compute mask
        mask = (gt_flow != 0.0).any(dim=-1, keepdim=True) \
            & (gt_depth != 0.0) \
            & (pred_distance != 0.0)

        # Loss
        diff = torch.norm(pred_points - gt_points.unsqueeze(1), dim=-1) * mask.float()

        if self.num_points > 0:
            diff = torch.sort(diff, dim=-1)[0][..., :self.num_points]

        loss = torch.mean(diff)
        return loss

    @property
    def render_kwargs(self):
        return {
            'fields': self.fields,
            'no_over_fields': self.fields,
            'no_flow_jitter': True
        }


class RenderWeightRegularizer(BaseRegularizer):
    def __init__(
        self,
        system,
        cfg
    ):
        super().__init__(system, cfg)

        # Setup losses
        self.cfg = cfg

        # How many points to use
        self.num_points = cfg.num_points if 'num_points' in cfg else -1

        # Ease iterations
        self.window_iters = cfg.window_iters

    def ease_weight(self):
        return min(max(self.cur_iter / float(self.window_iters), 0.0), 1.0)

    def _loss(self, batch, batch_results, batch_idx):
        weights = batch_results['weights']
        render_weights = batch_results['render_weights'].view(*weights.shape).detach()

        w = self.ease_weight()

        if True:
            sparsity_loss_0 = torch.mean(torch.abs(weights))
            sparsity_loss_1 = torch.mean(torch.abs(1.0 - weights)) * 0.1
            match_loss = torch.mean(torch.abs(weights - render_weights))
            #return (sparsity_loss_0 + match_loss) * w + sparsity_loss_1 * (1 - w)
            return match_loss * w + sparsity_loss_1 * (1 - w)
        else:
            entropy_loss = torch.mean(-render_weights * torch.log(render_weights + 1e-8))
            return entropy_loss * w

    @property
    def render_kwargs(self):
        return {
            'fields': ['weights', 'render_weights'],
            'no_over_fields': ['weights'],
        }
