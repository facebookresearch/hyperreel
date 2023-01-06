#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn


class HuberLoss(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.loss = nn.HuberLoss(reduction='mean', delta=cfg.delta if 'delta' in cfg else 1.0)

    def forward(self, inputs, targets, **kwargs):
        loss = self.loss(inputs, targets)
        return loss


class MSELoss(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.loss = nn.MSELoss(reduction='mean')

    def forward(self, inputs, targets, **kwargs):
        loss = self.loss(inputs, targets)
        return loss


class HuberLoss(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.loss = nn.HuberLoss(reduction='mean', delta=cfg.delta)

    def forward(self, inputs, targets, **kwargs):
        loss = self.loss(inputs, targets)
        return loss


class WeightedMSELoss(nn.Module):
    def __init__(self, *args):
        super().__init__()

    def forward(self, inputs, targets, **kwargs):
        if 'weight' in kwargs:
            weight = kwargs['weight']
        else:
            weight = 1.0

        return torch.mean(weight * torch.square(inputs - targets))


class MAELoss(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.loss = nn.L1Loss(reduction='mean')

    def forward(self, inputs, targets, **kwargs):
        loss = self.loss(inputs, targets)
        return loss


class WeightedMAELoss(nn.Module):
    def __init__(self, *args):
        super().__init__()

    def forward(self, inputs, targets, **kwargs):
        if 'weight' in kwargs:
            weight = kwargs['weight']
        else:
            weight = 1.0

        return torch.mean(weight * torch.abs(inputs - targets))


class TVLoss(nn.Module):
    def __init__(self, *args):
        super().__init__()

    def forward(self, inputs, targets):
        return torch.sqrt(torch.square(inputs - targets).sum(-1) + 1e-8).mean()


class ComplexMSELoss(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.loss = nn.MSELoss(reduction='mean')

    def forward(self, inputs, targets):
        loss = self.loss(torch.real(inputs), torch.real(targets))
        loss += self.loss(torch.imag(inputs), torch.imag(targets))
        return loss


class ComplexMAELoss(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.loss = nn.L1Loss(reduction='mean')

    def forward(self, inputs, targets):
        loss = self.loss(torch.real(inputs), torch.real(targets))
        loss += self.loss(torch.imag(inputs), torch.imag(targets))
        return loss


class MSETopN(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.frac = cfg.frac
        self.loss = nn.MSELoss(reduction='mean')

    def forward(self, inputs, targets):
        diff = torch.abs(inputs - targets)
        n = int(self.frac * targets.shape[0])

        idx = torch.argsort(diff, dim=0)

        targets_sorted = torch.gather(targets, 0, idx)
        targets_sorted = targets_sorted[:n]

        inputs_sorted = torch.gather(inputs, 0, idx)
        inputs_sorted = inputs_sorted[:n]

        loss = self.loss(inputs_sorted, targets_sorted)
        return loss


class MAETopN(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.frac = cfg.frac
        self.loss = nn.L1Loss(reduction='mean')

    def forward(self, inputs, targets):
        diff = torch.abs(inputs - targets)
        n = int(self.frac * targets.shape[0])

        idx = torch.argsort(diff, dim=0)

        targets_sorted = torch.gather(targets, 0, idx)
        targets_sorted = targets_sorted[:n]

        inputs_sorted = torch.gather(inputs, 0, idx)
        inputs_sorted = inputs_sorted[:n]

        loss = self.loss(inputs_sorted, targets_sorted)
        return loss


loss_dict = {
    'huber': HuberLoss,
    'mse': MSELoss,
    'weighted_mse': WeightedMSELoss,
    'mae': MAELoss,
    'weighted_mae': WeightedMAELoss,
    'tv': TVLoss,
    'complex_mse': ComplexMSELoss,
    'complex_mae': ComplexMAELoss,
    'mse_top_n': MSETopN,
    'mae_top_n': MAETopN,
}
