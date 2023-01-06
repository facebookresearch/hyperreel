#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
from torch import nn
from nlf.activations import (
    get_activation,
    Tanh,
    IdentityTanh,
    Power
)


class BaseContract(nn.Module):
    def __init__(
        self,
        cfg,
        **kwargs
    ):
        super().__init__()

        self.use_dataset_bounds = cfg.use_dataset_bounds if 'use_dataset_bounds' in cfg else False
        self.contract_samples = cfg.contract_samples if 'contract_samples' in cfg else False

    def inverse_contract_distance(self, distance):
        return distance

    def contract_distance(self, distance):
        return distance

    def contract_points(self, points):
        return points

    def inverse_contract_points(self, contract_points):
        contract_distance = torch.norm(contract_points, dim=-1, keepdim=True)
        distance = self.inverse_contract_distance(contract_distance)
        return (contract_points / contract_distance) * distance

    def contract_points_and_distance(self, rays_o, points, distance):
        # Contract
        rays_o = self.contract_points(rays_o)
        points = self.contract_points(points)
        distance = torch.norm(points - rays_o[..., None, :], dim=-1, keepdim=True)

        # Return
        return points, distance


class IdentityContract(BaseContract):
    def __init__(
        self,
        cfg,
        **kwargs
    ):
        super().__init__(cfg)

    def contract_points_and_distance(self, rays_o, points, distance):
        return points, distance


class BBoxContract(BaseContract):
    def __init__(
        self,
        cfg,
        **kwargs
    ):
        super().__init__(cfg)

        self.bbox_min = torch.tensor(list(cfg.bbox_min) if 'bbox_min' in cfg else [-1.0, -1.0, -1.0]).cuda()
        self.bbox_max = torch.tensor(list(cfg.bbox_max) if 'bbox_max' in cfg else [1.0, 1.0, 1.0]).cuda()
        self.fac = torch.mean(torch.abs(self.bbox_max - self.bbox_min))

    def inverse_contract_distance(self, distance):
        return distance * self.fac

    def contract_distance(self, distance):
        return distance / self.fac

    def contract_points(self, points):
        return (points - self.bbox_min.view(1, 1, 3)) / (self.bbox_max.view(1, 1, 3) - self.bbox_min.view(1, 1, 3))


class ZDepthContract(BaseContract):
    def __init__(
        self,
        cfg,
        **kwargs
    ):
        super().__init__(cfg)

        if self.use_dataset_bounds:
            self.contract_end_radius = cfg.contract_end_radius if 'contract_end_radius' in cfg else \
                kwargs['system'].dm.train_dataset.depth_range[1]
        else:
            self.contract_end_radius = cfg.contract_end_radius if 'contract_end_radius' in cfg else float("inf")

        self.fac = self.contract_end_radius / 2.0

    def inverse_contract_distance(self, distance):
        return distance * self.fac

    def contract_distance(self, distance):
        return distance / self.fac

    def contract_points(self, points):
        return points / self.fac


class MIPNeRFContract(BaseContract):
    def __init__(
        self,
        cfg,
        **kwargs
    ):
        super().__init__(cfg)

        if self.use_dataset_bounds:
            self.contract_start_radius = cfg.contract_start_radius if 'contract_start_radius' in cfg else \
                max(kwargs['system'].dm.train_dataset.depth_range[0] * 1.5, 1.0)
            self.contract_end_radius = cfg.contract_end_radius if 'contract_end_radius' in cfg else \
                kwargs['system'].dm.train_dataset.depth_range[1] * 1.5
        else:
            self.contract_start_radius = cfg.contract_start_radius if 'contract_start_radius' in cfg else 1.0
            self.contract_end_radius = cfg.contract_end_radius if 'contract_end_radius' in cfg else float("inf")

        self.contract_start_distance = cfg.contract_start_distance if 'contract_start_distance' in cfg else self.contract_start_radius
        self.contract_end_distance = cfg.contract_end_distance if 'contract_end_distance' in cfg else self.contract_end_radius

        if 'distance_activation' in cfg:
            self.distance_activation = get_activation(
                cfg.distance_activation
            )
        else:
            #self.distance_activation = IdentityTanh({})
            #self.distance_activation = Tanh({'fac': 1.0})
            #self.distance_activation = Tanh({'fac': 2.0})
            self.distance_activation = get_activation('identity')

    def inverse_contract_distance(self, distance):
        # t varies linearly in disparity
        inverse_contract_end_distance = self.contract_start_distance / self.contract_end_distance
        scale_factor = 1.0 / (1.0 - inverse_contract_end_distance)

        # Inverse distance
        distance = self.distance_activation(distance / 2.0) * 2.0
        distance = distance.clamp(-2.0, 2.0)
        t = (2.0 - torch.abs(distance))
        inverse_distance = t / scale_factor + inverse_contract_end_distance

        return torch.where(
            torch.abs(distance) < 1,
            distance,
            torch.sign(distance) * ( 1.0 / inverse_distance )
        ) * self.contract_start_distance

    def contract_distance(self, distance):
        # Re-scale distance
        distance = distance / self.contract_start_distance
        inverse_distance = 1.0 / torch.abs(distance)

        # t varies linearly in disparity
        inverse_contract_end_distance = self.contract_start_distance / self.contract_end_distance
        scale_factor = 1.0 / (1.0 - inverse_contract_end_distance)
        t = (inverse_distance - inverse_contract_end_distance) * scale_factor

        distance = torch.where(
            torch.abs(distance) < 1.0,
            distance / 1.0,
            torch.sign(distance) * ( 2.0 - t ),
        )

        return self.distance_activation.inverse(distance / 2.0) * 2.0

    def contract_points(self, points):
        points = points / self.contract_start_radius
        distance = torch.norm(points, dim=-1, keepdim=True)

        # t varies linearly in disparity
        inverse_distance = 1.0 / torch.abs(distance)
        inverse_contract_end_radius = self.contract_start_radius / self.contract_end_radius
        scale_factor = 1.0 / (1.0 - inverse_contract_end_radius)
        t = (inverse_distance - inverse_contract_end_radius) * scale_factor

        return torch.where(
            distance < 1,
            points,
            (points / distance) * ( 2.0 - t )
        )


class DoNeRFContract(BaseContract):
    def __init__(
        self,
        cfg,
        **kwargs
    ):
        super().__init__(cfg)

        if self.use_dataset_bounds:
            self.contract_start_radius = cfg.contract_start_radius if 'contract_start_radius' in cfg else \
                max(kwargs['system'].dm.train_dataset.depth_range[0] * 1.75, 1.0)
            self.contract_end_radius = cfg.contract_end_radius if 'contract_end_radius' in cfg else \
                kwargs['system'].dm.train_dataset.depth_range[1] * 1.5
        else:
            self.contract_start_radius = cfg.contract_start_radius if 'contract_start_radius' in cfg else None
            self.contract_end_radius = cfg.contract_end_radius if 'contract_end_radius' in cfg else 10000.0

        if self.contract_start_radius is None:
            self.power = cfg.power if 'power' in cfg else 2.0
            self.fac = np.power(2.0, self.power) / self.contract_end_radius
        else:
            self.fac = 1.0 / self.contract_start_radius
            self.power = np.log(self.contract_end_radius / self.contract_start_radius) / np.log(2.0)

        if 'distance_activation' in cfg:
            self.distance_activation = get_activation(
                cfg.distance_activation
            )
        else:
            self.distance_activation = get_activation('identity')

    def inverse_contract_distance(self, distance):
        distance = self.distance_activation(distance / 2.0) * 2.0
        distance = distance.clamp(-2.0, 2.0)

        return torch.pow(torch.abs(distance) + 1e-8, self.power) * torch.sign(distance) / self.fac

    def contract_distance(self, distance):
        distance = distance * self.fac
        distance = torch.pow(torch.abs(distance) + 1e-8, 1.0 / self.power) * torch.sign(distance)
        
        return self.distance_activation.inverse(distance / 2.0) * 2.0

    def contract_points(self, points):
        dists = torch.norm(points, dim=-1, keepdim=True)
        return (points / dists) * torch.pow(dists * self.fac + 1e-8, 1.0 / self.power)


contract_dict = {
    'identity': IdentityContract,
    'bbox': BBoxContract,
    'z_depth': ZDepthContract,
    'mipnerf': MIPNeRFContract,
    'donerf': DoNeRFContract,
}
