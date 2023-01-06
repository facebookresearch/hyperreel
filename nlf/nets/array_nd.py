#!/usr/bin/env python
#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the # LICENSE file in the root directory of this source tree.

import numpy as np
import torch
from torch import nn

from nlf.activations import get_activation


def my_index(x, input_channels):
    cur_x = []

    for ch  in input_channels:
        cur_x.append(x[..., ch])

    cur_x = torch.stack(cur_x, -1)
    return cur_x


class ArrayND(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        cfg,
        **kwargs
    ):
        super().__init__()

        self.opt_group = cfg.group if 'group' in cfg else (kwargs['group'] if 'group' in kwargs else 'color')

        # Number of input, output channels
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Input channels for each tensor
        self.input_channels = list(cfg.input_channels)

        # Size and range
        self.size = cfg.size[::-1]

        if 'range' in cfg:
            self.range = cfg.range
        else:
            self.range = [[-1, 1] for s in self.size]

        if len(self.size) == 1:
            self.size = [1] + self.size

        self.min_range = [r[0] for r in self.range]
        self.max_range = [r[1] for r in self.range]
        self.mode = cfg.mode if 'mode' in cfg else 'bilinear'
        self.padding_mode = cfg.padding_mode if 'padding_mode' in cfg else 'zeros'

        # Activation
        self.activation = cfg.activation if 'activation' in cfg else 'identity'
        self.out_layer = get_activation(self.activation)

        # Tensor setup
        cur_size = (self.out_channels,) + tuple(self.size)

        self.tensor = nn.Parameter(torch.Tensor(*cur_size))
        if self.opt_group == 'color':
            self.tensor.data.uniform_(-1.0, 1.0)
        else:
            self.tensor.data.uniform_(-0.01, 0.01)

        #self.tensor = nn.Parameter(1.0 * torch.randn(cur_size, device='cuda'))

    def forward(self, x, **render_kwargs):
        # Index
        x = my_index(x, self.input_channels)

        # Ranges
        range_shape = tuple(1 for s in x.shape[:-1]) + x.shape[-1:]
        min_range = torch.tensor(self.min_range, device=x.device).float().view(range_shape)
        max_range = torch.tensor(self.max_range, device=x.device).float().view(range_shape)

        # Normalize
        x = ((x - min_range) / (max_range - min_range)) * 2 - 1

        # Get mask
        mask = ~torch.any((x < -1) | (x > 1), dim=-1, keepdim=True)

        # Append extra dimension
        if x.shape[-1] == 1:
            x = torch.cat([x, -torch.ones_like(x)], -1)

        # Mask
        #all_feature = x.new_zeros(x.shape[0], self.out_channels)
        #x = x[mask.repeat(1, x.shape[-1])].view(-1, x.shape[-1])
        x = torch.where(mask, x, 1e8 * torch.ones_like(x))

        # Reshape
        input_shape = (1,) + x.shape[:1] + tuple(1 for s in self.size[:-1]) + x.shape[-1:]
        x = x.view(input_shape)

        # Sample feature
        feature = nn.functional.grid_sample(
            self.tensor.unsqueeze(0), x, mode=self.mode, padding_mode=self.padding_mode, align_corners=False
        )

        # Reshape
        feature = feature.reshape(
            self.out_channels, feature.shape[2]
        ).permute(1, 0)

        ## All feature
        #all_feature[mask.repeat(1, self.out_channels)] = feature.reshape(-1)
        #feature = all_feature.view(-1, self.out_channels)
        return feature

    def set_iter(self, i):
        pass



class ArrayNDMultiple(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        cfg,
        **kwargs
    ):
        super().__init__()

        self.opt_group = cfg.group if 'group' in cfg else (kwargs['group'] if 'group' in kwargs else 'color')

        # Number of input, output channels
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_factors = cfg.num_factors if 'num_factors' in cfg else 1

        # Input channels for each tensor
        self.input_channels = list(cfg.input_channels)
        self.dims_per_factor = len(cfg.size)

        # Size and range
        self.size = cfg.size[::-1]
        self.size[0] = self.size[0] * self.num_factors

        self.range = cfg.range
        self.range = np.array(self.range).reshape(
            self.num_factors, self.dims_per_factor, 2
        )

        if self.dims_per_factor == 1:
            self.size = [1] + self.size
            self.range = np.concatenate(
                [
                    -np.ones_like(self.range[:, :1, :]),
                    self.range,
                ],
                axis=1
            )

        self.width = self.size[1]
        self.height = self.size[0] // self.num_factors
        self.height_factor = (self.height - 1) / (self.size[0] - 1)

        self.min_range = self.range[..., 0]
        self.max_range = self.range[..., 1]

        # Product mode
        self.product_mode = cfg.product_mode if 'product_mode' in cfg else 'product'

        # Grid sample mode
        self.mode = cfg.mode if 'mode' in cfg else 'bilinear'
        self.padding_mode = cfg.padding_mode if 'padding_mode' in cfg else 'zeros'

        # Activation
        self.activation = cfg.activation if 'activation' in cfg else 'identity'
        self.out_layer = get_activation(self.activation)

        # Create tensors
        cur_size = (self.out_channels,) + tuple(self.size)

        self.tensor = nn.Parameter(torch.Tensor(*cur_size))
        if self.opt_group == 'color':
            self.tensor.data.uniform_(-1.0, 1.0)
        else:
            self.tensor.data.uniform_(-0.01, 0.01)

        #self.tensor = nn.Parameter(0.1 * torch.randn(cur_size, device='cuda'))

    def forward(self, x, **render_kwargs):
        batch_size = x.shape[0]

        # Index
        x = my_index(x, self.input_channels)

        # Ranges
        range_shape = tuple(1 for s in x.shape[:-1]) + x.shape[-1:]
        min_range = torch.tensor(self.min_range, device=x.device).float().view(range_shape)
        max_range = torch.tensor(self.max_range, device=x.device).float().view(range_shape)

        # Normalize
        x = ((x - min_range) / (max_range - min_range)) * 2 - 1

        # Get mask
        mask = ~torch.any((x < -1) | (x > 1), dim=-1, keepdim=True)

        # Append extra dimension if necessary
        x = x.view(batch_size, self.num_factors, self.dims_per_factor)

        if self.dims_per_factor == 1:
            x = torch.cat([x, torch.zeros_like(x)], -1)

        # Offset
        offset = (torch.linspace(
            0.0,
            (self.num_factors - 1) * self.height,
            self.num_factors,
            device=x.device
        )[None] / (self.size[0] - 1)) * 2 - 1
        x = torch.stack(
            [
                x[..., 0],
                (x[..., 1] + 1) * self.height_factor + offset,
            ],
            -1
        )

        # Mask
        #all_feature = x.new_zeros(x.shape[0], self.out_channels)
        #x = x[mask.repeat(1, x.shape[-1])].view(-1, x.shape[-1])
        x = torch.where(mask, x, 1e8 * torch.ones_like(x))

        # Reshape
        x = x.view(-1, self.dims_per_factor)

        # REshape again
        input_shape = (1,) + x.shape[:1] + tuple(1 for s in self.size[:-1]) + x.shape[-1:]
        x = x.view(input_shape)

        # Sample feature
        feature = nn.functional.grid_sample(
            self.tensor.unsqueeze(0), x, mode=self.mode, padding_mode=self.padding_mode, align_corners=False
        )

        # Reshape feature
        feature = feature.reshape(
            self.out_channels, feature.shape[2]
        ).permute(1, 0)

        ## All feature
        #all_feature[mask.repeat(1, self.out_channels)] = feature.reshape(-1)
        #feature = all_feature.view(-1, self.out_channels)

        # Product
        feature = feature.reshape(batch_size, self.num_factors, self.out_channels)

        if self.product_mode == 'product':
            feature = torch.prod(feature, dim=1)
        elif self.product_mode == 'concat':
            feature = feature.view(batch_size, -1)

        return feature

    def set_iter(self, i):
        pass


class ArrayNDSubdivided(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        cfg,
        **kwargs
    ):
        super().__init__()

        self.opt_group = cfg.group if 'group' in cfg else (kwargs['group'] if 'group' in kwargs else 'color')

        # Number of input, output channels
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Input channels for each tensor
        self.input_channels = list(cfg.input_channels)

        # Size and range
        self.size = list(cfg.size)
        self.range = list(cfg.range)

        if len(self.size) == 1:
            self.size = self.size + [1]

        self.min_range = [r[0] for r in self.range]
        self.max_range = [r[1] for r in self.range]

        # Subdivisions
        self.grid_size = list(cfg.grid_size)
        self.tensor_size = [self.size[idx] // self.grid_size[self.input_channels[idx]] for idx in range(2)]
        self.full_size = self.tensor_size[0:1] \
            + [self.grid_size[0] * self.grid_size[1] * self.grid_size[2] * self.tensor_size[1]]

        # Grid sample options
        self.mode = cfg.mode if 'mode' in cfg else 'bilinear'
        self.padding_mode = cfg.padding_mode if 'padding_mode' in cfg else 'zeros'

        # Activation
        self.activation = cfg.activation if 'activation' in cfg else 'identity'
        self.out_layer = get_activation(self.activation)

        # Tensor setup
        cur_size = (self.out_channels,) + tuple(self.full_size[::-1])

        self.tensor = nn.Parameter(torch.Tensor(*cur_size))
        if self.opt_group == 'color':
            self.tensor.data.uniform_(-1.0, 1.0)
        else:
            self.tensor.data.uniform_(-0.01, 0.01)

        #self.tensor = nn.Parameter(0.1 * torch.randn(cur_size, device='cuda'))

    def forward(self, x, **render_kwargs):
        # Bounds
        range_shape = tuple(1 for s in x.shape[:-1]) + (3,)
        min_range = torch.tensor(self.min_range, device=x.device).float().view(range_shape)
        max_range = torch.tensor(self.max_range, device=x.device).float().view(range_shape)

        # Voxel index
        voxel_idx = torch.clip(x[..., :3], min_range, max_range)
        grid_size = torch.tensor(self.grid_size, device=x.device).float().view(range_shape)
        voxel_idx = torch.floor(((voxel_idx - min_range) / (max_range - min_range)) * grid_size)
        voxel_idx = torch.clip(voxel_idx, torch.zeros_like(grid_size), grid_size - 1)

        voxel_idx = voxel_idx[..., 2] * np.prod(self.grid_size[0:2]) \
            + voxel_idx[..., 1] * np.prod(self.grid_size[0:1]) \
            + voxel_idx[..., 0]

        # Get relevant coordinates
        x = my_index(x, self.input_channels)
        min_range = min_range[..., self.input_channels]
        max_range = max_range[..., self.input_channels]

        # Normalize
        x = ((x - min_range) / (max_range - min_range))

        # Get mask
        mask = ~torch.any((x < 0) | (x > 1), dim=-1, keepdim=True)

        # Append extra dimension if necessary
        if x.shape[-1] == 1:
            x = torch.cat([x, torch.zeros_like(x)], -1)

        # Get tensor look-up coordinates
        size_shape = tuple(1 for s in x.shape[:-1]) + x.shape[-1:]
        size = torch.tensor(self.size, device=x.device).float().view(size_shape)
        tensor_size = torch.tensor(self.tensor_size, device=x.device).float().view(size_shape)
        full_size = torch.tensor(self.full_size, device=x.device).float().view(size_shape)

        x = torch.remainder(x * size, tensor_size)
        x = (torch.stack(
            [
                x[..., 0],
                x[..., 1] + voxel_idx * tensor_size[..., 1],
            ],
            dim=-1
        ) / full_size) * 2 - 1

        # Only evalaute valid outputs
        #all_feature = x.new_zeros(x.shape[0], self.out_channels)
        #x = x[mask.repeat(1, x.shape[-1])].view(-1, x.shape[-1]) # (Option 1) tensor containing only valid
        x = torch.where(mask, x, 1e8 * torch.ones_like(x)) # (Option 2) push invalid out of bounds

        # Reshape
        input_shape = (1,) + x.shape[:1] + tuple(1 for s in self.size[:-1]) + x.shape[-1:]
        x = x.view(input_shape)

        # Sample feature
        feature = nn.functional.grid_sample(
            self.tensor.unsqueeze(0), x, mode=self.mode, padding_mode=self.padding_mode, align_corners=False
        )

        # Reshape
        feature = feature.reshape(
            self.out_channels, feature.shape[2]
        ).permute(1, 0)

        ## All feature
        #all_feature[mask.repeat(1, self.out_channels)] = feature.reshape(-1)
        #feature = all_feature.view(-1, self.out_channels)

        # Return
        return feature

    def set_iter(self, i):
        pass


array_dict = {
    'array_nd': ArrayND,
    'array_nd_multiple': ArrayNDMultiple,
    'array_nd_subdivided': ArrayNDSubdivided,
}
