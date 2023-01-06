#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import numpy as np
from torch import nn


class IdentityPE(nn.Module):
    def __init__(
        self,
        in_channels,
        *args,
        **kwargs
    ):

        super().__init__()

        self.in_channels = in_channels
        self.out_channels = in_channels

    def forward(self, x):
        return x

    def set_iter(self, i):
        self.cur_iter = i


class BasicPE(nn.Module):
    def __init__(
        self,
        in_channels,
        cfg,
    ):
        super().__init__()

        self.n_freqs = cfg.n_freqs
        self.cur_iter = 0

        self.funcs = [torch.sin, torch.cos]
        self.freq_multiplier = cfg.freq_multiplier if 'freq_multiplier' in cfg else 2.0
        self.freq_bands = (self.freq_multiplier ** torch.linspace(1, cfg.n_freqs, cfg.n_freqs)).cuda()

        self.in_channels = in_channels
        self.out_channels = in_channels * (len(self.funcs) * cfg.n_freqs + 1)

        self.dummy_layer = nn.Linear(1, 1)

    def forward(self, x):
        # Input shape
        input_shape = x.shape

        # Reshape
        x = x.view(-1, x.shape[-1])
        batch_size = x.shape[0]

        # Get PE
        out = [x]

        if self.n_freqs > 0:
            cur_x = (self.freq_bands[None, None] * x[..., None]).view(batch_size, -1)
            out += [torch.sin(cur_x)]
            out += [torch.cos(cur_x)]

        return torch.cat(out, -1)

    def set_iter(self, i):
        self.cur_iter = i


class BasicWindowedPE(nn.Module):
    def __init__(
        self,
        in_channels,
        cfg,
    ):
        super().__init__()

        self.n_freqs = cfg.n_freqs
        self.cur_iter = 0
        self.wait_iters = cfg.wait_iters
        self.max_freq_iter = float(cfg.max_freq_iter)
        self.exclude_identity = cfg.exclude_identity \
            if 'exclude_identity' in cfg \
            else False

        self.funcs = [torch.sin, torch.cos]
        self.freq_multiplier = cfg.freq_multiplier if 'freq_multiplier' in cfg else 2.0
        self.freq_bands = self.freq_multiplier ** torch.linspace(1, cfg.n_freqs, cfg.n_freqs)

        self.in_channels = in_channels
        if self.exclude_identity:
            self.out_channels = in_channels * (len(self.funcs) * cfg.n_freqs)
        else:
            self.out_channels = in_channels * (len(self.funcs) * cfg.n_freqs + 1)

        self.dummy_layer = nn.Linear(1, 1)

    def weight(self, j):
        if self.max_freq_iter == 0:
            return 1.0
        elif self.cur_iter < self.wait_iters:
            return 0.0
        elif self.cur_iter > self.max_freq_iter:
            return 1.0

        cur_iter = (self.cur_iter - self.wait_iters)
        alpha = (cur_iter / self.max_freq_iter) * self.n_freqs
        return (1.0 - np.cos(np.pi * np.clip(alpha - j, 0.0, 1.0))) / 2

    def forward(self, x):
        out = []

        if not self.exclude_identity:
            out += [x]

        for j, freq in enumerate(self.freq_bands):
            for func in self.funcs:
                out += [self.weight(j) * func(freq * x)]

        return torch.cat(out, -1)

    def set_iter(self, i):
        self.cur_iter = i


class WindowedPE(nn.Module):
    def __init__(
        self,
        in_channels,
        cfg,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.funcs = [torch.sin, torch.cos]

        self.cur_iter = 0
        self.wait_iters = cfg.wait_iters
        self.max_freq_iter = float(cfg.max_freq_iter)

        # PE
        self.n_freqs = cfg.n_freqs
        self.freq_multiplier = cfg.freq_multiplier if 'freq_multiplier' in cfg else 2.0
        self.freq_bands = self.freq_multiplier ** torch.linspace(1.0, cfg.n_freqs, cfg.n_freqs)

        # What to do about identity
        self.base_multiplier = cfg.base_multiplier if 'base_multiplier' in cfg else 1.0

        self.ceil = cfg.ceil \
            if 'ceil' in cfg \
            else False
        self.exclude_identity = cfg.exclude_identity \
            if 'exclude_identity' in cfg \
            else False
        self.window_identity = 1 \
            if 'window_identity' in cfg and cfg.window_identity \
            else 0

        if self.exclude_identity:
            self.out_channels = in_channels * (len(self.funcs) * cfg.n_freqs)
        else:
            self.out_channels = in_channels * (len(self.funcs) * cfg.n_freqs + 1)

        # Windowing
        if self.max_freq_iter > 0 or 'window_iters' in cfg:
            self.window_after = (self.max_freq_iter / self.n_freqs)

            if 'window_iters' in cfg:
                self.window_iters = cfg.window_iters
                self.max_freq_iter = np.max(cfg.window_iters)
            elif self.window_identity != 0:
                self.window_iters = [(self.wait_iters, self.window_after + self.wait_iters)] \
                    + [(self.window_after * i + self.wait_iters, self.window_after * (i + 1) + self.wait_iters) \
                        for i in range(1, self.n_freqs + 1)]
                self.max_freq_iter = (self.n_freqs + 1) * self.window_after
            else:
                self.window_iters = [(self.window_after * i + self.wait_iters, self.window_after * (i + 1) + self.wait_iters) \
                    for i in range(0, self.n_freqs)]

        self.dummy_layer = nn.Linear(1, 1)

    def weight(self, j):
        cur_iter = (self.cur_iter - self.wait_iters)

        if j < 0:
            return 1.0
        elif cur_iter < 0:
            return 0.0
        elif self.max_freq_iter == 0:
            return 1.0
        elif self.cur_iter > self.max_freq_iter:
            return 1.0
        elif (self.window_iters[j][1] - self.window_iters[j][0]) == 0:
            if self.cur_iter >= self.window_iters[j][0]:
                return 1.0
            else:
                return 0.0

        alpha = (cur_iter - self.window_iters[j][0]) / float(self.window_iters[j][1] - self.window_iters[j][0])

        if self.ceil:
            return np.ceil((1.0 - np.cos(np.pi * np.clip(alpha, 0.0, 1.0))) / 2)
        else:
            return (1.0 - np.cos(np.pi * np.clip(alpha, 0.0, 1.0))) / 2

    def forward(self, x):
        out = []

        if not self.exclude_identity:
            #out = [self.base_multiplier * self.weight(-1 + self.window_identity) * x]
            out = [x]

        for j, freq in enumerate(self.freq_bands):
            for func in self.funcs:
                out += [self.weight(j + self.window_identity) * func(self.base_multiplier * freq * x)]

        return torch.cat(out, -1)

    def set_iter(self, i):
        self.cur_iter = i


class SelectPE(nn.Module):
    def __init__(
        self,
        in_channels,
        cfg,
    ):
        super().__init__()

        self.start_channel = cfg.start_channel if 'start_channel' in cfg else 0
        self.in_channels = in_channels - self.start_channel
        self.select_channels = cfg.select_channels
        self.discard = cfg.discard if 'discard' in cfg else False

        self.pe = WindowedPE(
            self.select_channels,
            cfg
        )

        if self.discard:
            self.out_channels = self.pe.out_channels
        else:
            self.out_channels = (self.in_channels - self.select_channels) \
                + self.pe.out_channels

    def forward(self, x):
        out_x = self.pe(x[..., self.start_channel:self.start_channel+self.select_channels])

        if not self.discard:
            out_x = torch.cat([ out_x, x[..., self.select_channels:] ], -1)

        return out_x

    def set_iter(self, i):
        self.pe.set_iter(i)


class RandomPE(nn.Module):
    def __init__(
        self,
        in_channels,
        cfg,
    ):
        super().__init__()

        self.n_freqs = cfg.n_freqs
        self.sigma = cfg.sigma
        self.funcs = [torch.sin, torch.cos]

        self.in_channels = in_channels
        self.out_channels = len(self.funcs) * cfg.n_freqs

        self.embedding_matrix = (torch.randn(
            (self.n_freqs, self.in_channels)
        ) * self.sigma).cuda()

    def forward(self, x):
        # Convert to correct device
        embedding_matrix = self.embedding_matrix.type_as(x)

        out = []
        raw = (embedding_matrix @ x.permute(1, 0)).permute(1, 0)

        for func in self.funcs:
            out += [func(raw)]

        return torch.cat(out, -1)

    def set_iter(self, i):
        self.cur_iter = i


class WindowedRandomPE(nn.Module):
    def __init__(
        self,
        in_channels,
        cfg,
    ):
        super().__init__()

        self.n_freqs = cfg.n_freqs
        self.sigma = cfg.sigma
        self.funcs = [torch.sin, torch.cos]

        self.in_channels = in_channels
        self.out_channels = len(self.funcs) * cfg.n_freqs + self.in_channels

        # Embedding matrix
        self.embedding_matrix = (torch.randn(
            (self.in_channels, self.n_freqs)
        ) * self.sigma).cuda()

        self.embedding_mag = torch.norm(self.embedding_matrix, dim=0)
        sort_idx = torch.argsort(self.embedding_mag, dim=0)

        self.embedding_matrix = torch.gather(
            self.embedding_matrix, -1,
            sort_idx[None].repeat(self.in_channels, 1)
        )

        self.embedding_mag = torch.norm(self.embedding_matrix, dim=0)

        # Windowing
        self.cur_iter = 0
        self.wait_iters = cfg.wait_iters
        self.max_freq_iter = float(cfg.max_freq_iter)

        self.ceil = cfg.ceil \
            if 'ceil' in cfg \
            else False
        self.window_identity = 1 \
            if 'window_identity' in cfg and cfg.window_identity \
            else 0

        # Windowing
        if self.max_freq_iter > 0 or 'window_iters' in cfg:
            self.window_after = (self.max_freq_iter / self.n_freqs)

            if 'window_iters' in cfg:
                self.window_iters = cfg.window_iters
                self.max_freq_iter = np.max(cfg.window_iters)
            elif self.window_identity != 0:
                self.window_iters = [(self.wait_iters, self.window_after + self.wait_iters)] \
                    + [(self.window_after * i + self.wait_iters, self.window_after * (i + 1) + self.wait_iters) \
                        for i in range(1, self.n_freqs + 1)]
                self.max_freq_iter = (self.n_freqs + 1) * self.window_after
            else:
                self.window_iters = [(self.window_after * i + self.wait_iters, self.window_after * (i + 1) + self.wait_iters) \
                    for i in range(0, self.n_freqs)]

    def weight(self, j):
        cur_iter = (self.cur_iter - self.wait_iters)

        if cur_iter < 0:
            return 0.0
        elif j < 0:
            return 1.0
        elif self.max_freq_iter == 0:
            return 1.0
        elif self.cur_iter > self.max_freq_iter:
            return 1.0
        elif (self.window_iters[j][1] - self.window_iters[j][0]) == 0:
            if self.cur_iter >= self.window_iters[j][0]:
                return 1.0
            else:
                return 0.0

        alpha = (cur_iter - self.window_iters[j][0]) / float(self.window_iters[j][1] - self.window_iters[j][0])

        if self.ceil:
            return np.ceil((1.0 - np.cos(np.pi * np.clip(alpha, 0.0, 1.0))) / 2)
        else:
            return (1.0 - np.cos(np.pi * np.clip(alpha, 0.0, 1.0))) / 2

    def forward(self, x):
        # Convert to correct device
        embedding_matrix = self.embedding_matrix.type_as(x)

        raw = (x @ embedding_matrix)

        out = [self.weight(-1 + self.window_identity) * x]

        for j in range(raw.shape[-1]):
            for func in self.funcs:
                out += [self.weight(j + self.window_identity) * func(raw[..., j:j+1])]

        return torch.cat(out, -1)

    def set_iter(self, i):
        self.cur_iter = i


class LearnablePE(nn.Module):
    def __init__(
        self,
        in_channels,
        cfg,
    ):
        super().__init__()

        self.n_freqs = cfg.n_freqs
        self.sigma = cfg.sigma
        self.funcs = [torch.sin, torch.cos]

        self.in_channels = in_channels
        self.out_channels = len(self.funcs) * cfg.n_freqs
        self.embedding_layer = nn.Linear(in_channels, cfg.n_freqs)

        self.embedding_matrix = (torch.randn(
            (self.n_freqs, self.in_channels)
        ) * self.sigma).cuda()
        self.embedding_matrix = nn.Parameter(
            self.embedding_matrix, requires_grad=True
        )

        self.embedding_bias = (torch.randn(
            (1, self.n_freqs)
        ) * self.sigma).cuda()
        self.embedding_bias = nn.Parameter(
            self.embedding_bias, requires_grad=True
        )

    def forward(self, x):
        # Convert to correct device
        embedding_matrix = self.embedding_matrix.type_as(x)
        embedding_bias = self.embedding_bias.type_as(x)

        out = []
        raw = (embedding_matrix @ x.permute(1, 0)).permute(1, 0) + embedding_bias

        for func in self.funcs:
            out += [func(raw)]

        return torch.cat(out, -1)

    def set_iter(self, i):
        self.cur_iter = i


pe_dict = {
    'basic': BasicPE,
    'windowed': WindowedPE,
    'identity': IdentityPE,
    'random': RandomPE,
    'windowed_random': WindowedRandomPE,
    'learnable': LearnablePE,
    'select': SelectPE,
}
