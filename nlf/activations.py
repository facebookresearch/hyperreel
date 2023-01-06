#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
from torch import nn

from utils.rotation_conversions import axis_angle_to_matrix


class LeakyReLU(nn.Module):
    def __init__(self, cfg, **kwargs):
        super().__init__()

        if "a" in cfg and not isinstance(cfg, str):
            self.a = cfg.a
        else:
            self.a = 0.01

        if "inplace" not in kwargs:
            kwargs["inplace"] = True

        self.act = nn.LeakyReLU(self.a, **kwargs)

    def forward(self, x):
        return self.act(x)


class ReLU(nn.Module):
    def __init__(self, cfg, **kwargs):
        super().__init__()

        if "inplace" not in kwargs:
            kwargs["inplace"] = False

        self.act = nn.ReLU(**kwargs)

    def forward(self, x):
        return self.act(x)


class Abs(nn.Module):
    def __init__(self, cfg):
        super().__init__()

    def forward(self, x):
        return torch.abs(x)


class Sigmoid(nn.Module):
    def __init__(self, cfg, **kwargs):
        super().__init__()

        self.act = nn.Sigmoid(**kwargs)
        self.inner_fac = cfg["inner_fac"] if "inner_fac" in cfg else 1.0
        self.outer_fac = cfg["outer_fac"] if "outer_fac" in cfg else 1.0
        self.shift = cfg.shift if "shift" in cfg else 0.0

        if "fac" in cfg:
            self.outer_fac = cfg["fac"]

    def forward(self, x):
        return self.act(x * self.inner_fac + self.shift) * self.outer_fac

    def set_iter(self, i):
        self.cur_iter = i


class Softplus(nn.Module):
    def __init__(self, cfg, **kwargs):
        super().__init__()

        self.inner_fac = cfg["inner_fac"] if "inner_fac" in cfg else 1.0
        self.outer_fac = cfg["outer_fac"] if "outer_fac" in cfg else 1.0
        self.shift = cfg.shift if "shift" in cfg else 0.0

        if "fac" in cfg:
            self.outer_fac = cfg["fac"]

    def forward(self, x):
        return nn.functional.softplus(x * self.inner_fac + self.shift) * self.outer_fac

    def set_iter(self, i):
        self.cur_iter = i


class Softmax(nn.Module):
    def __init__(self, cfg, **kwargs):
        super().__init__()

        self.act = nn.Softmax(dim=-1, **kwargs)

    def forward(self, x):
        return self.act(x)


class SparseMagnitude(nn.Module):
    def __init__(self, cfg, **kwargs):
        super().__init__()

        self.act = nn.Softmax(dim=-1, **kwargs)
        self.inner_fac = cfg["inner_fac"] if "inner_fac" in cfg else 1.0
        self.outer_fac = cfg["outer_fac"] if "outer_fac" in cfg else 1.0

        if "param_channels" in cfg and not isinstance(cfg, str):
            self.param_channels = cfg.param_channels
        else:
            self.param_channels = 3

    def forward(self, x):
        x = x.view(x.shape[0], -1, self.param_channels)
        mag = torch.linalg.norm(x, dim=-1)
        mag = self.act(mag * self.inner_fac) * self.outer_fac
        x = torch.nn.functional.normalize(x, dim=-1) * mag[..., None]
        return x


class Tanh(nn.Module):
    def __init__(self, cfg, **kwargs):
        super().__init__()

        self.act = nn.Tanh(**kwargs)
        self.inner_fac = cfg["inner_fac"] if "inner_fac" in cfg else 1.0
        self.outer_fac = cfg["outer_fac"] if "outer_fac" in cfg else 1.0
        self.shift = cfg.shift if "shift" in cfg else 0.0

        if "fac" in cfg:
            self.outer_fac = cfg["fac"]

    def forward(self, x):
        return self.act(x * self.inner_fac + self.shift) * self.outer_fac

    def inverse(self, x):
        return (torch.atanh(x / self.outer_fac) - self.shift) / self.inner_fac


class IdentityTanh(nn.Module):
    def __init__(self, cfg, **kwargs):
        super().__init__()

        self.act = nn.Tanh(**kwargs)
        self.fac = cfg["fac"] if "fac" in cfg else 1.0

        # Mapping from [-inf, +inf] to [-1, 1] that acts as an *almost* identity mapping (for most of the space)
        # Derived from an *almost* identity mapping from [-inf, +inf] to [-2, 2] (identity on [-1.9, +1.9])

    def forward(self, x):
        x = x * 2.0

        return (
            torch.where(torch.abs(x) < 1.91501, x, self.act(x) * 2.0) * self.fac / 2.0
        )

    def inverse(self, x):
        x = (x / self.fac) * 2.0

        return torch.where(torch.abs(x) < 1.91501, x, torch.atanh(x / 2.0)) / 2.0


class Identity(nn.Module):
    def __init__(self, cfg, **kwargs):
        super().__init__()

        self.shift = cfg.shift if "shift" in cfg else 0.0
        self.inner_fac = cfg["inner_fac"] if "inner_fac" in cfg else 1.0
        self.outer_fac = cfg["outer_fac"] if "outer_fac" in cfg else 1.0

        if "fac" in cfg:
            self.outer_fac = cfg["fac"]

    def forward(self, x):
        return (x * self.inner_fac + self.shift) * self.outer_fac

    def inverse(self, x):
        return (x / self.outer_fac - self.shift) / self.inner_fac


class Power(nn.Module):
    def __init__(self, cfg, **kwargs):
        super().__init__()

        self.power = cfg["power"] if "power" in cfg else 1.0

    def forward(self, x):
        return torch.pow(torch.abs(x) + 1e-8, self.power) * torch.sign(x)

    def inverse(self, x):
        return torch.pow(torch.abs(x) + 1e-8, 1.0 / self.power) * torch.sign(x)


class L1Norm(nn.Module):
    def __init__(self, cfg):
        super().__init__()

    def forward(self, x):
        return torch.nn.functional.normalize(x, p=1, dim=-1) * x.shape[-1]


class Probs(nn.Module):
    def __init__(self, cfg):
        super().__init__()

    def forward(self, x):
        return torch.nn.functional.normalize(torch.abs(x), p=1, dim=-1)


class RowL2Norm(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        if "param_channels" in cfg and not isinstance(cfg, str):
            self.param_channels = cfg.param_channels
        else:
            self.param_channels = 4

        if "fac" in cfg and not isinstance(cfg, str):
            self.fac = cfg.fac
        else:
            self.fac = 1.0

    def forward(self, x):
        batch_size = x.shape[0]
        total_channels = x.shape[-1]

        if total_channels > 0:
            x = x.view(-1, total_channels // self.param_channels, self.param_channels)
            x = torch.nn.functional.normalize(x, p=2.0, dim=-1)

        return x.view(batch_size, total_channels) * self.fac


class RowL2NormZOnly(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        if "param_channels" in cfg and not isinstance(cfg, str):
            self.param_channels = cfg.param_channels
        else:
            self.param_channels = 4

        if "fac" in cfg and not isinstance(cfg, str):
            self.fac = cfg.fac
        else:
            self.fac = 1.0

    def forward(self, x):
        batch_size = x.shape[0]
        total_channels = x.shape[-1]

        if total_channels > 0:
            x = x.view(-1, total_channels // self.param_channels, self.param_channels)
            x = torch.nn.functional.normalize(x, p=2.0, dim=-1)
            x[..., :-1, :] = torch.eye(
                total_channels // self.param_channels - 1,
                self.param_channels,
                device=x.device,
            )

        return x.view(batch_size, total_channels) * self.fac


class RowLInfNorm(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        if "param_channels" in cfg and not isinstance(cfg, str):
            self.param_channels = cfg.param_channels
        else:
            self.param_channels = 4

        if "fac" in cfg and not isinstance(cfg, str):
            self.fac = cfg.fac
        else:
            self.fac = 1.0

    def forward(self, x):
        batch_size = x.shape[0]
        total_channels = x.shape[-1]

        if total_channels > 0:
            x = x.view(-1, total_channels // self.param_channels, self.param_channels)
            x = torch.nn.functional.normalize(x, p=float("inf"), dim=-1)

        return x.view(batch_size, total_channels) * self.fac


class RowL1Norm(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        if "param_channels" in cfg and not isinstance(cfg, str):
            self.param_channels = cfg.param_channels
        else:
            self.param_channels = 4

        if "fac" in cfg and not isinstance(cfg, str):
            self.fac = cfg.fac
        else:
            self.fac = 1.0

    def forward(self, x):
        batch_size = x.shape[0]
        total_channels = x.shape[-1]

        if total_channels > 0:
            x = x.view(-1, total_channels // self.param_channels, self.param_channels)
            x = torch.nn.functional.normalize(x, p=1, dim=-1)

        return x.view(batch_size, total_channels) * self.fac


class L2Norm(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        if "param_channels" in cfg and cfg.param_channels is not None:
            self.fac = 1.0 / np.sqrt(cfg.param_channels)
        else:
            self.fac = 1.0

    def forward(self, x):
        return (
            torch.nn.functional.normalize(x, p=2.0, dim=-1)
            * np.sqrt(x.shape[-1])
            * self.fac
        )


class Zero(nn.Module):
    def __init__(self, cfg):
        super().__init__()

    def forward(self, x):
        return torch.zeros_like(x)


class RGBA(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.voxel_size = cfg.voxel_size if "voxel_size" in cfg else None
        self.window_iters = cfg.window_iters if "window_iters" in cfg else 0.0
        self.cur_iter = 0

    def forward(self, x):
        raw_alpha = x[..., -1:]

        if self.voxel_size is not None:
            alpha = 1.0 - torch.exp(self.voxel_size * -torch.abs(raw_alpha))
        else:
            alpha = torch.sigmoid(raw_alpha)

        return torch.cat([torch.sigmoid(x[..., :-1]), alpha], -1)

    def set_iter(self, i):
        self.cur_iter = i


class Alpha(nn.Module):
    def __init__(self, cfg):
        super().__init__()

    def forward(self, x):
        return 1.0 - torch.exp(-torch.relu(x))


class Gaussian(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        if "sigma" in cfg and not isinstance(cfg, str):
            self.sigma = cfg.sigma
        else:
            self.sigma = 0.05

    def forward(self, x):
        return torch.exp(-0.5 * torch.square(x / self.sigma))


def se3_hat(twist):
    zero = torch.zeros_like(twist[..., 0])

    mat = torch.stack(
        [
            torch.stack([zero, twist[..., 2], -twist[..., 1], zero], axis=-1),
            torch.stack([-twist[..., 2], zero, twist[..., 0], zero], axis=-1),
            torch.stack([twist[..., 1], -twist[..., 0], zero, zero], axis=-1),
            torch.stack([twist[..., 3], twist[..., 4], twist[..., 5], zero], axis=-1),
        ],
        axis=-1,
    )

    return torch.linalg.matrix_exp(mat)


class TwistToMatrix(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        if "rot_fac" in cfg and not isinstance(cfg, str):
            self.rot_fac = cfg.rot_fac
        else:
            self.rot_fac = 1.0

        if "trans_fac" in cfg and not isinstance(cfg, str):
            self.trans_fac = cfg.trans_fac
        else:
            self.trans_fac = 1.0

    def forward(self, twist):
        twist = torch.cat(
            [
                twist[..., 0:3] * self.rot_fac,
                twist[..., 3:6] * self.trans_fac,
            ],
            -1,
        )

        return se3_hat(twist).view(twist.shape[0], -1)


class AxisAngle(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        if "fac" in cfg and not isinstance(cfg, str):
            self.fac = cfg.fac
        else:
            self.fac = 1.0

    def forward(self, twist):
        axis_angle = twist[..., 0:3] * self.fac
        rot_mat = axis_angle_to_matrix(axis_angle)
        return rot_mat


class AxisAngleTranslation(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        if "rot_fac" in cfg and not isinstance(cfg, str):
            self.rot_fac = cfg.rot_fac
        else:
            self.rot_fac = 1.0

        if "trans_fac" in cfg and not isinstance(cfg, str):
            self.trans_fac = cfg.trans_fac
        else:
            self.trans_fac = 1.0

    def forward(self, twist):
        axis_angle = twist[..., 0:3] * self.rot_fac
        trans = twist[..., 3:6] * self.trans_fac
        rot_mat = axis_angle_to_matrix(axis_angle)

        return torch.cat([rot_mat, trans.unsqueeze(-1)], dim=-1)


class EaseValue(nn.Module):
    def __init__(self, cfg, **kwargs):
        super().__init__()

        self.act = get_activation(cfg.activation, **kwargs)

        self.start_value = cfg.start_value if "start_value" in cfg else 0.0
        self.wait_iters = cfg.wait_iters if "wait_iters" in cfg else 0.0
        self.window_iters = cfg.window_iters if "window_iters" in cfg else 0.0
        self.cur_iter = 0

    def weight(self):
        if self.cur_iter >= self.window_iters:
            return 1.0
        elif self.window_iters == 0:
            return 0.0
        else:
            w = min(max(float(self.cur_iter) / self.window_iters, 0.0), 1.0)
            return w

    def ease_out(self, out):
        if self.cur_iter >= self.window_iters:
            return out
        elif self.window_iters == 0:
            return torch.ones_like(out) * self.start_value
        else:
            w = min(max(float(self.cur_iter) / self.window_iters, 0.0), 1.0)
            return w * out + (1 - w) * self.start_value

    def forward(self, x):
        out = self.act(x)
        return self.ease_out(out)

    def set_iter(self, i):
        self.cur_iter = i - self.wait_iters


class InterpValue(nn.Module):
    def __init__(self, cfg, **kwargs):
        super().__init__()

        self.act1 = get_activation(cfg.act1, **kwargs)
        self.act2 = get_activation(cfg.act2, **kwargs)

        self.wait_iters = cfg.wait_iters if "wait_iters" in cfg else 0.0
        self.window_iters = cfg.window_iters if "window_iters" in cfg else 0.0
        self.cur_iter = 0

    def weight(self):
        if self.cur_iter >= self.window_iters:
            return 1.0
        elif self.window_iters == 0:
            return 0.0
        else:
            w = min(max(float(self.cur_iter) / self.window_iters, 0.0), 1.0)
            return w

    def forward(self, x):
        w = self.weight()

        if w <= 0.0:
            return self.act1(x)
        elif w >= 1.0:
            return self.act2(x)
        else:
            val1 = self.act1(x)
            val2 = self.act2(x)
            return (1.0 - w) * val1 + w * val2

    def set_iter(self, i):
        self.cur_iter = i - self.wait_iters



activation_map = {
    "alpha": Alpha,
    "rgba": RGBA,
    "sigmoid": Sigmoid,
    "softplus": Softplus,
    "softmax": Softmax,
    "sparse_magnitude": SparseMagnitude,
    "tanh": Tanh,
    "identity_tanh": IdentityTanh,
    "identity": Identity,
    "power": Power,
    "probs": Probs,
    "l1_norm": L1Norm,
    "l2_norm": L2Norm,
    "row_l1_norm": RowL1Norm,
    "row_l2_norm": RowL2Norm,
    "row_l2_norm_z_only": RowL2NormZOnly,
    "row_linf_norm": RowLInfNorm,
    "zero": Zero,
    "gaussian": Gaussian,
    "leaky_relu": LeakyReLU,
    "relu": ReLU,
    "abs": Abs,
    "twist_to_matrix": TwistToMatrix,
    "axis_angle_translation": AxisAngleTranslation,
    "ease_value": EaseValue,
    "interp_value": InterpValue,
}


def get_activation(cfg, **kwargs):
    if isinstance(cfg, str):
        return activation_map[cfg]({}, **kwargs)
    else:
        return activation_map[cfg.type](cfg, **kwargs)
