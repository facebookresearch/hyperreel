#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import numpy as np

from functools import partial
from torch.optim import SGD, Adam, RMSprop
from torch.optim.lr_scheduler import (
    CosineAnnealingLR, MultiStepLR, LambdaLR
)

from .warmup_scheduler import GradualWarmupScheduler


def no_init(cfg):
    def init(m):
        pass

    return init

def uniform_weights_init(cfg):
    def init(m):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.uniform_(m.weight, a=-0.1, b=0.1)
            torch.nn.init.uniform_(m.bias, a=-0.1, b=0.1)

    return init

def xavier_uniform_weights_init(cfg):
    def init(m):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            torch.nn.init.uniform_(m.bias, a=-0.01, b=0.01)

    return init

weight_init_dict = {
    'none': no_init,
    'uniform': uniform_weights_init,
    'xavier_uniform': xavier_uniform_weights_init,
}

def to8b(x): return (255*np.clip(x, 0, 1)).astype(np.uint8)

def get_optimizer(hparams, models):
    eps = 1e-8
    parameters = []

    for model in models:
        parameters += list(model.parameters())
    if hparams.optimizer == 'sgd':
        optimizer = SGD(
            parameters, lr=hparams.lr,
            momentum=hparams.momentum,
            weight_decay=hparams.weight_decay
        )
    elif hparams.optimizer == 'adam':
        optimizer = Adam(
            parameters, lr=hparams.lr, eps=eps,
            weight_decay=hparams.weight_decay,
            betas=(0.9, 0.99)
        )
    elif hparams.optimizer == 'rmsprop':
        optimizer = RMSprop(
            parameters, alpha=hparams.alpha, momentum=hparams.momentum,
            lr=hparams.lr, eps=eps,
            weight_decay=hparams.weight_decay
        )
    else:
        raise ValueError('optimizer not recognized!')

    return optimizer

def exp_decay(decay_gamma, stop_epoch, decay_epoch, epoch):
    if epoch > stop_epoch:
        decay = 0.0
    else:
        decay = decay_gamma ** (epoch / decay_epoch)

    return decay

def poly_exp_decay(num_epochs, poly_exp, epoch):
    return (1 - epoch / num_epochs) ** poly_exp

def get_scheduler(hparams, optimizer, iters_per_epoch):
    eps = 1e-8

    if hparams.lr_scheduler == 'exp':
        scheduler = LambdaLR(
            optimizer,
            partial(
                exp_decay,
                hparams.decay_gamma,
                hparams.stop_epoch if 'stop_epoch' in hparams else float("inf"),
                float(hparams.decay_epoch)
            ),
        )
    elif hparams.lr_scheduler == 'steplr':
        scheduler = MultiStepLR(
            optimizer, milestones=[hparams.decay_epoch],
            gamma=hparams.decay_gamma
        )
    elif hparams.lr_scheduler == 'cosine':
        scheduler = CosineAnnealingLR(
            optimizer, T_max=hparams.num_epochs, eta_min=eps
        )
    elif hparams.lr_scheduler == 'poly':
        scheduler = LambdaLR(
            optimizer,
            partial(poly_exp_decay, hparams.num_epochs, hparams.poly_exp),
        )
    else:
        raise ValueError('scheduler not recognized!')

    if hparams.warmup_epochs > 0:
        scheduler = GradualWarmupScheduler(
            optimizer, multiplier=hparams.warmup_multiplier,
            total_epoch=hparams.warmup_epochs, after_scheduler=scheduler
        )

    return scheduler

def get_learning_rate(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def extract_model_state_dict(ckpt_path, model_name='model', prefixes_to_ignore=[]): # noqa
    checkpoint = torch.load(ckpt_path, map_location=torch.device('cpu'))
    checkpoint_ = {}

    if 'state_dict' in checkpoint: # if it's a pytorch-lightning checkpoint
        checkpoint = checkpoint['state_dict']

    for k, v in checkpoint.items():
        if not k.startswith(model_name):
            continue

        k = k[len(model_name)+1:]

        for prefix in prefixes_to_ignore:
            if k.startswith(prefix):
                print('ignore', k)
                break
        else:
            checkpoint_[k] = v

    return checkpoint_

def load_ckpt(model, ckpt_path, model_name='model', prefixes_to_ignore=[]): # noqa
    model_dict = model.state_dict()
    checkpoint_ = extract_model_state_dict(ckpt_path, model_name, prefixes_to_ignore)
    model_dict.update(checkpoint_)
    model.load_state_dict(model_dict)
