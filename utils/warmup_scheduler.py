#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from torch.optim.lr_scheduler import _LRScheduler


class GradualWarmupScheduler(_LRScheduler):
    """ Gradually warm-up(increasing) learning rate in optimizer.
    Implementation: https://github.com/kwea123/nerf_pl
    Proposed in 'Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour'.
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        multiplier: target learning rate = base lr * multiplier
        total_epoch: target learning rate is reached at total_epoch, gradually
        after_scheduler: after target_epoch, use this scheduler(eg. ReduceLROnPlateau)
    """

    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        self.multiplier = multiplier
        self.total_epoch = total_epoch
        self.after_scheduler = after_scheduler
        self.finished = False

        super().__init__(optimizer)

    def get_lr(self):
        if self.last_epoch >= self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True

                lrs = self.after_scheduler.get_lr()
            else:
                lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
        else:
            lrs = [base_lr * ((self.last_epoch + 1) / (self.total_epoch + 1)) * self.multiplier for base_lr in self.base_lrs]

        return lrs

    def step(self, epoch=None, metrics=None):
        if self.finished and self.after_scheduler:
            self.get_lr()

            if epoch is None:
                self.after_scheduler.step(None)
            else:
                self.after_scheduler.step(epoch - self.total_epoch)
        else:
            return super(GradualWarmupScheduler, self).step(epoch)
