#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# @manual //github/third-party/omry/omegaconf:omegaconf
from omegaconf import DictConfig


def format_config(cfg: DictConfig):
    format_config_helper(cfg, cfg)


def format_config_helper(cfg, master_config: DictConfig):
    if isinstance(cfg, DictConfig):
        for key, _ in cfg.items():
            if isinstance(cfg[key], str):
                cfg[key] = cfg[key].format(config=master_config)
            else:
                format_config_helper(cfg[key], master_config)


def replace_config(cfg, **kwargs):
    if isinstance(cfg, DictConfig):
        for key, _ in cfg.items():
            if key in kwargs.keys() and cfg[key] is None:
                cfg[key] = kwargs[key]
            else:
                replace_config(cfg[key], **kwargs)

def lambda_config(cfg, find_key, fn):
    if isinstance(cfg, DictConfig):
        for key, _ in cfg.items():
            if key == find_key:
                fn(cfg, key)
            else:
                lambda_config(cfg[key], find_key, fn)
