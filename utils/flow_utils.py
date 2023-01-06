#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch


def get_base_time(
    t,
    flow_keyframes=-1,
    total_frames=-1,
    flow_scale=0.0,
    jitter=False,
):
    # Get time offset
    if flow_keyframes > 0:
        fac = flow_keyframes * (total_frames - 1) / total_frames
        t = t * fac

        # Offset base time
        if jitter and flow_scale > 0.0:
            base_t = t + (
                torch.rand_like(t) * flow_scale - flow_scale / 2.0
            )
        else:
            base_t = t

        # Round
        base_t = torch.round(base_t.clamp(0.0, flow_keyframes - 1.0) - 1e-5) * (1.0 / fac)
    else:
        base_t = torch.zeros_like(t)

    return base_t


def get_flow_and_time(
    flow,
    t,
    flow_keyframes=-1,
    flow_scale=0.0,
    rigid_flow=False,
    add_rand=False,
    flow_activation=None,
):
    # Get time offset
    if flow_keyframes > 0:
        # Offset base time
        if add_rand and flow_scale > 0.0:
            base_t = t + (
                torch.rand_like(t) * flow_scale - flow_scale / 2.0
            ) / (flow_keyframes - 1)
        else:
            base_t = t

        # Round
        base_t = torch.round(base_t * (flow_keyframes - 1)) / (flow_keyframes - 1)
    else:
        base_t = torch.zeros_like(t)

    base_t = base_t.clamp(0.0, 1.0)

    # Get flow and advect points
    time_offset = (t - base_t)[..., None, :]

    if rigid_flow:
        flow = flow_activation(flow * time_offset)
    else:
        flow = flow_activation(flow) * time_offset

    # Return
    return flow, base_t
