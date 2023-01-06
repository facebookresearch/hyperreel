#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn
import numpy as np

from collections import defaultdict

from .nets import net_dict


class Render(nn.Module):
    def __init__(
        self,
        model,
        subdivision,
        cfg,
        **kwargs
    ):
        super().__init__()

        if 'net_chunk' in kwargs:
            self.net_chunk = kwargs['net_chunk']
        else:
            self.net_chunk = 32768

    def _run(self, x, fn, **render_kwargs):
        x = x.view(-1, x.shape[-1])
        out = fn(x, render_kwargs)
        return out.view(-1, out.shape[-1])

    def _run_multiple(self, x, fn, **render_kwargs):
        x = x.view(-1, x.shape[-1])
        out = fn(x, render_kwargs)

        for key in out.keys():
            out[key] = out[key].view(-1, out[key].shape[-1])

        return out

    def _run_chunked(self, x, fn, **render_kwargs):
        x = x.view(-1, x.shape[-1])

        # Chunked inference
        B = x.shape[0]
        out_chunks = []

        for i in range(0, B, self.net_chunk):
            out_chunks += [fn(x[i:i+self.net_chunk], render_kwargs)]

        out = torch.cat(out_chunks, 0)
        return out.view(-1, out.shape[-1])


class RenderLightfield(Render):
    def __init__(
        self,
        model,
        subdivision,
        cfg,
        *args,
        **kwargs
    ):
        super().__init__(model, subdivision, cfg, **kwargs)

        self.model = model

    def forward(self, rays, **render_kwargs):
        return self._run_multiple(
            rays,
            self.model,
            **render_kwargs
        )

    def embed(self, rays, **render_kwargs):
        return self._run_multiple(
            rays,
            self.model.embed,
            **render_kwargs
        )

    def forward_multiple(self, rays, **render_kwargs):
        return self._run_multiple(
            rays,
            self.model,
            **render_kwargs
        )



render_fn_dict = {
    'lightfield': RenderLightfield,
}


def render_chunked(
    rays,
    render_fn,
    render_kwargs,
    chunk,
    ):
    B = rays.shape[0]
    results = defaultdict(list)
    chunk_args = getattr(render_kwargs, 'chunk_args', None)

    for i in range(0, B, chunk):
        # Create render arguments
        if chunk_args is None:
            chunk_render_kwargs = render_kwargs
        else:
            chunk_render_kwargs = {}

            for k in render_kwargs.keys():
                # Chunk this argument
                if k in chunk_args:
                    chunk_render_kwargs[k] = {}

                    for j in render_kwargs[k]:
                        chunk_render_kwargs[k][j] = render_kwargs[k][j][i:i+chunk]
                # Pass full argument
                else:
                    chunk_render_kwargs[k] = render_kwargs[k]

        # Run forward in chunks
        rendered_ray_chunks = \
            render_fn(
                rays[i:i+chunk],
                **chunk_render_kwargs
            )

        # Accumulate chunks
        for k, v in rendered_ray_chunks.items():
            results[k] += [v]

    # Concatenate chunks
    for k, v in results.items():
        if isinstance(v[0], list):
            if 'weights' in k:
                results[k] = v[0]
            else:
                results[k] = torch.cat([torch.stack(item, 0) for item in v], 1)
                results[k] = [results[k][idx] for idx in range(results[k].shape[0])]
        else:
            results[k] = torch.cat(v, 0)

    return results
