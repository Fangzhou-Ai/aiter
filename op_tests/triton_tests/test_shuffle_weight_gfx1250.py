# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import pytest
import torch

from aiter.ops.shuffle import shuffle_weight_gfx1250


def _explicit_interleave_last(w):
    """Independent [gate|up] -> [g0,u0,g1,u1,...] on the last axis via explicit
    column placement (no reshape/permute), used as the reference."""
    N = w.shape[-1]
    half = N // 2
    out = torch.empty_like(w)
    out[..., 0::2] = w[..., :half]  # gate -> even cols
    out[..., 1::2] = w[..., half:]  # up   -> odd cols
    return out


def _explicit_interleave_dim0(w):
    """Same interleave along dim 0 (2D N-major)."""
    N = w.shape[0]
    half = N // 2
    out = torch.empty_like(w)
    out[0::2] = w[:half]
    out[1::2] = w[half:]
    return out


@pytest.mark.parametrize("E", [1, 3])
@pytest.mark.parametrize("N", [64, 128])
@pytest.mark.parametrize("K", [32, 64])
def test_guinterleave_matches_manual_3d(E, N, K):
    # 3D layout is (E, K, N); gate/up live on the last (N) axis.
    w = torch.randint(0, 256, (E, K, N), dtype=torch.uint8)
    fused = shuffle_weight_gfx1250(w, is_guinterleave=True, gate_up=True)
    manual = shuffle_weight_gfx1250(_explicit_interleave_last(w))
    assert torch.equal(fused, manual)


@pytest.mark.parametrize("N", [64, 128])
@pytest.mark.parametrize("K", [32, 64])
def test_guinterleave_matches_manual_2d(N, K):
    w = torch.randint(0, 256, (N, K), dtype=torch.uint8)
    fused = shuffle_weight_gfx1250(w, is_guinterleave=True, gate_up=True)
    manual = shuffle_weight_gfx1250(_explicit_interleave_dim0(w))
    assert torch.equal(fused, manual)


def test_default_unchanged():
    # is_guinterleave=False must be byte-identical to the pre-change behavior.
    w = torch.randint(0, 256, (2, 64, 64), dtype=torch.uint8)
    assert torch.equal(
        shuffle_weight_gfx1250(w), shuffle_weight_gfx1250(w, gate_up=True)
    )


def test_guinterleave_requires_gate_up():
    w = torch.randint(0, 256, (2, 64, 64), dtype=torch.uint8)
    with pytest.raises(ValueError):
        shuffle_weight_gfx1250(w, is_guinterleave=True, gate_up=False)
