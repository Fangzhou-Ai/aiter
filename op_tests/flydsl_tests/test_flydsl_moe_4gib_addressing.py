# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Check full outputs for experts before/at/after 2 and 4 GiB byte offsets.

Shuffle one expert and copy it to the boundary probes, avoiding a second copy
of the >4 GiB allocation. Stage2 uses a small analytic reference; stage1 compares
with the identical low-address expert, including quantized output scales.
"""

import pytest
import torch

from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.flydsl.moe_kernels import flydsl_moe_stage1, flydsl_moe_stage2
from aiter.ops.flydsl.utils import is_flydsl_available
from aiter.ops.shuffle import shuffle_weight_a16w4

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or get_gfx() != "gfx950" or not is_flydsl_available(),
    reason="gfx950 FlyDSL required",
)
M, N, K = 32, 2048, 2048


def _weights(n, fp8=False, interleave=False):
    per_expert = n * K // (1 if fp8 else 2)
    experts = 2**32 // per_expert + 2
    required = experts * per_expert * 1.2 + 512 * 2**20
    if torch.cuda.mem_get_info()[0] < required:
        pytest.skip(f"requires {required / 2**30:.1f} GiB free VRAM")
    probes = [1, experts - 1]
    for boundary in (2**31, 2**32):
        first = boundary // per_expert
        assert first * per_expert == boundary
        probes.extend((first - 1, first, first + 1))
    probes = sorted(set(probes))
    w = torch.full(
        (experts, n, K // (1 if fp8 else 2)),
        0.5 if fp8 else 0x11,
        dtype=torch.float8_e4m3fn if fp8 else torch.uint8,
        device="cuda",
    )
    # Alternate columns of the GEMM output to catch errors a mean would hide.
    one = torch.full_like(w[:1], 1.0 if fp8 else 0x22)
    one[:, 1::2] = 2.0 if fp8 else 0x44
    one = shuffle_weight_a16w4(one, 16, interleave)
    for expert in probes:
        w[expert].copy_(one[0])
    # Constant e8m0=127 is invariant under scale preshuffling.
    scales = torch.full((experts, n, K // 32), 127, dtype=torch.uint8, device="cuda")
    return w, scales, probes


def _routing(experts):
    # One full sorted block, topk=1; no sorting extension needed for this probe.
    ids = torch.full(((experts + 1) * M,), M, dtype=torch.int32, device="cuda")
    ids[:M] = torch.arange(M, dtype=torch.int32, device="cuda")
    eids = torch.full((experts + 1,), -1, dtype=torch.int32, device="cuda")
    valid = torch.tensor([M, M], dtype=torch.int32, device="cuda")
    return ids, eids, valid, torch.ones_like(ids, dtype=torch.float32)


@pytest.mark.parametrize("backend", ["a16w4", "a16w4_interleave", "port"])
def test_stage1_past_4gib(backend):
    w, scales, probes = _weights(2 * N, interleave=backend == "a16w4_interleave")
    experts = w.shape[0]
    ids, eids, valid, weights = _routing(experts)
    a = torch.ones((M, K), dtype=torch.bfloat16, device="cuda")
    aq = torch.full((M, K // 2), 0x22, dtype=torch.uint8, device="cuda")
    asc = torch.full((M, K // 32), 127, dtype=torch.uint8, device="cuda")

    def run(expert):
        eids[0] = expert
        if backend == "a16w4_interleave":
            from aiter.ops.flydsl.kernels.moe_2stage_a16wmix import flydsl_a16w4_gemm1

            out = torch.empty((M, N), dtype=torch.bfloat16, device="cuda")
            flydsl_a16w4_gemm1(
                a_bf16=a,
                w1_u8=w,
                w1_scale_u8=scales,
                sorted_expert_ids=eids,
                cumsum_tensor=valid,
                m_indices=ids,
                inter_sorted_bf16=out,
                n_tokens=M,
                NE=experts,
                D_HIDDEN=K,
                D_INTER=N,
                topk=1,
                tile_m=M,
                tile_n=256,
                tile_k=256,
                w_layout="guinterleave",
            )
            return (out,)
        if backend == "a16w4":
            out = flydsl_moe_stage1(
                a=a,
                w1=w,
                sorted_token_ids=ids,
                sorted_expert_ids=eids,
                num_valid_ids=valid,
                topk=1,
                tile_m=M,
                tile_n=256,
                tile_k=256,
                a_dtype="bf16",
                b_dtype="fp4",
                out_dtype="bf16",
                w1_scale=scales,
                sorted_weights=weights,
            )
            # The remaining sorted output rows have not been written.
            return (out[:M].clone(),)
        from aiter.ops.flydsl.mxfp4_gemm1_kernels import flydsl_mxfp4_gemm1

        out = torch.zeros((M, N // 2), dtype=torch.uint8, device="cuda")
        out_scale = torch.zeros((M, N // 32), dtype=torch.uint8, device="cuda")
        flydsl_mxfp4_gemm1(
            a_quant=aq,
            a_scale_sorted_shuffled=asc,
            w1_u8=w,
            w1_scale_u8=scales,
            sorted_expert_ids=eids,
            cumsum_tensor=valid,
            m_indices=ids,
            inter_sorted_quant=out,
            inter_sorted_shuffled_scale=out_scale,
            hidden_states=a,
            n_tokens=M,
            BM=M,
            use_nt=False,
            inline_quant=False,
            NE=experts,
            D_HIDDEN=K,
            D_INTER=N,
            topk=1,
        )
        return out, out_scale

    background, expected = run(0), run(probes[0])
    assert any(not torch.equal(a, b) for a, b in zip(background, expected))
    assert all(torch.isfinite(t).all() for t in expected)
    for expert in probes[1:]:
        for result, reference in zip(run(expert), expected):
            torch.testing.assert_close(result, reference, rtol=0, atol=0)


@pytest.mark.parametrize("backend", ["a16w4", "a4w4", "a8w8", "port"])
def test_stage2_past_4gib(backend):
    fp8 = backend == "a8w8"
    w, scales, probes = _weights(N, fp8=fp8)
    experts = w.shape[0]
    ids, eids, valid, weights = _routing(experts)
    if backend == "a16w4":
        a = torch.ones((M, K), dtype=torch.bfloat16, device="cuda")
        asc = None
    else:
        a = torch.full(
            (M, 1, K if fp8 else K // 2),
            1.0 if fp8 else 0x22,
            dtype=torch.float8_e4m3fn if fp8 else torch.uint8,
            device="cuda",
        )
        asc = torch.full((M, K // 32), 127, dtype=torch.uint8, device="cuda")

    def run(expert):
        eids[0] = expert
        out = torch.zeros((M, N), dtype=torch.bfloat16, device="cuda")
        if backend == "port":
            from aiter.ops.flydsl.mxfp4_gemm2_kernels import flydsl_mxfp4_gemm2

            flydsl_mxfp4_gemm2(
                inter_sorted_quant=a,
                inter_sorted_shuffled_scale=asc,
                w2_u8=w,
                w2_scale_u8=scales,
                sorted_expert_ids=eids,
                cumsum_tensor=valid,
                sorted_token_ids=ids,
                sorted_weights=weights,
                flat_out=out,
                M_logical=M,
                max_sorted=M,
                BM=M,
                use_nt=False,
                atomic=True,
                mxfp4out=False,
                NE=experts,
                D_HIDDEN=N,
                D_INTER=K,
                topk=1,
            )
        else:
            flydsl_moe_stage2(
                inter_states=a,
                w2=w,
                sorted_token_ids=ids,
                sorted_expert_ids=eids,
                num_valid_ids=valid,
                topk=1,
                tile_m=M,
                tile_n=256,
                tile_k=256,
                a_dtype="bf16" if backend == "a16w4" else ("fp8" if fp8 else "fp4"),
                b_dtype="fp8" if fp8 else "fp4",
                out_dtype="bf16",
                mode="atomic",
                w2_scale=scales,
                a2_scale=asc,
                sorted_weights=weights,
                out=out,
            )
        return out

    torch.testing.assert_close(
        run(0),
        torch.full((M, N), K * 0.5, dtype=torch.bfloat16, device="cuda"),
        rtol=0,
        atol=0,
    )
    reference = torch.full((M, N), K, dtype=torch.bfloat16, device="cuda")
    reference[:, 1::2] *= 2
    for expert in probes:
        torch.testing.assert_close(run(expert), reference, rtol=0, atol=0)
