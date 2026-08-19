# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Kimi-K3 DSpark MLA decode, served from prebuilt code objects.

Same kernels and same launch policy as the JIT ops, but the device code is
compiled ahead of time into ``hsa/gfx950/kimi_k3_mla/{draft,verify}_mla.co``
so the first call costs no compile and the
deployment image needs no compiler.  Only the thin loader in
``csrc/py_itfs_cu/kimi_k3_mla.cu`` is built here.

Draft is non-causal, verify is causal, and the mask is compile-time -- hence two
code objects rather than one.  They are not interchangeable: ``draft_mla.co``
carries no ``partial_kernel_w32`` symbol at all, so pointing the verify entry at
it fails at load rather than at build.

Splits, row tile and the partial-accumulator scratch are decided here rather
than in the kernel: they must not allocate while a cudagraph is capturing, so
the buffers are cached and sized by total workgroups, which the launch rule
keeps at roughly one per CU for every batch.
"""

import torch

from aiter.jit.core import compile_ops

D_QK, D_V, KV_TILE = 576, 512, 128
_ROW_TILES = (64, 128)
LOG2E = 1.4426950408889634

_SCRATCH: dict[tuple, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
_NUM_CU: int | None = None


@compile_ops(
    "module_kimi_k3_mla",
    fc_name="kimi_k3_verify_mla_decode",
    ffi_type="ctypes",
)
def _verify_co(
    q: torch.Tensor,
    kv: torch.Tensor,
    out: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    part_m: torch.Tensor,
    part_l: torch.Tensor,
    part_acc: torch.Tensor,
    qlen: int,
    num_splits: int,
    qk_scale_log2: float,
    output_scale: float,
) -> int: ...


@compile_ops(
    "module_kimi_k3_mla",
    fc_name="dspark_draft_mla_decode",
    ffi_type="ctypes",
)
def _draft_co(
    q: torch.Tensor,
    kv: torch.Tensor,
    out: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    part_m: torch.Tensor,
    part_l: torch.Tensor,
    part_acc: torch.Tensor,
    qlen: int,
    num_splits: int,
    qk_scale_log2: float,
    output_scale: float,
) -> int: ...


def _num_cu() -> int:
    global _NUM_CU
    if _NUM_CU is None:
        _NUM_CU = torch.cuda.get_device_properties(
            torch.cuda.current_device()
        ).multi_processor_count
    return _NUM_CU


def _row_tile(rows: int) -> int:
    # Cost is set by the padded tile and the single KV pass, not by the real row
    # count, so a request filling half a 128-row tile pays for the empty half.
    # Above 64 rows the narrow tile would need a second KV pass, which costs far
    # more than the padding it saves.
    for t in _ROW_TILES:
        if rows <= t:
            return t
    return _ROW_TILES[-1]


def _run(
    fn,
    name: str,
    q: torch.Tensor,
    kv_buffer: torch.Tensor,
    out: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_kv: int,
    sm_scale: float,
    q_scale: float,
    kv_scale: float,
) -> None:
    total_s, nhead, dqk = q.shape
    T = int(seq_lens.numel())
    qlen = int(max_seqlen_q)
    page_size = int(kv_buffer.shape[1])

    if dqk != D_QK or out.shape[-1] != D_V:
        raise ValueError(f"expected D_QK={D_QK} D_V={D_V}, got {dqk}/{out.shape[-1]}")
    if total_s != T * qlen:
        raise ValueError(f"ragged batch: {total_s} rows for T={T} qlen={qlen}")
    if page_size % KV_TILE:
        raise ValueError(f"page_size={page_size} must be a multiple of {KV_TILE}")

    rows = nhead * qlen
    tile = _row_tile(rows)
    blocks = -(-rows // tile)
    # One workgroup per CU, exactly one round.  Depends only on shapes, never on
    # seq_lens, so the launch stays identical under graph capture and replay.
    splits = max(1, _num_cu() // (T * blocks))
    splits = max(1, min(splits, max(1, int(max_seqlen_kv) // KV_TILE)))

    wgs = T * splits
    key = (rows, q.device.index)
    scratch = _SCRATCH.get(key)
    if scratch is None or scratch[0].shape[0] < wgs:
        cap = max(wgs, _num_cu())
        opts = {"dtype": torch.float32, "device": q.device}
        scratch = (
            torch.empty(cap, rows, **opts),
            torch.empty(cap, rows, **opts),
            torch.empty(cap, rows, D_V, **opts),
        )
        _SCRATCH[key] = scratch

    err = fn(
        q,
        kv_buffer,
        out,
        block_table,
        seq_lens,
        scratch[0][:wgs].view(T, splits, rows),
        scratch[1][:wgs].view(T, splits, rows),
        scratch[2][:wgs].view(T, splits, rows, D_V),
        qlen,
        splits,
        float(sm_scale) * float(q_scale) * float(kv_scale) * LOG2E,
        float(kv_scale),
    )
    if err:
        raise RuntimeError(f"{name} failed with {err}")


def kimi_k3_verify_mla_decode(*args, **kwargs) -> None:
    """Causal MTP verify block. fp8 q and paged fp8 KV in, bf16 out."""
    _run(_verify_co, "kimi_k3_verify_mla_decode", *args, **kwargs)


def dspark_draft_mla_decode(*args, **kwargs) -> None:
    """Non-causal DSpark draft block. fp8 q and paged fp8 KV in, bf16 out."""
    _run(_draft_co, "dspark_draft_mla_decode", *args, **kwargs)


@compile_ops("module_kimi_k3_mla", fc_name="kimi_k3_mla_warmup", ffi_type="ctypes")
def _warmup() -> int: ...


# Register the code objects now, at import.  Registration is illegal while a
# stream is capturing, and the first real call can land inside cudagraph
# capture: vLLM's pre-capture eager pass is skippable, and when it is skipped
# the kernels would otherwise first be touched from inside the graph.
try:
    _warmup()
except Exception:  # noqa: BLE001 - a missing .co must not break `import aiter`
    pass
