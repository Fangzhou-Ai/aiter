# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""gfx950 FlyDSL backend for mxfp8_128 bpreshuffle GEMM.

The gfx950 counterpart of ``mxfp8_128_bpreshuffle_gemm_gfx1250.py``. Same
public shape (a ``run_*`` entry plus a tuned-kernelName dispatch entry), but
backed by the CDNA4 wave64 MFMA kernel ``kernels/mxfp4_preshuffle.py``
(``V_MFMA_SCALE_F32_16X16X128_F8F6F4``) instead of the gfx1250 WMMA/TDM kernel.

mxfp8_128: fp8 e4m3 activations/weights with fp8_e8m0 block scales over a
128-element block (distinct from the 32-element-block MXFP8 and from the
fp32-scale blockscale GEMM).

Differences from the gfx1250 backend that the caller must be aware of:

* **Scale layout.** gfx1250 takes ``x_scale`` as ``(M, K//128)`` e8m0 with
  M-contiguous (transposed) storage and ``w_scale`` as ``(N//128, K//128)``
  dense row-major. The CDNA4 scaled-MFMA reads its scale operand from a
  32-row x 256-K dword block, so this backend wants the ``compact_scale_w4``
  layout instead -- a pure integer permutation of the same e8m0 bytes, with
  the B scale broadcast along N to 32-row granularity.
* **Tile knobs.** gfx1250 exposes ``m_warp / n_warp / num_buffers /
  cluster_m / cluster_n``; the MFMA kernel exposes ``waves_per_eu /
  xcd_swizzle`` (wave count is fixed at 4 N-waves, the pipeline is a fixed
  double buffer).
* **split_k.** Not implemented by the MFMA kernel; ``split_k`` must be 1.
"""

from __future__ import annotations

import re

import torch
from torch import Tensor

_launch_gemm = None
_ptr_arg = None

_BLOCK_K = 128
_BLOCK_N = 128
_ROW_GROUP = 32  # compact_scale_w4 works on 32-row groups
_SCALE_CHUNK_K = 256  # ... x 256-K chunks
_OUT_DTYPE_NAME = {torch.bfloat16: "bf16", torch.float16: "f16"}
_MAX_SPLIT_K = 1


def _lazy_import():
    global _launch_gemm, _ptr_arg
    if _launch_gemm is not None:
        return
    from .kernels.mxfp4_preshuffle import launch_gemm
    from .kernels.tensor_shim import ptr_arg

    _launch_gemm = launch_gemm
    _ptr_arg = ptr_arg


def compact_scale_w4(scale_e8m0: Tensor, block_n: int, K: int) -> Tensor:
    """``(R//block_n, K//128)`` e8m0 -> the dword layout the MFMA kernel reads.

    Byte ``(r, kb)`` of the logical scale lands at flat offset
    ``((((r//32)*(K//256) + kb//2)*16 + r%16)*2 + kb%2)*2 + (r%32)//16``.
    ``block_n`` broadcasts along rows (1 for the A scale, 128 for the B scale).
    Returns a 1-D uint32 tensor of ``R//32 * K//256 * 16`` words.
    """
    R = scale_e8m0.shape[0] * block_n
    k_blocks = K // _BLOCK_K
    if scale_e8m0.shape[1] != k_blocks:
        raise RuntimeError(
            f"[FlyDSL gfx950 mxfp8_128] scale must have {k_blocks} K-blocks, "
            f"got {scale_e8m0.shape[1]}"
        )
    if R % _ROW_GROUP != 0 or K % _SCALE_CHUNK_K != 0:
        raise RuntimeError(
            f"[FlyDSL gfx950 mxfp8_128] compact scale needs rows%{_ROW_GROUP}==0 and "
            f"K%{_SCALE_CHUNK_K}==0, got rows={R}, K={K}"
        )
    s = scale_e8m0.view(torch.uint8).repeat_interleave(block_n, 0)
    s = s.view(R // _ROW_GROUP, 2, 16, K // _SCALE_CHUNK_K, 2).permute(0, 3, 2, 4, 1)
    return s.contiguous().reshape(-1).view(torch.uint32)


def _require_compact_scale(scale: Tensor, words: int, name: str) -> Tensor:
    if scale.dtype not in (torch.uint32, torch.int32, torch.uint8):
        raise RuntimeError(
            f"[FlyDSL gfx950 mxfp8_128] {name} must be a compact_scale_w4 buffer "
            f"(uint32/uint8), got {scale.dtype}"
        )
    got = scale.numel() * scale.element_size() // 4
    if got != words:
        raise RuntimeError(
            f"[FlyDSL gfx950 mxfp8_128] {name} must hold {words} dwords, got {got}"
        )
    return scale


def run_mxfp8_128_preshuffle_gemm_a8_gfx950(
    XQ: Tensor,
    WQ: Tensor,
    x_scale: Tensor,
    w_scale: Tensor,
    Out: Tensor,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    *,
    waves_per_eu: int = 0,
    xcd_swizzle: int = 8,
    split_k: int = 1,
    scale_is_compact: bool = True,
) -> Tensor:
    """Run the gfx950 MFMA mxfp8_128 bpreshuffle GEMM.

    XQ: ``(M, K)`` FP8 E4M3 row-major. WQ: ``(N, K)`` FP8 E4M3, already 16x16
    preshuffled (``aiter.ops.shuffle.shuffle_weight(w, layout=(16,16))``,
    byte-identical to FlyDSL's ``shuffle_weight_w4(w, 16)``). Out: ``(M, N)``
    bf16/f16 row-major, written in place.

    With ``scale_is_compact=True`` (default) ``x_scale`` / ``w_scale`` are the
    1-D ``compact_scale_w4`` buffers. With ``scale_is_compact=False`` they are
    the dense ``(M, K//128)`` / ``(N//128, K//128)`` e8m0 tensors and this
    function repacks them on the fly -- correct but it costs two extra kernels
    per call, so production callers should emit the compact layout upstream.
    """
    _lazy_import()

    if XQ.dim() != 2 or WQ.dim() != 2:
        raise RuntimeError(
            "[FlyDSL gfx950 mxfp8_128] A/B must be 2-D, got "
            f"{tuple(XQ.shape)}, {tuple(WQ.shape)}"
        )
    if XQ.element_size() != 1 or WQ.element_size() != 1:
        raise RuntimeError("[FlyDSL gfx950 mxfp8_128] A/B must be 1-byte fp8 storage")

    M, K = XQ.shape
    N = WQ.shape[0]
    if K != WQ.shape[1]:
        raise RuntimeError(
            f"[FlyDSL gfx950 mxfp8_128] K mismatch: A.K={K} vs B.K={WQ.shape[1]}"
        )
    if N % _BLOCK_N != 0 or K % _BLOCK_K != 0:
        raise RuntimeError(
            f"[FlyDSL gfx950 mxfp8_128] N/K must be multiples of "
            f"{_BLOCK_N}/{_BLOCK_K}, got N={N}, K={K}"
        )
    if K % _SCALE_CHUNK_K != 0:
        raise RuntimeError(
            f"[FlyDSL gfx950 mxfp8_128] K must be a multiple of {_SCALE_CHUNK_K} "
            f"(the 128-granular e8m0 scale is stored in 256-K chunks), got K={K}"
        )
    if M % _ROW_GROUP != 0:
        raise RuntimeError(
            f"[FlyDSL gfx950 mxfp8_128] M must be a multiple of {_ROW_GROUP} "
            f"(compact scale row group); pad the activation, got M={M}"
        )

    out_dtype = _OUT_DTYPE_NAME.get(Out.dtype)
    if out_dtype is None:
        raise RuntimeError(
            f"[FlyDSL gfx950 mxfp8_128] unsupported out dtype {Out.dtype}; "
            "expected bf16/fp16"
        )

    split_k = max(1, int(split_k))
    if split_k > _MAX_SPLIT_K:
        raise RuntimeError(
            f"[FlyDSL gfx950 mxfp8_128] split_k={split_k} unsupported; the MFMA "
            f"kernel has no split-k path (max {_MAX_SPLIT_K})"
        )

    # Tile constraints of the MFMA preshuffle pipeline (see kernels/mxfp4_preshuffle.py):
    # a K-tile is one LDS double-buffer half, and the scale chunk is 256-K.
    if N % tile_n != 0 or tile_n % 64 != 0:
        raise RuntimeError(
            f"[FlyDSL gfx950 mxfp8_128] tile_n={tile_n} must divide N={N} and be a "
            "multiple of 64"
        )
    if K % tile_k != 0 or _SCALE_CHUNK_K % tile_k != 0:
        raise RuntimeError(
            f"[FlyDSL gfx950 mxfp8_128] tile_k={tile_k} must divide both K={K} and "
            f"{_SCALE_CHUNK_K}"
        )
    if tile_m % 16 != 0 or (tile_m * tile_k) % 4096 != 0:
        raise RuntimeError(
            f"[FlyDSL gfx950 mxfp8_128] tile_m={tile_m} must be a multiple of 16 with "
            f"tile_m*tile_k%4096==0 (got {tile_m * tile_k})"
        )
    if 2 * tile_m * tile_k > 65536:
        raise RuntimeError(
            f"[FlyDSL gfx950 mxfp8_128] LDS A double buffer 2*{tile_m}*{tile_k} exceeds "
            "64KB"
        )

    k_blocks = K // _BLOCK_K
    if scale_is_compact:
        a_words = (M // _ROW_GROUP) * (K // _SCALE_CHUNK_K) * 16
        b_words = (N // _ROW_GROUP) * (K // _SCALE_CHUNK_K) * 16
        a_scale = _require_compact_scale(x_scale, a_words, "x_scale")
        b_scale = _require_compact_scale(w_scale, b_words, "w_scale")
    else:
        a_scale = compact_scale_w4(x_scale.view(M, k_blocks), 1, K)
        b_scale = compact_scale_w4(w_scale.view(N // _BLOCK_N, k_blocks), _BLOCK_N, K)

    _launch_gemm(
        _ptr_arg(Out),
        _ptr_arg(XQ),
        _ptr_arg(WQ),
        _ptr_arg(a_scale),
        _ptr_arg(b_scale),
        M,
        N,
        torch.cuda.current_stream(device=XQ.device),
        N,
        K,
        tile_m,
        tile_n,
        tile_k,
        "fp8",
        out_dtype,
        "fp8",
        1,  # batch
        -1,  # a_row_stride   (-1 = contiguous default)
        -1,  # a_batch_stride
        -1,  # sca_row_stride
        -1,  # sca_batch_stride
        -1,  # c_row_stride
        -1,  # c_batch_stride
        int(waves_per_eu),
        int(xcd_swizzle),
        _BLOCK_K,  # scale_block_k = 128 (blockscale, not per-32 MX)
    )
    return Out


_KERNEL_NAME_RE = re.compile(
    r"^flydsl_mxfp8_128_bpreshuffle_mfma_"
    r"t(?P<tile_m>\d+)x(?P<tile_n>\d+)x(?P<tile_k>\d+)_"
    r"wpe(?P<waves_per_eu>\d+)_xcd(?P<xcd_swizzle>\d+)$"
)


def parse_mfma_kernel_name(name: str):
    """Parse a flydsl_mxfp8_128_bpreshuffle_mfma_ kernelName, or return None."""
    m = _KERNEL_NAME_RE.fullmatch(name)
    return {k: int(v) for k, v in m.groupdict().items()} if m else None


def default_kernel_name(M: int, N: int, K: int) -> str:
    """Untuned fallback tile, from the N=65536 K=1536 M-sweep."""
    tile_m, tile_n, tile_k, xcd = (
        (32, 128, 128, 0)
        if M <= 32
        else (96, 128, 128, 0)
        if M <= 64
        else (128, 128, 128, 0)
        if M <= 128
        else (128, 256, 128, 8)
    )
    if N % tile_n != 0:
        tile_n = 128
    return (
        f"flydsl_mxfp8_128_bpreshuffle_mfma_t{tile_m}x{tile_n}x{tile_k}_wpe0_xcd{xcd}"
    )


def run_gemm_a8w8_mxfp8_128_bpreshuffle_gfx950(
    XQ: Tensor,
    WQ: Tensor,
    x_scale: Tensor,
    w_scale: Tensor,
    Out: Tensor,
    kernel_name: str = "",
) -> Tensor:
    """Dispatch entry: decode a tuned mfma kernelName and run the kernel.

    An empty ``kernel_name`` falls back to ``default_kernel_name``. A ``uint32``
    scale is taken as already ``compact_scale_w4``-packed; a dense ``fp8_e8m0``
    scale (the layout the gfx1250 backend takes) is repacked on the fly.
    """
    if not kernel_name:
        kernel_name = default_kernel_name(XQ.shape[0], WQ.shape[0], XQ.shape[1])
    cfg = parse_mfma_kernel_name(kernel_name)
    if cfg is None:
        raise ValueError(
            f"[FlyDSL gfx950 mxfp8_128] unrecognised kernelName: {kernel_name!r}"
        )
    return run_mxfp8_128_preshuffle_gemm_a8_gfx950(
        XQ,
        WQ,
        x_scale,
        w_scale,
        Out,
        cfg["tile_m"],
        cfg["tile_n"],
        cfg["tile_k"],
        waves_per_eu=cfg["waves_per_eu"],
        xcd_swizzle=cfg["xcd_swizzle"],
        scale_is_compact=x_scale.dtype in (torch.uint32, torch.int32),
    )
