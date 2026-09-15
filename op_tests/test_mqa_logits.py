# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""MQA logits (sparse-attention lightning indexer) prefill + decode sweep.

The head geometry is fixed by the model: KV is a single head (the MQA in the
name), head_dim is 128, and Q has 32 or 64 heads. That leaves M (query rows) and
N (KV length) as the axes worth sweeping. Both phases compute

    logits[m, n] = sum_h weights[m, h] * ReLU(<q[m, h, :], k[n, :]>)

inside row m's window and -inf outside.

Prefill and decode are different kernels with different calling conventions, so
they get one table each:

  prefill  fp8_mqa_logits -- contiguous KV [N, 128], Q [M, H, 128], out [M, N].
           M is the chunked-prefill token count, N the context it attends to.
  decode   deepgemm_fp8_paged_mqa_logits -- paged KV cache. Native decode uses
           Q [B, next_n, H, 128]. The current flattened scorer uses
           Q [B*next_n, 1, H, 128], repeated block tables, and per-row lengths.
           Both write B*next_n logical rows.

Q and KV arrive already quantized; neither kernel quantizes. Prefill takes a
plain fp8 cast of Q (its per-(token, head) scale folds into `weights`, which the
kernel multiplies per head) plus an explicit per-token KV scale. Decode reads the
KV scale out of the paged cache, where it is packed behind the data bytes.

The references dequantize the same fp8 bytes the kernels read, so `err` measures
the kernel, not the quantization -- a non-zero `err` column is a real bug.

The DSV4.1 matrix is expressed with the regular sweep axes, for example:

    -m 1 2 3 4 5 6 8 10 12 16 20 24 32 40 48 64 80 96
       128 160 192 256 320 384 512 640 768 1024
    -n 10000 50000 100000 200000 500000 1000000
    -b 1 2 4 8 16 32 64 128 -mtp 1 2 3 4 5 6
    -hq 32 -dh 128 -kb 32 64 128 -c 0 1 --max-model-len 1048576
    --decode-layout native flattened --execution eager graph
"""

import argparse
import itertools

import pandas as pd
import torch

import aiter
from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.shuffle import shuffle_weight
from aiter.ops.triton.attention.fp8_mqa_logits import fp8_mqa_logits
from aiter.ops.triton.attention.pa_mqa_logits import deepgemm_fp8_paged_mqa_logits
from aiter.ops.triton.utils.types import get_fp8_e4m3_dtype
from aiter.test_common import benchmark, checkAllclose, run_perftest

torch.set_default_device("cuda")

SUPPORTED_GFX = ["gfx942", "gfx950"]

# Long-context references sample rows and columns from the quantized cache.
# Exact mask checks scan bounded output slices, so neither path materializes an
# [M, H, N] score tensor or an additional full [M, N] reference/mask tensor.
_REF_MAX_ROWS = 16
_REF_MAX_COLS = 256
_MASK_CHUNK_ELEMENTS = 8 << 20

# Above this argument footprint, run_perftest's automatic argument rotation
# (deep-copies of every input, to defeat L2) costs GiBs and buys nothing,
# because the working set already blows past a 4 MB L2.
_ROTATE_MAX_BYTES = 256 << 20


def _pertoken_cast_to_fp8(x, fp8_dtype):
    """Block-fp8 with block size (1, D): one fp32 scale per token row."""
    amax = x.abs().float().amax(dim=-1, keepdim=True).clamp(1e-4)
    sf = amax / torch.finfo(fp8_dtype).max
    return (x / sf).to(fp8_dtype), sf.squeeze(-1).float()


def _round_up(value, alignment):
    return (value + alignment - 1) // alignment * alignment


def _rotate_args(*tensors):
    tensor_bytes = sum(x.nbytes for x in tensors)
    return 0 if tensor_bytes < _ROTATE_MAX_BYTES else 1


def _sample_indices(length, boundary_sizes=(), limit=_REF_MAX_COLS):
    """Sample the range [0, length), including tile and range boundaries."""
    if length <= 0:
        return torch.empty(0, dtype=torch.long)
    edges = {0, 1, length // 2, length - 2, length - 1}
    for boundary in {64, 256, *boundary_sizes}:
        last_boundary = (length - 1) // boundary * boundary
        for seam in (boundary, last_boundary):
            edges.update((seam - 1, seam, seam + 1))
    edges = {col for col in edges if 0 <= col < length}
    spread_count = min(length, max(0, limit - len(edges)))
    spread = (
        torch.linspace(0, length - 1, steps=spread_count).round().long().tolist()
        if spread_count
        else []
    )
    cols = edges | set(spread)
    return torch.tensor(sorted(cols), dtype=torch.long)


def _ref_rows(m):
    """Query rows to check: grid edges and the BLOCK_M=2 block seam first (a
    row-indexing bug lands there), then an even spread over the rest."""
    want = max(1, min(m, _REF_MAX_ROWS))
    spread = torch.linspace(0, m - 1, steps=want).round().long().tolist()
    rows = []
    for r in [0, 1, m // 2, m // 2 + 1, m - 2, m - 1] + spread:
        if 0 <= r < m and r not in rows:
            rows.append(r)
        if len(rows) >= want:
            break
    return torch.tensor(sorted(rows), dtype=torch.long)


def run_torch_prefill(q, kv, kv_scales, weights, ks, ke, rows):
    """Return sampled fp32 logits and their (row, column) locations."""
    refs = []
    locations = []
    for row in rows.tolist():
        start = int(ks[row])
        end = int(ke[row])
        cols = _sample_indices(end - start) + start
        k = kv[cols].float() * kv_scales[cols, None]
        scores = torch.einsum("hd,nd->hn", q[row].float(), k).relu()
        refs.append((scores * weights[row, :, None]).sum(dim=0))
        locations.append((row, cols))
    return torch.cat(refs), locations


def run_torch_decode(
    q,
    kv_q,
    kv_sf,
    weights,
    context_lens,
    block_tables,
    rows,
    kernel_next_n,
):
    """Return sampled fp32 logits from the fragmented, quantized paged cache."""
    _batch, q_next_n, _heads, _dim = q.shape
    assert q_next_n == kernel_next_n
    block_size = kv_q.shape[1]
    refs = []
    locations = []
    for row in rows.tolist():
        batch_idx, spec_idx = divmod(row, kernel_next_n)
        causal_end = int(context_lens[batch_idx]) - kernel_next_n + spec_idx + 1
        cols = _sample_indices(causal_end, (block_size,))
        logical_blocks = torch.div(cols, block_size, rounding_mode="floor")
        offsets = cols % block_size
        physical_blocks = block_tables[batch_idx, logical_blocks].long()
        k = kv_q[physical_blocks, offsets].float()
        k *= kv_sf[physical_blocks, offsets, None]
        scores = torch.einsum("hd,nd->hn", q[batch_idx, spec_idx], k).relu()
        refs.append((scores * weights[row, :, None]).sum(dim=0))
        locations.append((row, cols))
    return torch.cat(refs), locations


def _sample_output(output, locations):
    return torch.cat([output[row, cols] for row, cols in locations]).float()


def _assert_all(values, message):
    if values.numel() and not bool(values.all()):
        raise AssertionError(message)


def _check_prefill_extent(output, ks, ke, clean_logits, label):
    cols = torch.arange(output.shape[1])[None, :]
    rows_per_chunk = max(1, _MASK_CHUNK_ELEMENTS // output.shape[1])
    for start in range(0, output.shape[0], rows_per_chunk):
        end = min(start + rows_per_chunk, output.shape[0])
        values = output[start:end]
        valid = (cols >= ks[start:end, None]) & (cols < ke[start:end, None])
        _assert_all(torch.isfinite(values[valid]), f"{label}: invalid value")
        if clean_logits:
            _assert_all(torch.isneginf(values[~valid]), f"{label}: bad window mask")


def _check_decode_extent(output, context_lens, kernel_next_n, max_model_len, label):
    """Check exact writes without allocating a dense boolean reference mask."""
    for row in range(output.shape[0]):
        batch_idx, spec_idx = divmod(row, kernel_next_n)
        context_len = int(context_lens[batch_idx])
        causal_end = context_len - kernel_next_n + spec_idx + 1
        untouched_start = min(_round_up(context_len, 256), max_model_len)
        _assert_all(
            torch.isfinite(output[row, :causal_end]),
            f"{label}: non-finite valid logits at row {row}",
        )
        _assert_all(
            torch.isneginf(output[row, causal_end:context_len]),
            f"{label}: causal mask mismatch at row {row}",
        )
        _assert_all(
            torch.isnan(output[row, untouched_start:max_model_len]),
            f"{label}: wrote beyond ChunkK-rounded context at row {row}",
        )


def _capture_graph(fn, args, weights, reset_output=None):
    """Warm JIT, capture only fn, then replay twice with stable tensors.

    Input mutation and optional output poisoning happen outside capture and
    before replay, so replay must read and write the original tensor storage.
    """
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        fn(*args)
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()

    if reset_output is not None:
        reset_output.fill_(float("nan"))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = fn(*args)
    torch.cuda.synchronize()

    weights.add_(0.125)
    for _ in range(2):
        if reset_output is not None:
            reset_output.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
    return graph, output


def _replay_graph(graph, output, *stable_tensors):
    # Graph replay requires the captured pointers, so rotation is disabled.
    graph.replay()
    return output


def _decode_visible_lengths(context_lens, kernel_next_n):
    spec_idx = torch.arange(kernel_next_n, dtype=torch.int64)
    visible = context_lens.to(torch.int64)[:, None] - kernel_next_n + spec_idx + 1
    return visible.flatten()


def _pack_paged_kv(kv_q, kv_sf, preshuffle):
    """Pack fp8 data + fp32 scales into the paged layout the kernel reads: per
    block, block_size*D data bytes followed by block_size fp32 scales."""
    num_blocks, block_size, dim = kv_q.shape
    data = shuffle_weight(kv_q) if preshuffle else kv_q
    packed = torch.empty((num_blocks, block_size * (dim + 4)), dtype=torch.uint8)
    packed[:, : block_size * dim] = data.reshape(num_blocks, -1).view(torch.uint8)
    packed[:, block_size * dim :] = kv_sf.reshape(num_blocks, -1).view(torch.uint8)
    return packed.view(num_blocks, block_size, 1, dim + 4)


@benchmark()
def test_mqa_logits_prefill(m, n, num_heads, head_dim, clean_logits, execution="eager"):
    torch.manual_seed(0)
    fp8_dtype = get_fp8_e4m3_dtype()

    q = torch.randn(m, num_heads, head_dim, dtype=dtypes.bf16)
    kv = torch.randn(n, head_dim, dtype=dtypes.bf16)
    q_fp8 = q.to(fp8_dtype)
    kv_fp8, kv_scales = _pertoken_cast_to_fp8(kv, fp8_dtype)
    weights = torch.randn(m, num_heads, dtype=dtypes.fp32)

    # Chunked prefill: row i is absolute position n-m+i and attends to [0, i].
    ks = torch.zeros(m, dtype=dtypes.i32)
    ke = torch.arange(m, dtype=dtypes.i32) + (n - m) + 1

    def run_prefill(q_arg, kv_arg, scales_arg, weights_arg, ks_arg, ke_arg):
        return fp8_mqa_logits(
            q_arg,
            kv_arg,
            scales_arg,
            weights_arg,
            ks_arg,
            ke_arg,
            clean_logits,
        )

    kernel_args = (q_fp8, kv_fp8, kv_scales, weights, ks, ke)
    if execution == "eager":
        correctness_out = run_prefill(*kernel_args)
        candidates = {
            "triton": (
                run_prefill,
                kernel_args,
                _rotate_args(*kernel_args),
            )
        }
    elif execution == "graph":
        graph, correctness_out = _capture_graph(run_prefill, kernel_args, weights)
        candidates = {
            "triton": (
                _replay_graph,
                (graph, correctness_out, *kernel_args),
                1,
            )
        }
    else:
        raise ValueError(f"unsupported execution mode: {execution}")

    rows = _ref_rows(m)
    ref, locations = run_torch_prefill(q_fp8, kv_fp8, kv_scales, weights, ks, ke, rows)

    valid = int((ke.clamp(max=n) - ks).clamp(min=0).sum().item())
    n_pad = (n + 255) // 256 * 256
    flops = 2.0 * num_heads * head_dim * valid
    read_bytes = q_fp8.nbytes + kv_fp8.nbytes + kv_scales.nbytes + weights.nbytes
    # Compulsory traffic only. KV is re-read per query row, so once N*128 stops
    # fitting in L2 the achieved bandwidth is well above this number.
    nbytes = read_bytes + (m * n_pad * 4 if clean_logits else valid * 4)

    ret = {"gfx": get_gfx(), "ref_rows": len(rows)}
    _check_prefill_extent(
        correctness_out,
        ks,
        ke,
        clean_logits,
        f"triton {execution} prefill m={m} n={n} h={num_heads}",
    )
    for name, (fn, perf_args, num_rotate_args) in candidates.items():
        _, us = run_perftest(fn, *perf_args, num_rotate_args=num_rotate_args)
        err = checkAllclose(
            ref,
            _sample_output(correctness_out, locations),
            rtol=1e-2,
            atol=1e-2,
            msg=f"{name} {execution}: prefill m={m} n={n} h={num_heads}",
        )
        assert err == 0, f"{name} {execution}: prefill mismatch ratio {err}"
        ret[f"{name} us"] = us
        ret[f"{name} TFLOPS"] = flops / us / 1e6
        ret[f"{name} TB/s"] = nbytes / us / 1e6
        ret[f"{name} err"] = err
    return ret


@benchmark()
def test_mqa_logits_decode(
    batch,
    n,
    num_heads,
    head_dim,
    next_n,
    kv_block,
    max_model_len=None,
    execution="eager",
    decode_layout="native",
):
    if head_dim != 128:
        raise ValueError("preshuffled decode only supports head_dim=128")
    if n <= next_n:
        raise ValueError(f"context length {n} must exceed next_n {next_n}")
    if decode_layout not in ("native", "flattened"):
        raise ValueError(f"unsupported decode layout: {decode_layout}")
    if max_model_len is None:
        max_model_len = n
    if max_model_len < n:
        raise ValueError(f"max_model_len {max_model_len} must cover context length {n}")
    max_model_len = _round_up(max_model_len, 256)

    torch.manual_seed(0)
    fp8_dtype = get_fp8_e4m3_dtype()

    rows = batch * next_n
    blocks_per_seq = (n + kv_block - 1) // kv_block
    # Every sequence owns its blocks, so the sweep sees realistic cache pressure
    # rather than the L2 hit rate of a shared block pool.
    num_blocks = batch * blocks_per_seq

    q = torch.randn(batch, next_n, num_heads, head_dim, dtype=dtypes.bf16)
    q_fp8 = q.to(fp8_dtype)
    del q
    if decode_layout == "flattened":
        q_fp8 = q_fp8.view(rows, 1, num_heads, head_dim)
        kernel_next_n = 1
    else:
        kernel_next_n = next_n

    kv = torch.randn(num_blocks * kv_block, head_dim, dtype=dtypes.bf16)
    kv_q, kv_sf = _pertoken_cast_to_fp8(kv, fp8_dtype)
    del kv
    kv_q = kv_q.view(num_blocks, kv_block, head_dim)
    kv_sf = kv_sf.view(num_blocks, kv_block)

    weights = torch.randn(rows, num_heads, dtype=dtypes.fp32)
    # Blocks are handed out in random order: a real KV cache is fragmented, and a
    # sequential block table would hand the kernel a perfectly coalesced stream
    # it never sees in serving.
    base_block_tables = (
        torch.randperm(num_blocks).to(dtypes.i32).view(batch, blocks_per_seq)
    )
    if decode_layout == "flattened":
        context_lens = (
            torch.arange(next_n, dtype=dtypes.i32) + (n - next_n + 1)
        ).repeat(batch)
        block_tables = base_block_tables.repeat_interleave(next_n, dim=0)
    else:
        context_lens = torch.full((batch,), n, dtype=dtypes.i32)
        block_tables = base_block_tables

    # The profiled DSV4.1 decode configuration uses preshuffled paged KV blocks,
    # ChunkK=256, WavePerEU=2, no varctx schedule, and caller-owned output.
    kv_cache = _pack_paged_kv(kv_q, kv_sf, preshuffle=True)
    out_logits = torch.full((rows, max_model_len), float("nan"), dtype=dtypes.fp32)

    def run_decode(
        q_arg,
        cache_arg,
        weights_arg,
        output_arg,
        context_lens_arg,
        block_tables_arg,
    ):
        deepgemm_fp8_paged_mqa_logits(
            q_arg,
            cache_arg,
            weights_arg,
            output_arg,
            context_lens_arg,
            block_tables_arg,
            max_model_len,
            Preshuffle=True,
            KVBlockSize=kv_block,
            ChunkK=256,
            WavePerEU=2,
        )
        return output_arg

    kernel_args = (
        q_fp8,
        kv_cache,
        weights,
        out_logits,
        context_lens,
        block_tables,
    )
    if execution == "eager":
        out_logits.fill_(float("nan"))
        correctness_out = run_decode(*kernel_args)
        candidates = {
            "triton": (
                run_decode,
                kernel_args,
                _rotate_args(*kernel_args),
            )
        }
    elif execution == "graph":
        graph, correctness_out = _capture_graph(
            run_decode, kernel_args, weights, reset_output=out_logits
        )
        candidates = {
            "triton": (
                _replay_graph,
                (graph, correctness_out, *kernel_args),
                1,
            )
        }
    else:
        raise ValueError(f"unsupported execution mode: {execution}")

    sample_rows = _ref_rows(rows)
    ref, locations = run_torch_decode(
        q_fp8.float(),
        kv_q,
        kv_sf,
        weights,
        context_lens,
        block_tables,
        sample_rows,
        kernel_next_n,
    )

    visible_lengths = _decode_visible_lengths(context_lens, kernel_next_n)
    visible_tokens = int(visible_lengths.sum().item())
    written_tokens = int(
        (_round_up(context_lens.to(torch.int64), 256) * kernel_next_n).sum().item()
    )
    flops = 2.0 * num_heads * head_dim * visible_tokens
    read_bytes = q_fp8.nbytes + batch * n * (head_dim + 4) + weights.nbytes
    nbytes = read_bytes + written_tokens * 4

    ret = {"gfx": get_gfx(), "m": rows, "ref_rows": len(sample_rows)}
    _check_decode_extent(
        correctness_out,
        context_lens,
        kernel_next_n,
        max_model_len,
        (f"triton {execution} {decode_layout} decode b={batch} n={n} mtp={next_n}"),
    )
    for name, (fn, perf_args, num_rotate_args) in candidates.items():
        _, us = run_perftest(fn, *perf_args, num_rotate_args=num_rotate_args)
        err = checkAllclose(
            ref,
            _sample_output(correctness_out, locations),
            rtol=1e-2,
            atol=1e-2,
            msg=(
                f"{name} {execution} {decode_layout}: decode "
                f"b={batch} n={n} h={num_heads} mtp={next_n}"
            ),
        )
        assert err == 0, f"{name} {execution} {decode_layout}: decode error {err}"
        ret[f"{name} us"] = us
        ret[f"{name} TFLOPS"] = flops / us / 1e6
        ret[f"{name} TB/s"] = nbytes / us / 1e6
        ret[f"{name} err"] = err
    return ret


def _summarize(name, rows):
    if not rows:
        return
    df = pd.DataFrame(rows)
    aiter.logger.info("%s summary (markdown):\n%s", name, df.to_markdown(index=False))


def main():
    if get_gfx() not in SUPPORTED_GFX:
        aiter.logger.warning("mqa_logits unsupported on %s; skipping", get_gfx())
        return

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="config input of test",
    )
    parser.add_argument(
        "-m",
        "--m",
        type=int,
        nargs="*",
        default=[1024, 2048, 4096, 8192, 16384],
        help="""Prefill query rows (chunked-prefill token count).
    e.g.: -m 4096 8192""",
    )
    parser.add_argument(
        "-n",
        "--n",
        type=int,
        nargs="*",
        default=[4096, 16384, 65664, 131072],
        help="""KV length. Shared by both phases; prefill skips n < m.
    e.g.: -n 8192 131072""",
    )
    parser.add_argument(
        "-hq",
        "--num_heads",
        type=int,
        nargs="*",
        default=[32, 64],
        help="""Q heads. 32 and 64 take different kernel configs.
    e.g.: -hq 64""",
    )
    parser.add_argument(
        "-dh",
        "--head_dim",
        type=int,
        nargs="*",
        default=[128],
        help="""Head dim.
    e.g.: -dh 128""",
    )
    parser.add_argument(
        "-b",
        "--batch",
        type=int,
        nargs="*",
        default=[1, 16, 64, 128, 256],
        help="""Decode batch size (concurrency); decode M = batch * next_n.
    e.g.: -b 128 256""",
    )
    parser.add_argument(
        "-mtp",
        "--next_n",
        type=int,
        nargs="*",
        default=[1],
        help="""Decode speculative rows per sequence.
    e.g.: -mtp 1 2""",
    )
    parser.add_argument(
        "-kb",
        "--kv_block",
        type=int,
        nargs="*",
        default=[64],
        help="""Paged KV block size (preshuffle needs a multiple of 16).
    vLLM uses 32/64; use 128 to reproduce the long-context issue.
    e.g.: -kb 32 64 128""",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        nargs="*",
        default=[],
        help="""Decode output row width. Omitted means ChunkK-rounded context.
    e.g.: --max-model-len 1048576""",
    )
    parser.add_argument(
        "--execution",
        type=str,
        nargs="*",
        choices=["eager", "graph"],
        default=["eager"],
        help="""Execution modes to sweep.
    e.g.: --execution eager graph""",
    )
    parser.add_argument(
        "--decode-layout",
        type=str,
        nargs="*",
        choices=["native", "flattened"],
        default=["native"],
        help="""Decode query/metadata layouts to sweep.
    native: [C, S, H, D]; flattened: [C*S, 1, H, D].
    e.g.: --decode-layout native flattened""",
    )
    parser.add_argument(
        "-c",
        "--clean_logits",
        type=int,
        nargs="*",
        default=[1],
        help="""Prefill: fill the out-of-window logits with -inf in-kernel.
    e.g.: -c 0 1""",
    )
    parser.add_argument(
        "-p",
        "--phase",
        type=str,
        nargs="*",
        choices=["prefill", "decode"],
        default=["prefill", "decode"],
        help="""Which phases to sweep.
    e.g.: -p prefill""",
    )
    args = parser.parse_args()

    if "decode" in args.phase and any(dim != 128 for dim in args.head_dim):
        parser.error("preshuffled decode only supports --head_dim 128")

    if "prefill" in args.phase:
        rows = []
        for num_heads, head_dim, clean, execution, m, n in itertools.product(
            args.num_heads,
            args.head_dim,
            args.clean_logits,
            args.execution,
            args.m,
            args.n,
        ):
            if n < m:
                continue  # a prefill chunk always attends to at least itself
            rows.append(
                test_mqa_logits_prefill(
                    m,
                    n,
                    num_heads,
                    head_dim,
                    bool(clean),
                    execution,
                )
            )
            torch.cuda.empty_cache()
        _summarize("mqa_logits prefill", rows)

    if "decode" in args.phase:
        rows = []
        model_lens = args.max_model_len or [None]
        for (
            num_heads,
            head_dim,
            next_n,
            kv_block,
            execution,
            decode_layout,
            batch,
            n,
            max_model_len,
        ) in itertools.product(
            args.num_heads,
            args.head_dim,
            args.next_n,
            args.kv_block,
            args.execution,
            args.decode_layout,
            args.batch,
            args.n,
            model_lens,
        ):
            if n <= next_n:
                continue
            if max_model_len is not None and max_model_len < n:
                continue
            output_width = _round_up(n if max_model_len is None else max_model_len, 256)
            rows.append(
                test_mqa_logits_decode(
                    batch,
                    n,
                    num_heads,
                    head_dim,
                    next_n,
                    kv_block,
                    output_width,
                    execution,
                    decode_layout,
                )
            )
            torch.cuda.empty_cache()
        _summarize("mqa_logits decode", rows)


if __name__ == "__main__":
    main()
