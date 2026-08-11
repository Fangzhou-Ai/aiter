#!/usr/bin/env python3
"""Interleaved H40-vs-ASM target-verify comparison on identical tensors."""

from __future__ import annotations

import argparse
import math
import os
import statistics
import sys

import torch

from bench_kimi_verify_h40 import H40Verify, LOG2E, make_case


PROD_AITER = "/home/wenwzhan/kimi3-aiter-kk3-dev-bytedance"
sys.path.insert(0, PROD_AITER)

from aiter import dtypes, get_mla_metadata_v1  # noqa: E402
from aiter.mla import mla_decode_fwd  # noqa: E402
from aiter.ops.attention import get_mla_metadata_info_v1  # noqa: E402


ROOT = os.path.dirname(os.path.abspath(__file__))
QS1 = os.path.join(
    ROOT, "op_tests", "opus", "device", "kimi_verify_h40_qs1.so"
)
QS2 = os.path.join(
    ROOT, "op_tests", "opus", "device", "kimi_verify_h40_qs2.so"
)
REAL_100K_LENGTHS = [
    100000,
    100000,
    100001,
    100001,
    100000,
    100000,
    99999,
    99999,
]


def timed(fn, warmup=8, iters=43, batch=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    values = []
    for _ in range(iters):
        begin = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        begin.record()
        for _ in range(batch):
            fn()
        end.record()
        end.synchronize()
        values.append(begin.elapsed_time(end) * 1000.0 / batch)
    return sorted(values)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--t", type=int, default=8)
    p.add_argument("--q", type=int, default=7)
    p.add_argument("--h", type=int, default=24)
    p.add_argument("--kv", type=int, default=100000)
    p.add_argument(
        "--lens",
        help="comma-separated ragged KV lengths; count must equal --t",
    )
    p.add_argument(
        "--scale-mode",
        choices=("amax", "unit"),
        default="amax",
    )
    p.add_argument(
        "--real-preset",
        action="store_true",
        help="use re-tokenized ShareGPT 100K lengths and production unit scales",
    )
    p.add_argument("--splits", type=int)
    p.add_argument("--h40-so")
    p.add_argument("--iters", type=int, default=43)
    p.add_argument("--breakdown", action="store_true")
    p.add_argument("--include-flydsl", action="store_true")
    args = p.parse_args()

    if args.real_preset:
        args.q = 8
        args.h = 32
        defaults = {
            1: (QS2, 128),
            2: (QS2, 64),
            4: (QS2, 32),
            8: (QS2, 16),
        }
    else:
        defaults = {1: (QS1, 80), 2: (QS1, 43), 4: (QS1, 21), 8: (QS2, 16)}
    so, default_splits = defaults.get(args.t, (QS2, 16))
    if args.h40_so:
        so = args.h40_so
    splits = args.splits or default_splits
    if args.real_preset:
        if args.t > len(REAL_100K_LENGTHS):
            raise ValueError("real preset contains at most 8 requests per DP rank")
        lengths = REAL_100K_LENGTHS[: args.t]
        args.scale_mode = "unit"
    else:
        lengths = (
            [int(x) for x in args.lens.split(",")]
            if args.lens
            else [args.kv] * args.t
        )
    case = make_case(
        args.t,
        args.q,
        args.h,
        args.kv,
        splits,
        lengths=lengths,
        scale_mode=args.scale_mode,
    )
    h40 = H40Verify(so)
    q_view = case["q"].view(args.t, args.q * args.h, 576)
    sm = 1.0 / math.sqrt(576)

    def run_h40():
        h40(
            q_view,
            case["kv"],
            case["indptr"],
            case["indices"],
            case["part_m"],
            case["part_l"],
            case["part_acc"],
            case["out"],
            qlen=args.q,
            nhead=args.h,
            splits=splits,
            qk_scale_log2=sm * case["qs"] * case["ks"] * LOG2E,
            output_scale=case["ks"],
        )

    def run_h40_partial():
        h40.run_partial(
            q_view,
            case["kv"],
            case["indptr"],
            case["indices"],
            case["part_m"],
            case["part_l"],
            case["part_acc"],
            qlen=args.q,
            nhead=args.h,
            splits=splits,
            qk_scale_log2=sm * case["qs"] * case["ks"] * LOG2E,
        )

    def run_h40_reduce():
        h40.run_reduce(
            case["part_m"],
            case["part_l"],
            case["part_acc"],
            case["out"],
            qlen=args.q,
            nhead=args.h,
            splits=splits,
            output_scale=case["ks"],
        )

    # asm ships qh8/16/32/64/128; 40 real heads has to be padded up to 64,
    # which is the whole reason a native h40 kernel is worth having.
    nhead_pad = next(n for n in (8, 16, 32, 64, 128) if n >= args.h)
    q_pad = torch.zeros(
        (args.t * args.q, nhead_pad, 576),
        dtype=case["q"].dtype,
        device=case["q"].device,
    )
    q_pad[:, : args.h].copy_(case["q"].view(args.t * args.q, args.h, 576))
    out_asm = torch.zeros(
        (args.t * args.q, nhead_pad, 512),
        dtype=torch.bfloat16,
        device=case["q"].device,
    )
    out_fly = torch.zeros_like(out_asm)
    qo = torch.arange(
        0,
        (args.t + 1) * args.q,
        args.q,
        dtype=torch.int32,
        device=case["q"].device,
    )
    last = torch.ones(args.t, dtype=torch.int32, device=case["q"].device)
    qs_t = torch.tensor([case["qs"]], dtype=torch.float32, device=case["q"].device)
    ks_t = torch.tensor([case["ks"]], dtype=torch.float32, device=case["q"].device)

    sizes = get_mla_metadata_info_v1(
        args.t,
        args.q,
        nhead_pad,
        dtypes.fp8,
        dtypes.fp8,
        is_sparse=False,
        fast_mode=True,
    )
    meta, work_indptr, info_set, red_indptr, red_final, red_partial = [
        torch.empty(size, dtype=dtype, device=case["q"].device)
        for size, dtype in sizes
    ]
    get_mla_metadata_v1(
        qo,
        case["indptr"],
        last,
        nhead_pad,
        1,
        True,
        work_metadata_ptrs=meta,
        work_info_set=info_set,
        work_indptr=work_indptr,
        reduce_indptr=red_indptr,
        reduce_final_map=red_final,
        reduce_partial_map=red_partial,
        page_size=1,
        kv_granularity=16,
        max_seqlen_qo=args.q,
        uni_seqlen_qo=args.q,
        fast_mode=True,
        dtype_q=dtypes.fp8,
        dtype_kv=dtypes.fp8,
    )
    metadata = dict(
        work_meta_data=meta,
        work_indptr=work_indptr,
        work_info_set=info_set,
        reduce_indptr=red_indptr,
        reduce_final_map=red_final,
        reduce_partial_map=red_partial,
    )

    os.environ["AITER_MLA_FLYDSL"] = "0"

    def run_asm():
        mla_decode_fwd(
            q_pad,
            case["kv"].view(-1, 1, 1, 576),
            out_asm,
            qo,
            case["indptr"],
            case["indices"],
            last,
            args.q,
            page_size=1,
            nhead_kv=1,
            sm_scale=sm,
            q_scale=qs_t,
            kv_scale=ks_t,
            **metadata,
        )

    def run_fly():
        mla_decode_fwd(
            case["q"].view(args.t * args.q, args.h, 576),
            case["kv"].view(-1, 1, 1, 576),
            out_fly,
            qo,
            case["indptr"],
            case["indices"],
            last,
            args.q,
            page_size=1,
            nhead_kv=1,
            sm_scale=sm,
            q_scale=qs_t,
            kv_scale=ks_t,
        )

    run_h40()
    run_asm()
    torch.cuda.synchronize()
    got = case["out"].view(args.t * args.q, args.h, 512).float()
    ref = out_asm[:, : args.h].float()
    rel = float((got - ref).norm() / ref.norm())
    max_abs = float((got - ref).abs().max())

    # Interleave two rounds so thermal/clock drift reaches both kernels.
    h40_times = []
    asm_times = []
    for _ in range(2):
        h40_times.extend(timed(run_h40, iters=args.iters))
        asm_times.extend(timed(run_asm, iters=args.iters))
    h40_times.sort()
    asm_times.sort()
    print(
        f"T={args.t} Q={args.q} H={args.h} "
        f"KV={min(lengths)}..{max(lengths)} splits={splits} "
        f"scales={args.scale_mode} "
        f"H40={os.path.basename(so)} rel={rel:.6e} max_abs={max_abs:.6e}"
    )
    print(
        f"H40 min={h40_times[0]:.1f} med={statistics.median(h40_times):.1f} us  "
        f"ASM min={asm_times[0]:.1f} med={statistics.median(asm_times):.1f} us  "
        f"ratio(min)={h40_times[0]/asm_times[0]:.3f}x"
    )
    if args.include_flydsl:
        os.environ["AITER_MLA_FLYDSL"] = "1"
        os.environ["AITER_MLA_RAW_FP8_QK"] = "1"
        os.environ["AITER_MLA_TARGET_RAW_FP8_PV"] = "0"
        fly_times = timed(run_fly, iters=args.iters)
        fly = out_fly[:, : args.h].float()
        fly_rel = float((got - fly).norm() / fly.norm())
        print(
            f"FlyDSL min={fly_times[0]:.1f} "
            f"med={statistics.median(fly_times):.1f} us  "
            f"H40/FlyDSL={h40_times[0]/fly_times[0]:.3f}x "
            f"rel={fly_rel:.6e}"
        )
        os.environ["AITER_MLA_FLYDSL"] = "0"
    if args.breakdown:
        partial = timed(run_h40_partial, iters=args.iters)
        reduce = timed(run_h40_reduce, iters=args.iters)
        print(
            f"H40 breakdown partial={partial[0]:.1f} us "
            f"reduce={reduce[0]:.1f} us"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

