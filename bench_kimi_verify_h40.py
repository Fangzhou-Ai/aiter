#!/usr/bin/env python3
"""Correctness and timing harness for the standalone H40 verify prototype."""

from __future__ import annotations

import argparse
import ctypes
import math
import os
import statistics
import sys

import torch


ROOT = os.path.dirname(os.path.abspath(__file__))
SO = os.environ.get(
    "KIMI_H40_SO",
    os.path.join(ROOT, "op_tests", "opus", "device", "kimi_verify_h40.so"),
)
FP8 = torch.float8_e4m3fn
LOG2E = 1.4426950408889634


class H40Verify:
    def __init__(self, so_path: str = SO):
        self.lib = ctypes.CDLL(so_path)
        self.run = self.lib.kimi_verify_h40_run
        vp = ctypes.c_void_p
        self.run.argtypes = [
            vp,
            vp,
            vp,
            vp,
            vp,
            vp,
            vp,
            vp,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_float,
            ctypes.c_float,
            vp,
        ]
        self.run.restype = ctypes.c_int
        self.partial = self.lib.kimi_verify_h40_partial
        self.partial.argtypes = [
            vp,
            vp,
            vp,
            vp,
            vp,
            vp,
            vp,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_float,
            vp,
        ]
        self.partial.restype = ctypes.c_int
        self.reduce = self.lib.kimi_verify_h40_reduce
        self.reduce.argtypes = [
            vp,
            vp,
            vp,
            vp,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_float,
            vp,
        ]
        self.reduce.restype = ctypes.c_int
        self.partial_range = self.lib.kimi_verify_h40_partial_range
        self.partial_range.argtypes = [
            vp,
            vp,
            vp,
            vp,
            vp,
            vp,
            vp,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_float,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            vp,
        ]
        self.partial_range.restype = ctypes.c_int

    @staticmethod
    def ptr(x: torch.Tensor) -> ctypes.c_void_p:
        return ctypes.c_void_p(x.data_ptr())

    def __call__(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        indptr: torch.Tensor,
        indices: torch.Tensor,
        part_m: torch.Tensor,
        part_l: torch.Tensor,
        part_acc: torch.Tensor,
        out: torch.Tensor,
        *,
        qlen: int,
        nhead: int,
        splits: int,
        qk_scale_log2: float,
        output_scale: float,
    ) -> None:
        err = self.run(
            self.ptr(q),
            self.ptr(kv),
            self.ptr(indptr),
            self.ptr(indices),
            self.ptr(part_m),
            self.ptr(part_l),
            self.ptr(part_acc),
            self.ptr(out),
            q.shape[0],
            qlen,
            nhead,
            kv.shape[0],
            splits,
            qk_scale_log2,
            output_scale,
            ctypes.c_void_p(torch.cuda.current_stream().cuda_stream),
        )
        if err:
            raise RuntimeError(f"kimi_verify_h40_run failed with HIP error {err}")

    def run_partial(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        indptr: torch.Tensor,
        indices: torch.Tensor,
        part_m: torch.Tensor,
        part_l: torch.Tensor,
        part_acc: torch.Tensor,
        *,
        qlen: int,
        nhead: int,
        splits: int,
        qk_scale_log2: float,
    ) -> None:
        err = self.partial(
            self.ptr(q),
            self.ptr(kv),
            self.ptr(indptr),
            self.ptr(indices),
            self.ptr(part_m),
            self.ptr(part_l),
            self.ptr(part_acc),
            q.shape[0],
            qlen,
            nhead,
            kv.shape[0],
            splits,
            qk_scale_log2,
            ctypes.c_void_p(torch.cuda.current_stream().cuda_stream),
        )
        if err:
            raise RuntimeError(f"kimi_verify_h40_partial failed with HIP error {err}")

    def run_reduce(
        self,
        part_m: torch.Tensor,
        part_l: torch.Tensor,
        part_acc: torch.Tensor,
        out: torch.Tensor,
        *,
        qlen: int,
        nhead: int,
        splits: int,
        output_scale: float,
    ) -> None:
        err = self.reduce(
            self.ptr(part_m),
            self.ptr(part_l),
            self.ptr(part_acc),
            self.ptr(out),
            out.shape[0],
            qlen,
            nhead,
            splits,
            output_scale,
            ctypes.c_void_p(torch.cuda.current_stream().cuda_stream),
        )
        if err:
            raise RuntimeError(f"kimi_verify_h40_reduce failed with HIP error {err}")

    def run_partial_range(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        indptr: torch.Tensor,
        indices: torch.Tensor,
        part_m: torch.Tensor,
        part_l: torch.Tensor,
        part_acc: torch.Tensor,
        *,
        qlen: int,
        nhead: int,
        splits: int,
        qk_scale_log2: float,
        h_offset: int,
        h_blocks: int,
        rows_pad: int,
        stream: torch.cuda.Stream,
    ) -> None:
        err = self.partial_range(
            self.ptr(q),
            self.ptr(kv),
            self.ptr(indptr),
            self.ptr(indices),
            self.ptr(part_m),
            self.ptr(part_l),
            self.ptr(part_acc),
            q.shape[0],
            qlen,
            nhead,
            kv.shape[0],
            splits,
            qk_scale_log2,
            h_offset,
            h_blocks,
            rows_pad,
            ctypes.c_void_p(stream.cuda_stream),
        )
        if err:
            raise RuntimeError(
                f"kimi_verify_h40_partial_range failed with HIP error {err}"
            )


def quantize(x: torch.Tensor) -> tuple[torch.Tensor, float]:
    scale = max(float(x.abs().max().item()) / 448.0, 1.0e-8)
    return (x / scale).to(FP8), scale


def make_case(
    t: int,
    qlen: int,
    nhead: int,
    kv_len: int,
    splits: int,
    *,
    lengths: list[int] | None = None,
    scale_mode: str = "amax",
):
    torch.manual_seed(7)
    dev = torch.device("cuda")
    if lengths is None:
        lengths = [kv_len] * t
    if len(lengths) != t:
        raise ValueError(f"expected {t} KV lengths, got {len(lengths)}")
    q_real = torch.randn(t, qlen, nhead, 576, device=dev) * 0.35
    kv_real = torch.randn(sum(lengths), 576, device=dev) * 0.35
    if scale_mode == "unit":
        q, qs = q_real.to(FP8), 1.0
        kv, ks = kv_real.to(FP8), 1.0
    elif scale_mode == "amax":
        q, qs = quantize(q_real)
        kv, ks = quantize(kv_real)
    else:
        raise ValueError(f"unknown scale_mode={scale_mode!r}")

    indptr = torch.tensor(
        [0, *torch.tensor(lengths).cumsum(0).tolist()],
        dtype=torch.int32,
        device=dev,
    )
    indices = torch.arange(sum(lengths), dtype=torch.int32, device=dev)

    rows = qlen * nhead
    rows_pad = math.ceil(rows / 128) * 128
    part_m = torch.empty((t, splits, rows_pad), dtype=torch.float32, device=dev)
    part_l = torch.empty_like(part_m)
    part_acc = torch.empty(
        (t, splits, rows_pad, 512), dtype=torch.float32, device=dev
    )
    out = torch.empty((t, qlen, nhead, 512), dtype=torch.bfloat16, device=dev)
    return {
        "q_real": q_real,
        "kv_real": kv_real,
        "q": q,
        "kv": kv,
        "qs": qs,
        "ks": ks,
        "indptr": indptr,
        "indices": indices,
        "part_m": part_m,
        "part_l": part_l,
        "part_acc": part_acc,
        "out": out,
    }


@torch.inference_mode()
def reference(case: dict, qlen: int, nhead: int, kv_len: int) -> torch.Tensor:
    q = case["q"].float() * case["qs"]
    kv = case["kv"].float() * case["ks"]
    scale = 1.0 / math.sqrt(576)
    refs = []
    for req in range(q.shape[0]):
        kr = kv[req * kv_len : (req + 1) * kv_len]
        per_pos = []
        for qpos in range(qlen):
            valid = kv_len - (qlen - 1) + qpos
            score = q[req, qpos].float() @ kr[:valid].float().T
            p = torch.softmax(score * scale, dim=-1)
            per_pos.append((p @ kr[:valid, :512].float()).to(torch.bfloat16))
        refs.append(torch.stack(per_pos))
    return torch.stack(refs)


def timed(fn, warmup: int, iters: int) -> list[float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    values = []
    for _ in range(iters):
        begin = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        begin.record()
        fn()
        end.record()
        end.synchronize()
        values.append(begin.elapsed_time(end) * 1000.0)
    return sorted(values)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--t", type=int, default=1)
    p.add_argument("--q", type=int, default=7)
    p.add_argument("--h", type=int, default=24)
    p.add_argument("--kv", type=int, default=512)
    p.add_argument("--splits", type=int, default=4)
    p.add_argument("--warmup", type=int, default=8)
    p.add_argument("--iters", type=int, default=33)
    p.add_argument("--no-ref", action="store_true")
    p.add_argument(
        "--timing",
        choices=("full", "partial", "reduce"),
        default="full",
    )
    args = p.parse_args()

    max_splits = max(1, math.ceil(args.kv / 128))
    splits = min(args.splits, max_splits)
    case = make_case(args.t, args.q, args.h, args.kv, splits)
    op = H40Verify()
    sm = 1.0 / math.sqrt(576)

    def run():
        op(
            case["q"].view(args.t, args.q * args.h, 576),
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

    def run_partial():
        op.run_partial(
            case["q"].view(args.t, args.q * args.h, 576),
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

    def run_reduce():
        op.run_reduce(
            case["part_m"],
            case["part_l"],
            case["part_acc"],
            case["out"],
            qlen=args.q,
            nhead=args.h,
            splits=splits,
            output_scale=case["ks"],
        )

    run()
    torch.cuda.synchronize()
    got = case["out"].float()
    print(
        f"shape T={args.t} Q={args.q} H={args.h} KV={args.kv} "
        f"splits={splits} nan={int(torch.isnan(got).sum())} "
        f"abs_sum={float(got.abs().sum()):.6e}"
    )
    if not args.no_ref:
        ref = reference(case, args.q, args.h, args.kv).float()
        rel = float((got - ref).norm() / ref.norm())
        max_abs = float((got - ref).abs().max())
        print(f"correctness rel_l2={rel:.6e} max_abs={max_abs:.6e}")

    timing_fn = {
        "full": run,
        "partial": run_partial,
        "reduce": run_reduce,
    }[args.timing]
    values = timed(timing_fn, args.warmup, args.iters)
    print(
        f"latency[{args.timing}] min={values[0]:.1f} "
        f"med={statistics.median(values):.1f} "
        f"p95={values[int(0.95 * (len(values) - 1))]:.1f} us"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

