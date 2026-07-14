"""Per-M gemm1/gemm2 launch comparison for dsv4 a8w4 (fp8 a / fp4 w):

  baseline = current vendored FlyDSL mixed_moe stage1 (aiter.ops.flydsl.flydsl_moe_stage1)
  v2       = FlyDSL#753 "mxmoe v2" gemm1 (aiter.ops.flydsl.kernels.mxmoe_dispatcher.mxfp4_moe_gemm1)

Both kernels compute the same a8w4 up/gate MoE gemm but through different input
pipelines (baseline: aiter sort + unsorted a-scale, gather+fused-quant-out; v2:
opus sort + sorted/shuffled mxfp8 a-scale). We time ONLY each kernel's launch in
isolation (identical run_perftest settings), so the differing input prep does not
affect the comparison -- mirroring test_moe_2stage.py --kernel.

Baseline config per M comes from the tuned CSV's kernelName1 (parsed via
get_flydsl_kernel_params); v2 config per M comes from the #753 dispatcher's own
select_pipe_config / gemm1_use_nt (its production choice).

Usage:
  /opt/venv/bin/python bench_gemm1_v2_vs_baseline.py \
      --csv aiter/configs/model_configs/dsv4_fp8fp4_tuned_fmoe.csv \
      --model-dim 7168 --inter-dim 512 -E 384 -k 6

  /opt/venv/bin/python bench_gemm1_v2_vs_baseline.py --stage gemm2 \
      --csv aiter/configs/model_configs/dsv4_fp8fp4_tuned_fmoe.csv \
      --model-dim 7168 --inter-dim 512 -E 384 -k 6
"""

import argparse
import csv as _csv
import os

import torch

import aiter
from aiter import dtypes
from aiter.fused_moe import fused_topk
from aiter.utility import fp4_utils
from aiter.test_common import run_perftest
from aiter.ops.flydsl.moe_kernels import (
    flydsl_moe_stage1,
    flydsl_moe_stage2,
    get_flydsl_kernel_params,
    flydsl_kernel_name,
    _run_moe_reduction,
)
from aiter.fused_moe import parse_flydsl_v2_gemm2_kernel
from aiter.ops.flydsl.kernels.mxmoe_dispatcher import (
    mxfp4_moe_gemm1,
    mxfp4_moe_gemm2,
    select_pipe_config,
    select_gemm2_config,
    gemm1_use_nt,
    gemm2_use_nt,
)
from aiter.ops.opus import moe_stage2_a8w4_fused_adapter as _opus_a8w4
from aiter.ops.flydsl.mxfp4_v2_tune_utils import (
    balanced_score, quant_a_fp8, quant_a_fp4, quant_a, quant_w_fp4,
    _a_deq, _stage2_quant_sort, _dequant_inter_sorted_quant, _baseline_w1_shuffle,
    _mxfp4_shuffle_weight_a16w4, _mxfp4_shuffle_scale_a16w4,
    _mxfp4_a_scale_sorted_shuffled, _u8v, gen, build_v2_inputs,
    populate_baseline_v2_intermediate, _v2_group_cosine,
)

torch.set_default_device("cuda")

WARMUP, ITERS = 10, 50


def time_baseline(d, token, topk, params):
    b = d["base"]
    adtype = b.get("adtype", "fp8")
    default_gate = "interleave" if adtype == "fp8" else "separated"
    def fn():
        return flydsl_moe_stage1(
            a=b["a1_qt"], w1=b["w1_qt_shuf"],
            sorted_token_ids=b["sorted_ids"],
            sorted_expert_ids=b["sorted_expert_ids"],
            num_valid_ids=b["num_valid_ids"], topk=topk,
            tile_m=params["tile_m"], tile_n=params["tile_n"], tile_k=params["tile_k"],
            a_dtype=adtype, b_dtype="fp4", out_dtype="bf16",
            w1_scale=b["w1_scale_shuf"], a1_scale=b["a1_scale_sort"],
            sorted_weights=None,
            k_batch=params.get("k_batch", 1),
            waves_per_eu=params.get("waves_per_eu", 3),
            gate_mode=params.get("gate_mode", default_gate),
            b_nt=params.get("b_nt", 2),
            k_wave=params.get("k_wave", 1),
        )
    out = fn()
    if isinstance(out, tuple):
        out = out[0]
    torch.cuda.synchronize()
    ref = d["ref1"]
    o = out.float().reshape(ref.shape)
    ok = torch.isclose(ref.float(), o, atol=1.0, rtol=0.05).float().mean().item() * 100
    _, us = run_perftest(fn, num_warmup=WARMUP, num_iters=ITERS)
    return us, ok


def _print_tensor(name, tensor):
    torch.cuda.synchronize()
    print(f"\n{name}:")
    print(tensor.detach().cpu())


def _print_close_stats(name, ref, got, atol=1.0, rtol=0.05):
    ref_f = ref.float()
    got_f = got.float()
    diff = (ref_f - got_f).abs()
    close = torch.isclose(ref_f, got_f, atol=atol, rtol=rtol).float().mean().item() * 100
    print(
        f"\n{name} diff stats: "
        f"close={close:.2f}% atol={atol} rtol={rtol} "
        f"max_abs={diff.max().item():.6g} mean_abs={diff.mean().item():.6g}"
    )


def time_baseline_gemm2(d, token, model_dim, topk, params, row=None, print_output=False):
    b = d["base"]
    row = row or {}
    kn2 = row.get("kernelName2", "")
    is_opus2 = _opus_a8w4.is_opus_a8w4_stage2_kernel(kn2)
    mode = params.get("mode", "atomic")
    stage2_adtype = b.get("a2_dtype", params.get("a_dtype", b.get("adtype", "fp8")))
    # gemm2 final output is always [token, model_dim]. flydsl reduce mode writes a
    # per-slot [token, topk, model_dim] buffer that _run_moe_reduction reduces below
    # (timed together, mirroring the v2 path); opus route mode reduces in its wrapper.
    out = torch.empty((token, model_dim), dtype=dtypes.bf16, device="cuda")

    if is_opus2:
        opus_values = _opus_a8w4.stage2_cfg_values(row, row.get("block_m", params["tile_m"]))

        def fn():
            return _opus_a8w4.opus_a8w4_stage2_wrapper(
                inter_states=b["a2_qt"],
                w1=None,
                w2=b["w2_qt_shuf"],
                sorted_token_ids=b["sorted_ids"],
                sorted_expert_ids=b["sorted_expert_ids"],
                num_valid_ids=b["num_valid_ids"],
                out=out,
                topk=topk,
                kernelName=kn2,
                w2_scale=b["w2_scale_shuf"].view(dtypes.fp8_e8m0),
                a2_scale=b["a2_scale"],
                sorted_weights=b["sorted_weights"],
                block_m=int(row.get("block_m", params["tile_m"])),
                **opus_values,
            )

        out.zero_()
        fn()
        torch.cuda.synchronize()
        ref = d["ref2"].float()
        got = out.float()
        if print_output:
            _print_close_stats("gemm2 baseline vs torch ref", ref, got)
            _print_tensor("torch ref gemm2 output", ref)
            _print_tensor("baseline gemm2 output", got)
        ok = torch.isclose(ref, got, atol=1.0, rtol=0.05).float().mean().item() * 100
        _, us = run_perftest(fn, num_warmup=WARMUP, num_iters=ITERS)
        return us, ok

    is_reduce = mode == "reduce"
    inter = (
        torch.empty((token, topk, model_dim), dtype=dtypes.bf16, device="cuda")
        if is_reduce
        else None
    )
    gemm2_out = inter if is_reduce else out

    def fn():
        flydsl_moe_stage2(
            inter_states=b["a2_qt"],
            w2=b["w2_qt_shuf"],
            sorted_token_ids=b["sorted_ids"],
            sorted_expert_ids=b["sorted_expert_ids"],
            num_valid_ids=b["num_valid_ids"],
            out=gemm2_out,
            topk=topk,
            tile_m=params["tile_m"],
            tile_n=params["tile_n"],
            tile_k=params["tile_k"],
            a_dtype=stage2_adtype,
            b_dtype=params.get("b_dtype", "fp4"),
            out_dtype=params.get("out_dtype", "bf16"),
            mode=mode,
            w2_scale=b["w2_scale_shuf"].view(dtypes.fp8_e8m0),
            a2_scale=b["a2_scale"],
            sorted_weights=b["sorted_weights"],
            sort_block_m=params.get("sort_block_m", 0),
            persist=params.get("persist", None),
            waves_per_eu=params.get("waves_per_eu", None),
            b_nt=params.get("b_nt", 0),
            xcd_swizzle=params.get("xcd_swizzle", 0),
            return_per_slot=is_reduce,
        )
        if is_reduce:
            _run_moe_reduction(inter, out, token, topk, model_dim)
        return out

    if not is_reduce:
        out.zero_()
    fn()
    torch.cuda.synchronize()
    ref = d["ref2"].float()
    got = out.float()
    if print_output:
        _print_close_stats("gemm2 baseline vs torch ref", ref, got)
        _print_tensor("torch ref gemm2 output", ref)
        _print_tensor("baseline gemm2 output", got)
    ok = torch.isclose(ref, got, atol=1.0, rtol=0.05).float().mean().item() * 100
    _, us = run_perftest(fn, num_warmup=WARMUP, num_iters=ITERS)
    return us, ok


def time_v2(d, v, token, model_dim, inter_dim, E, topk, BM_S1, use_nt, BN, k_wave):
    adtype = d.get("adtype", "fp8")
    stage2_adtype = d.get("stage2_adtype", adtype)
    def fn():
        return mxfp4_moe_gemm1(
            a_quant=v["aq"], a_scale_sorted_shuffled=v["assh"],
            w1_u8=v["w1u8"], w1_scale_u8=v["w1sc"],
            sorted_expert_ids=v["sei"], cumsum_tensor=v["cumsum"],
            sorted_token_ids=v["sti"], inter_sorted_quant=v["isq"],
            inter_sorted_shuffled_scale=v["iss"], hidden_states=v["hidden"],
            n_tokens=token, NE=E, D_HIDDEN=model_dim, D_INTER=inter_dim, topk=topk,
            BM=BM_S1, use_nt=use_nt, interleave=True,
            a_dtype=adtype, out_dtype=stage2_adtype, act="silu", swiglu_limit=0.0,
            SBM=BM_S1, k_wave=k_wave, BN=BN, n_sorted_padded=v["n"],
            model_dim_pad=0, inter_dim_pad=0,
        )
    v["isq"].zero_()
    fn()
    torch.cuda.synchronize()
    ok = _v2_group_cosine(d, v, token, inter_dim, E, BM_S1)
    _, us = run_perftest(fn, num_warmup=WARMUP, num_iters=ITERS)
    return us, ok


def print_gemm1_v2_layout_compare(d, v, token, model_dim, inter_dim, E, topk,
                                  BM_S1, use_nt, BN, k_wave, base_gemm1_params):
    populate_baseline_v2_intermediate(d, v, token, topk, base_gemm1_params, BM_S1)
    baseline_isq = v["isq"].clone()

    v["isq"].zero_()
    v["iss"].zero_()
    mxfp4_moe_gemm1(
        a_quant=v["aq"], a_scale_sorted_shuffled=v["assh"],
        w1_u8=v["w1u8"], w1_scale_u8=v["w1sc"],
        sorted_expert_ids=v["sei"], cumsum_tensor=v["cumsum"],
        sorted_token_ids=v["sti"], inter_sorted_quant=v["isq"],
        inter_sorted_shuffled_scale=v["iss"], hidden_states=v["hidden"],
        n_tokens=token, NE=E, D_HIDDEN=model_dim, D_INTER=inter_dim, topk=topk,
        BM=BM_S1, use_nt=use_nt, interleave=True,
        a_dtype=d.get("adtype", "fp8"), out_dtype=d.get("stage2_adtype", d.get("adtype", "fp8")),
        act="silu", swiglu_limit=0.0,
        SBM=BM_S1, k_wave=k_wave, BN=BN, n_sorted_padded=v["n"],
        model_dim_pad=0, inter_dim_pad=0,
    )
    torch.cuda.synchronize()

    stage2_adtype = d.get("stage2_adtype", d.get("adtype", "fp8"))
    baseline = torch.stack(
        [_dequant_inter_sorted_quant(row, inter_dim, stage2_adtype) for row in baseline_isq]
    )
    v2 = torch.stack(
        [_dequant_inter_sorted_quant(row, inter_dim, stage2_adtype) for row in v["isq"]]
    )
    n_valid = v["n"]
    _print_close_stats(
        "gemm1 baseline-v2-layout vs v2 full isq",
        baseline, v2, atol=0.0, rtol=0.0,
    )
    _print_close_stats(
        "gemm1 baseline-v2-layout vs v2 valid isq",
        baseline[:n_valid], v2[:n_valid], atol=0.0, rtol=0.0,
    )
    _print_tensor(f"baseline gemm1 v2-layout {stage2_adtype} output", baseline)
    _print_tensor(f"v2 gemm1 sorted {stage2_adtype} output", v2)


def time_v2_gemm2(d, v, token, model_dim, inter_dim, E, topk, BM_S1, BM_S2, use_nt,
                  epilog, persist, BN, k_wave, base_gemm1_params=None,
                  print_output=False, use_baseline_producer=False):
    stage2_adtype = d.get("stage2_adtype", d.get("adtype", "fp8"))
    if use_baseline_producer:
        # Baseline gemm1 (mixed_moe_gemm_2stage) as the intermediate producer.
        # NOTE: this kernel is incompatible with FLYDSL_DEBUG_ENABLE_DEBUG_INFO=1
        # (manual cross-statement InsertionPoint -> "Unbalanced InsertionPoint").
        if base_gemm1_params is None:
            raise ValueError("base_gemm1_params is required for FlyDSL v2 layout")
        populate_baseline_v2_intermediate(d, v, token, topk, base_gemm1_params, BM_S1)
    else:
        # Populate the sorted fp8 intermediate exactly as v2 production gemm2 consumes it.
        mxfp4_moe_gemm1(
            a_quant=v["aq"], a_scale_sorted_shuffled=v["assh"],
            w1_u8=v["w1u8"], w1_scale_u8=v["w1sc"],
            sorted_expert_ids=v["sei"], cumsum_tensor=v["cumsum"],
            sorted_token_ids=v["sti"], inter_sorted_quant=v["isq"],
            inter_sorted_shuffled_scale=v["iss"], hidden_states=v["hidden"],
            n_tokens=token, NE=E, D_HIDDEN=model_dim, D_INTER=inter_dim, topk=topk,
            BM=BM_S1, use_nt=gemm1_use_nt(E, topk, token, BM_S1), interleave=True,
            a_dtype=d.get("adtype", "fp8"), out_dtype=stage2_adtype, act="silu", swiglu_limit=0.0,
            SBM=BM_S1, k_wave=k_wave, BN=BN, n_sorted_padded=v["n"],
            model_dim_pad=0, inter_dim_pad=0,
        )
        torch.cuda.synchronize()

    # The real cost of the reduce epilog = gemm2 (writes per-slot [token,topk,model_dim])
    # + _run_moe_reduction (reduces to [token,model_dim]), matching the real v2 pipeline
    # (_flydsl_v2_stage2_wrapper). Both steps are counted in fn so it is a fair comparison
    # against atomic's single kernel that writes [token,model_dim] directly.
    is_reduce = epilog == "reduce"
    out = torch.empty((token, model_dim), dtype=dtypes.bf16, device="cuda")
    inter = (
        torch.empty((token, topk, model_dim), dtype=dtypes.bf16, device="cuda")
        if is_reduce
        else None
    )
    gemm2_out = inter if is_reduce else out

    def fn():
        mxfp4_moe_gemm2(
            inter_sorted_quant=v["isq"],
            inter_sorted_shuffled_scale=v["iss"],
            w2_u8=v["w2u8"],
            w2_scale_u8=v["w2sc"],
            sorted_expert_ids=v["sei"],
            cumsum_tensor=v["cumsum"],
            sorted_token_ids=v["sti"],
            sorted_weights=v["swt"],
            out=gemm2_out,
            M_logical=token,
            max_sorted=v["max_sorted"],
            NE=E,
            D_HIDDEN=model_dim,
            D_INTER=inter_dim,
            topk=topk,
            BM=BM_S2,
            use_nt=use_nt,
            a_dtype=stage2_adtype,
            epilog=epilog,
            SBM=BM_S1,
            persist=persist,
            n_sorted_padded=v["n"],
            model_dim_pad=0,
            inter_dim_pad=0,
        )
        if is_reduce:
            _run_moe_reduction(inter, out, token, topk, model_dim)
        return out

    if not is_reduce:
        out.zero_()
    fn()
    torch.cuda.synchronize()
    ref = d["ref2"].float()
    got = out.float()
    if print_output:
        _print_close_stats("gemm2 v2 vs torch ref", ref, got)
        _print_tensor("torch ref gemm2 output", ref)
        _print_tensor("v2 gemm2 output", got)
    ok = torch.isclose(ref, got, atol=1.0, rtol=0.05).float().mean().item() * 100
    _, us = run_perftest(fn, num_warmup=WARMUP, num_iters=ITERS)
    return us, ok


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=("gemm1", "gemm2"), default="gemm1")
    p.add_argument("--adtype", choices=("fp8", "fp4"), default="fp8",
                   help="stage1 activation dtype: fp8 (a8w4, default) or fp4 (a4w4). "
                        "Selects the matching a-quant + w1 preshuffle + kernel a_dtype.")
    p.add_argument("--csv", default="aiter/configs/model_configs/dsv4_fp8fp4_tuned_fmoe.csv")
    p.add_argument("--model-dim", type=int, default=7168)
    p.add_argument("--inter-dim", type=int, default=512)
    p.add_argument("-E", "--experts", type=int, default=384)
    p.add_argument("-k", "--topk", type=int, default=6)
    p.add_argument("--tokens", type=int, nargs="+", default=None,
                   help="override; default = all tuned M for the shape in the CSV")
    p.add_argument("--same-tile", action="store_true",
                   help="force v2 onto the baseline's tile config (BM=tile_m, k_wave, "
                        "use_nt=b_nt==2, BN=64 if k_wave>1 else 256) instead of v2's "
                        "own select_pipe_config -- isolates kernel-vs-kernel.")
    p.add_argument("--print-output", action="store_true",
                   help="print output tensors for the selected stage")
    p.add_argument("--print-baseline-output", action="store_true",
                   help="print the baseline gemm2 output tensor")
    p.add_argument("--v2-only", action="store_true",
                   help="only run the v2 kernels (gemm1 producer + gemm2), skipping "
                        "the baseline flydsl/opus side entirely (baseline column shows nan).")
    p.add_argument("--baseline-producer", action="store_true",
                   help="use the baseline gemm1 (mixed_moe_gemm_2stage) to produce the "
                        "gemm2 intermediate. Default is the v2 gemm1 (mxfp4_moe_gemm1), "
                        "which is also compatible with FLYDSL_DEBUG_ENABLE_DEBUG_INFO=1.")
    args = p.parse_args()

    # gather tuned (token, block_m, kernelName1) for the requested shape
    rows = []
    with open(args.csv, newline="") as f:
        for r in _csv.DictReader(f):
            if (int(r["model_dim"]) == args.model_dim and int(r["inter_dim"]) == args.inter_dim
                    and int(r["expert"]) == args.experts and int(r["topk"]) == args.topk):
                rows.append(r)
    # dedup by token (keep first occurrence per tuned M)
    _seen = set()
    _uniq = []
    for r in sorted(rows, key=lambda r: int(r["token"])):
        t = int(r["token"])
        if t in _seen:
            continue
        _seen.add(t)
        _uniq.append(r)
    rows = _uniq
    if args.tokens is not None:
        want = set(args.tokens)
        rows = [r for r in rows if int(r["token"]) in want]
    if not rows:
        print("no matching CSV rows for shape")
        return

    _qtag = "a8w4" if args.adtype == "fp8" else "a4w4"
    print(f"dsv4 {_qtag} {args.stage}  md={args.model_dim} id={args.inter_dim} "
          f"E={args.experts} topk={args.topk}  (BALANCED, launch-only)")
    if args.stage == "gemm1":
        print("baseline = flydsl mixed_moe stage1 (CSV kernelName1) | "
              "v2 = FlyDSL#753 mxfp4_moe_gemm1\n")
    else:
        print("baseline = CSV kernelName2 stage2 (flydsl/opus) | "
              "v2 = FlyDSL#753 mxfp4_moe_gemm2\n")
    hdr = f"{'M':>7} {'blk':>4} | {'base us':>9} {'ok%':>5} | {'v2 us':>9} {'cos%':>5} {'v2 cfg':>18} | {'delta%':>7}"
    print(hdr)
    print("-" * len(hdr))

    results = []
    for r in rows:
        token = int(r["token"])
        bm_csv = int(r["block_m"])
        kn1 = r["kernelName1"]
        params1 = get_flydsl_kernel_params(kn1) or {}
        params1.setdefault("tile_m", bm_csv)
        params1.setdefault("tile_n", 64)
        params1.setdefault("tile_k", 256)
        sort_bm = params1["tile_m"]

        kn2 = r.get("kernelName2", "")
        # A gemm2 row can pin itself to the v2 kernel via a flydslv2_moe2_* name in
        # kernelName2. When it does (and we're benching gemm2), the v2 config is
        # read from the name and the baseline flydsl/opus side is skipped for
        # that row -- the CSV row IS the "use v2 for moe2" decision.
        v2_g2 = parse_flydsl_v2_gemm2_kernel(kn2) if args.stage == "gemm2" else None
        is_opus2 = _opus_a8w4.is_opus_a8w4_stage2_kernel(kn2)
        params2 = get_flydsl_kernel_params(kn2) or {}
        params2.setdefault("tile_m", bm_csv)
        params2.setdefault("tile_n", 256)
        params2.setdefault("tile_k", 256)
        params2.setdefault("mode", "atomic")
        if is_opus2:
            opus_values = _opus_a8w4.stage2_cfg_values(r, bm_csv)
            params2["tile_m"] = int(opus_values["stage2_block_m"])
            params2["mode"] = "opus-route" if bool(opus_values["route_out"]) else "opus-atomic"

        d = gen(token, args.model_dim, args.inter_dim, args.experts, args.topk, sort_bm,
                adtype=args.adtype)
        if args.v2_only:
            # --v2-only: skip the baseline side entirely, only run the v2 kernels.
            base_us, base_ok = float("nan"), -1
        elif v2_g2 is not None:
            # v2-pinned gemm2 row: the CSV's chosen gemm2 IS the v2 kernel, so
            # the baseline side is that same v2 kernel. Time it below (after the
            # v2 config + inputs are built) so both columns measure it and match.
            base_us, base_ok = float("nan"), -1
        else:
            try:
                if args.stage == "gemm1":
                    base_us, base_ok = time_baseline(d, token, args.topk, params1)
                else:
                    base_us, base_ok = time_baseline_gemm2(
                        d, token, args.model_dim, args.topk, params2, row=r,
                        print_output=args.print_output or args.print_baseline_output
                    )
            except Exception as e:
                base_us, base_ok = float("nan"), -1
                print(f"{token:>7} {bm_csv:>4} | baseline FAIL: {str(e)[:70]}")

        if args.same_tile:
            # match the baseline tuned tile: BM, k_wave, nt; map baseline tile_n -> v2 BN.
            if args.stage == "gemm1":
                BM_S1 = params1["tile_m"]
                BM_v2 = BM_S1
                epilog = "atomic"
                persist = False
                KW_v2 = params1.get("k_wave", 1)
                use_nt = params1.get("b_nt", 2) == 2
                tn = params1.get("tile_n", 256)
            else:
                BM_S1 = sort_bm
                BM_v2 = min(params2["tile_m"], 64)
                epilog = params2.get("mode", "atomic")
                persist = bool(params2.get("persist", False))
                KW_v2 = params1.get("k_wave", 1)
                use_nt = params2.get("b_nt", 0) == 2
                tn = params1.get("tile_n", 256)
            # v2 BN in {64,256}: tile_n<=64 -> BN64, tile_n>=128 -> BN256 (v2 has no BN128,
            # so tile_n=128 is only approximated). BN64 structurally needs k_wave>=2.
            if tn <= 64:
                BN_v2 = 64
            elif tn <= 128:
                BN_v2 = 128
            else:
                BN_v2 = 256
            if BN_v2 == 64 and KW_v2 < 2:
                KW_v2 = 2
        else:
            # v2 dispatcher's own config for this M
            BM_v2, epilog, BM_S1, persist, BN_v2, KW_v2 = select_pipe_config(
                args.model_dim, args.inter_dim, args.experts, args.topk, token
            )
            if args.stage == "gemm2":
                # Honor tuned gemm2 overrides (_GEMM2_TUNED_TABLE via select_gemm2_config);
                # falls back to select_pipe_config/gemm2_use_nt defaults when no tuned entry.
                BM_v2, epilog, persist, use_nt = select_gemm2_config(
                    args.model_dim, args.inter_dim, args.experts, args.topk, token
                )
                if v2_g2 is not None:
                    # Align the sort padding unit to the CSV kernelName1 tile
                    # (shared SBM for the gemm1 producer and gemm2).
                    BM_S1 = sort_bm
                if BM_S1 % BM_v2 != 0:
                    # gemm2 consumes an SBM-strided sorted stream and requires SBM
                    # to be a multiple of its BM. Tiny-M gemm1 may choose BM16, so
                    # promote the standalone gemm2 bench input stream to BM32.
                    BM_S1 = BM_v2
            else:
                use_nt = gemm1_use_nt(args.experts, args.topk, token, BM_S1)

        if v2_g2 is not None:
            # Override the v2 gemm2 config with the CSV-pinned kernel name.
            # BN_v2/KW_v2 stay from select_pipe_config -- they only shape the
            # gemm1 producer that fills the intermediate, not the timed gemm2.
            BM_v2 = v2_g2["tile_m"]
            epilog = v2_g2["epilog"]
            persist = v2_g2["persist"]
            use_nt = v2_g2["use_nt"]
            BM_S1 = v2_g2["sort_block_m"] or BM_S1
            if v2_g2 is not None:
                # Align the sort unit to the CSV kernelName1 tile (shared SBM
                # for the gemm1 producer and gemm2).
                BM_S1 = sort_bm
            if BM_S1 % BM_v2 != 0:
                BM_S1 = BM_v2
        try:
            v = build_v2_inputs(d, token, args.model_dim, args.inter_dim,
                                args.experts, args.topk, BM_S1)
            if args.stage == "gemm1":
                if args.print_output:
                    print_gemm1_v2_layout_compare(
                        d, v, token, args.model_dim, args.inter_dim,
                        args.experts, args.topk, BM_S1, use_nt, BN_v2, KW_v2,
                        params1,
                    )
                v2_us, v2_nz = time_v2(
                    d, v, token, args.model_dim, args.inter_dim,
                    args.experts, args.topk, BM_S1, use_nt, BN_v2, KW_v2
                )
            else:
                v2_us, v2_nz = time_v2_gemm2(
                    d, v, token, args.model_dim, args.inter_dim,
                    args.experts, args.topk, BM_S1, BM_v2, use_nt, epilog, persist,
                    BN_v2, KW_v2, base_gemm1_params=params1,
                    print_output=args.print_output,
                    use_baseline_producer=args.baseline_producer,
                )
                if v2_g2 is not None and not args.v2_only:
                    # The CSV's chosen gemm2 is this same v2 kernel, so the
                    # baseline column re-times it independently -- both sides
                    # measure the identical kernel and should match.
                    base_us, base_ok = time_v2_gemm2(
                        d, v, token, args.model_dim, args.inter_dim,
                        args.experts, args.topk, BM_S1, BM_v2, use_nt, epilog,
                        persist, BN_v2, KW_v2, base_gemm1_params=params1,
                        use_baseline_producer=args.baseline_producer,
                    )
        except Exception as e:
            v2_us, v2_nz = float("nan"), -1
            print(f"{token:>7} {bm_csv:>4} | v2 FAIL: {str(e)[:80]}")

        delta = (v2_us - base_us) / base_us * 100 if base_us == base_us and v2_us == v2_us else float("nan")
        if args.stage == "gemm1":
            cfg = f"bm{BM_S1}bn{BN_v2}kw{KW_v2}{'nt' if use_nt else ''}"
        else:
            cfg = f"bm{BM_v2}{epilog[0]}sbm{BM_S1}{'p' if persist else ''}{'nt' if use_nt else ''}"
        print(f"{token:>7} {bm_csv:>4} | {base_us:9.3f} {base_ok:5.1f} | "
              f"{v2_us:9.3f} {v2_nz:5.1f} {cfg:>18} | {delta:+7.1f}")
        results.append((token, base_us, v2_us, delta))

    print(f"\nsummary {args.stage} (v2 delta% vs baseline; negative = v2 faster):")
    for token, b, vv, dl in results:
        print(f"  M={token:>7}: base {b:8.3f}us  v2 {vv:8.3f}us  {dl:+6.1f}%")


if __name__ == "__main__":
    main()
