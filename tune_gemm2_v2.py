"""Tune the v2 (FlyDSL#753) gemm2 kernel config per (shape, token).

For a given MoE shape (model_dim/inter_dim/E/topk) and the tuned M list from a CSV,
sweep the v2 gemm2 structural knobs and pick the fastest config that stays correct:

    BM_S2 (gemm2 compute tile)  x  use_nt  x  epilog  x  persist

SBM (the sorted-stream padding unit) is NOT tuned: it is fixed to the baseline gemm1
tile parsed from the CSV kernelName1 (tile_m). This mirrors real testing with
AITER_FMOE_V2=1, where the baseline gemm1 output feeds the v2 gemm2 -- so the sort
unit is dictated by baseline gemm1, and gemm2's own BM only needs to divide it.

Timing/correctness reuse the bench harness verbatim (gen / build_v2_inputs /
time_v2_gemm2), so the methodology matches bench_gemm12_v2_vs_baseline.py exactly.

Usage:
  /opt/venv/bin/python tune_gemm2_v2.py \
      --csv aiter/configs/model_configs/glm5_fp4_tuned_fmoe.csv \
      --model-dim 6144 --inter-dim 512 -E 257 -k 9 --adtype fp4
"""

import argparse
import csv as _csv
import itertools
import os

# The tuner feeds v2 gemm2 from the baseline gemm1 producer (matches real testing);
# time_v2_gemm2 reads this at call time to choose populate_baseline_v2_intermediate.
os.environ["AITER_FMOE_V2"] = "1"

import bench_gemm12_v2_vs_baseline as B  # noqa: E402
from aiter.ops.flydsl.moe_kernels import get_flydsl_kernel_params  # noqa: E402


def gemm2_candidates(sbm, adtype, include_bm16):
    """Cartesian v2 gemm2 knobs for a fixed SBM: BM_S2 (divides SBM) x epilog x persist x use_nt."""
    bms = [b for b in (16, 32, 64) if b <= sbm and sbm % b == 0]
    if not include_bm16:
        bms = [b for b in bms if b != 16]
    epilogs = ["atomic", "reduce"]
    persists = [False, True] if adtype == "fp4" else [False]  # fp8-A gemm2 persist is fail-fast
    use_nts = [True, False]
    for bm, ep, pf, nt in itertools.product(bms, epilogs, persists, use_nts):
        yield {"bm_s2": bm, "epilog": ep, "persist": pf, "use_nt": nt}


def read_shape_rows(csv_path, model_dim, inter_dim, experts, topk, tokens_filter):
    """Tuned rows for the shape, deduped by token (first per M), optionally filtered."""
    rows = []
    with open(csv_path, newline="") as f:
        for r in _csv.DictReader(f):
            if (
                int(r["model_dim"]) == model_dim
                and int(r["inter_dim"]) == inter_dim
                and int(r["expert"]) == experts
                and int(r["topk"]) == topk
            ):
                rows.append(r)
    seen, uniq = set(), []
    for r in sorted(rows, key=lambda r: int(r["token"])):
        t = int(r["token"])
        if t in seen:
            continue
        seen.add(t)
        uniq.append(r)
    if tokens_filter is not None:
        want = set(tokens_filter)
        uniq = [r for r in uniq if int(r["token"]) in want]
    return uniq


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--model-dim", type=int, required=True)
    p.add_argument("--inter-dim", type=int, required=True)
    p.add_argument("-E", "--experts", type=int, required=True)
    p.add_argument("-k", "--topk", type=int, required=True)
    p.add_argument("--adtype", choices=("fp8", "fp4"), default="fp4")
    p.add_argument("--tokens", type=int, nargs="+", default=None,
                   help="override; default = all tuned M for the shape in the CSV")
    p.add_argument("--min-ok", type=float, default=99.0,
                   help="min gemm2-vs-ref2 close%% for a config to be eligible (default 99)")
    p.add_argument("--include-bm16", action="store_true",
                   help="also sweep BM_S2=16 (stage2 normally uses 32/64)")
    p.add_argument("--warmup", type=int, default=B.WARMUP)
    p.add_argument("--iters", type=int, default=B.ITERS)
    args = p.parse_args()

    B.WARMUP, B.ITERS = args.warmup, args.iters

    rows = read_shape_rows(
        args.csv, args.model_dim, args.inter_dim, args.experts, args.topk, args.tokens
    )
    if not rows:
        print("no matching CSV rows for shape")
        return

    qtag = "a8w4" if args.adtype == "fp8" else "a4w4"
    print(
        f"tune v2 gemm2  {qtag}  md={args.model_dim} id={args.inter_dim} "
        f"E={args.experts} topk={args.topk}  "
        f"(AITER_FMOE_V2=1 baseline-gemm1 producer, BALANCED, launch-only)"
    )
    hdr = (
        f"{'M':>7} {'sbm':>4} | {'bm':>3} {'epilog':>7} {'persist':>7} {'nt':>3} "
        f"| {'us':>9} {'ok%':>6}"
    )

    best_by_token = {}
    for r in rows:
        token = int(r["token"])
        bm_csv = int(r["block_m"])
        # SBM fixed = baseline gemm1 tile (kernelName1's tile_m; CK/unknown -> block_m fallback),
        # exactly as bench derives sort_bm. params1 also drives the baseline-gemm1 producer.
        params1 = get_flydsl_kernel_params(r["kernelName1"]) or {}
        params1.setdefault("tile_m", bm_csv)
        params1.setdefault("tile_n", 64)
        params1.setdefault("tile_k", 256)
        sbm = params1["tile_m"]

        d = B.gen(
            token, args.model_dim, args.inter_dim, args.experts, args.topk, sbm,
            adtype=args.adtype,
        )
        v = B.build_v2_inputs(
            d, token, args.model_dim, args.inter_dim, args.experts, args.topk, sbm
        )

        print(f"\n{hdr}")
        print("-" * len(hdr))
        results = []
        for cfg in gemm2_candidates(sbm, args.adtype, args.include_bm16):
            nt_tag = "nt" if cfg["use_nt"] else "-"
            row_pfx = (
                f"{token:>7} {sbm:>4} | {cfg['bm_s2']:>3} {cfg['epilog']:>7} "
                f"{str(cfg['persist']):>7} {nt_tag:>3} |"
            )
            try:
                # BN/k_wave are producer (gemm1) knobs; unused under AITER_FMOE_V2=1
                # (baseline gemm1 producer uses params1), pass valid defaults.
                us, ok = B.time_v2_gemm2(
                    d, v, token, args.model_dim, args.inter_dim, args.experts,
                    args.topk, sbm, cfg["bm_s2"], cfg["use_nt"], cfg["epilog"],
                    cfg["persist"], 256, 1, base_gemm1_params=params1,
                )
            except Exception as e:  # noqa: BLE001 - invalid combos are expected, keep sweeping
                print(f"{row_pfx} FAIL: {str(e)[:50]}")
                continue
            results.append((cfg, us, ok))
            print(f"{row_pfx} {us:9.3f} {ok:6.1f}")

        valid = [(cfg, us, ok) for (cfg, us, ok) in results if us == us and ok >= args.min_ok]
        if not valid:
            print(f"  >> M={token}: no valid config >= {args.min_ok}% ok")
            continue
        best_cfg, best_us, best_ok = min(valid, key=lambda x: x[1])
        best_by_token[token] = (best_cfg, best_us, best_ok, sbm)
        print(
            f"  >> M={token} best: bm{best_cfg['bm_s2']} {best_cfg['epilog']} "
            f"persist={best_cfg['persist']} {'nt' if best_cfg['use_nt'] else 'cached'} "
            f"-> {best_us:.3f}us (ok {best_ok:.1f}%, sbm={sbm})"
        )

    if not best_by_token:
        print("\nno valid best config found for any M")
        return

    print("\n" + "=" * 72)
    print("paste into aiter/ops/flydsl/kernels/mxmoe_dispatcher.py _GEMM2_TUNED_TABLE:\n")
    sig = (args.model_dim, args.inter_dim, args.experts)
    print(f"    {sig}: {{  # (bm_s2, epilog, persist, use_nt)")
    for token in sorted(best_by_token):
        best_cfg, best_us, best_ok, sbm = best_by_token[token]
        print(
            f"        {token}: ({best_cfg['bm_s2']}, {best_cfg['epilog']!r}, "
            f"{best_cfg['persist']}, {best_cfg['use_nt']}),"
            f"  # {best_us:.3f}us ok{best_ok:.0f}% sbm{sbm}"
        )
    print("    },")


if __name__ == "__main__":
    main()
