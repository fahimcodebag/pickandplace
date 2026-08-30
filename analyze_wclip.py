#!/usr/bin/env python3
"""Does weight-range clipping shrink the per-tensor INT8 quantisation gap?

Per-tensor INT8 sets ONE scale per tensor from the largest weight, so a
tensor's max/std IS its quantisation cost (Results/int8_deployment.txt Fnd 4):
transport quantises free at 5.7, gripfix_s2 costs -7.2 at 9.3, bi_s0 -14.0 at
10.0. --actor-wclip projects every actor Linear weight onto |w| <= k*std(w)
after each update, capping that ratio at k exactly.

The question is not "is INT8 better" alone but WHERE THE TRADE TURNS OVER: a
lower k should shrink the gap while possibly costing FP32 quality. An arm that
gains 8 INT8 points and loses 10 in FP32 is a loss, so both cells are reported.

Pairing: arm k and the control share training seeds (same np/torch seed, same
env seed0) and the 12 eval seeds with identical spawn sequences, so
(training_seed, eval_seed) is a legitimate pairing unit -- 36 pairs per
comparison. Pooled two-proportion z understates these effects badly; that error
cost Fix A a "ns" verdict earlier in this campaign.
"""
import csv, glob, os, re
from collections import defaultdict
import numpy as np
from scipy import stats

ROOT = "Results/wclip/eval"
KS = [0, 4, 6, 8]
TSEEDS = [0, 1, 2]
# achieved max/std in the saved best checkpoints, measured not assumed
RATIO = {0: "8.3-10.3", 4: "4.00", 6: "6.00", 8: "~8.00"}


def cell(k, ts, prec):
    """-> {eval_seed: success_rate_pct}, over completed shards only."""
    out = {}
    for f in glob.glob(os.path.join(ROOT, f"w{k}_s{ts}_{prec}_e*.csv")):
        m = re.search(r"_e(\d+)\.csv$", f)
        rows = list(csv.DictReader(open(f)))
        if len(rows) != 100:
            continue
        out[int(m.group(1))] = sum(int(r["success"]) for r in rows)
    return out


def main():
    data = {(k, ts, p): cell(k, ts, p)
            for k in KS for ts in TSEEDS for p in ("fp32", "int8")}
    missing = [f"w{k}_s{ts}_{p}" for (k, ts, p), v in data.items() if len(v) != 12]
    print("=" * 78)
    print("WEIGHT-RANGE CLIPPING vs THE PER-TENSOR INT8 GAP")
    print("=" * 78)
    if missing:
        print(f"\nINCOMPLETE CELLS ({len(missing)}), reported on what exists:")
        for m in missing:
            k, ts, p = re.match(r"w(\d+)_s(\d+)_(\w+)", m).groups()
            print(f"   {m}: {len(data[(int(k), int(ts), p)])}/12 eval seeds")

    print(f"\n{'arm':6} {'max/std':9} {'tseed':6} {'FP32%':>7} {'INT8%':>7} {'gap':>7}")
    print("-" * 50)
    agg = defaultdict(list)
    for k in KS:
        for ts in TSEEDS:
            f, i = data[(k, ts, "fp32")], data[(k, ts, "int8")]
            ks_ = sorted(set(f) & set(i))
            if not ks_:
                continue
            fv = np.mean([f[s] for s in ks_]); iv = np.mean([i[s] for s in ks_])
            agg[k].append((fv, iv, fv - iv))
            print(f"wclip{k:<1} {RATIO[k]:9} s{ts:<5} {fv:6.2f}% {iv:6.2f}% {fv-iv:+6.2f}")
        print()

    print("=" * 78)
    print("AGGREGATE OVER 3 TRAINING SEEDS (mean +- sd)")
    print("=" * 78)
    print(f"{'arm':7} {'max/std':9} {'FP32%':>16} {'INT8%':>16} {'gap':>16}")
    for k in KS:
        a = np.array(agg[k])
        if not len(a):
            continue
        print(f"wclip{k:<2} {RATIO[k]:9} "
              f"{a[:,0].mean():8.2f} +-{a[:,0].std(ddof=1):4.2f} "
              f"{a[:,1].mean():8.2f} +-{a[:,1].std(ddof=1):4.2f} "
              f"{a[:,2].mean():8.2f} +-{a[:,2].std(ddof=1):4.2f}")

    print("\n" + "=" * 78)
    print("PAIRED vs CONTROL on (training seed, eval seed) -- 36 pairs")
    print("=" * 78)
    for prec in ("int8", "fp32"):
        print(f"\n  {prec.upper()}")
        for k in KS[1:]:
            x, y = [], []
            for ts in TSEEDS:
                c, t = data[(0, ts, prec)], data[(k, ts, prec)]
                for s in sorted(set(c) & set(t)):
                    y.append(c[s]); x.append(t[s])
            if len(x) < 6:
                print(f"    wclip{k}: only {len(x)} pairs, skipped"); continue
            x, y = np.array(x, float), np.array(y, float)
            d = x - y
            t_, p_ = stats.ttest_rel(x, y)
            print(f"    wclip{k} vs control: {x.mean():6.2f}% vs {y.mean():6.2f}%  "
                  f"({d.mean():+6.2f})  t={t_:6.2f} p={p_:.4f}  "
                  f"up/down {(d>0).sum()}/{(d<0).sum()} of {len(d)}")

    print("\n" + "=" * 78)
    print("THE TRADE: does a smaller gap survive the FP32 cost?")
    print("=" * 78)
    a0 = np.array(agg[0])
    for k in KS:
        a = np.array(agg[k])
        if not len(a):
            continue
        print(f"  wclip{k:<2} gap {a[:,2].mean():5.2f} "
              f"({a[:,2].mean()-a0[:,2].mean():+5.2f} vs control)   "
              f"FP32 {a[:,0].mean()-a0[:,0].mean():+5.2f}   "
              f"NET INT8 {a[:,1].mean()-a0[:,1].mean():+5.2f}")
    print("\n  NET INT8 is the decision column: it already contains both the")
    print("  quantisation gain and whatever FP32 quality the clip cost.")


if __name__ == "__main__":
    main()
