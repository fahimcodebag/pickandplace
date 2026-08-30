#!/usr/bin/env python3
"""Does QAT inside the RL loop beat weight-clipping alone?

Both arms carry --actor-wclip 8 and share the warm start, config and training
seeds, so QAT is the single variable. Pairing is on (training seed, eval seed)
-- 6 training seeds x 12 eval seeds = 72 pairs -- because the arms share both
levels. A pooled two-proportion z understates these effects badly; that error
gave Fix A a "ns" verdict earlier in this campaign.

Reports INT8 first: that is the deployed artifact and the whole point. FP32 is
reported to show what QAT costs (or does not cost) in full precision.
"""
import csv, glob, os, re
from collections import defaultdict
import numpy as np
from scipy import stats

SEEDS = [0, 1, 2, 3, 4, 5]


def cell(path_glob):
    out = {}
    for f in glob.glob(path_glob):
        rows = list(csv.DictReader(open(f)))
        if len(rows) == 100:
            out[int(re.search(r"_e(\d+)\.csv$", f).group(1))] = \
                sum(int(r["success"]) for r in rows)
    return out


def main():
    qat = {(s, p): cell(f"Results/qat8/eval/q{s}_{p}_e*.csv")
           for s in SEEDS for p in ("fp32", "int8")}
    ctl = {(s, p): cell(f"Results/wclip/eval/w8_s{s}_{p}_e*.csv")
           for s in SEEDS for p in ("fp32", "int8")}

    print("=" * 74)
    print("QAT-IN-TRAINING vs WEIGHT-CLIPPING ALONE (both --actor-wclip 8)")
    print("=" * 74)
    inc = [(k, len(v)) for k, v in {**{("qat",) + k: v for k, v in qat.items()},
                                    **{("ctl",) + k: v for k, v in ctl.items()}}.items()
           if len(v) != 12]
    if inc:
        print("\nINCOMPLETE CELLS (reported on what exists):")
        for k, n in inc:
            print(f"   {k}: {n}/12 eval seeds")

    print(f"\n{'tseed':6} {'QAT INT8':>10} {'ctl INT8':>10} {'d':>7}   "
          f"{'QAT FP32':>10} {'ctl FP32':>10} {'d':>7}")
    print("-" * 66)
    agg = []
    for s in SEEDS:
        qi, ci = qat[(s, "int8")], ctl[(s, "int8")]
        qf, cf = qat[(s, "fp32")], ctl[(s, "fp32")]
        ki, kf = sorted(set(qi) & set(ci)), sorted(set(qf) & set(cf))
        if not ki:
            continue
        a, b = np.mean([qi[k] for k in ki]), np.mean([ci[k] for k in ki])
        c, d = np.mean([qf[k] for k in kf]), np.mean([cf[k] for k in kf])
        agg.append((a, b, c, d))
        print(f"s{s:<5} {a:9.2f}% {b:9.2f}% {a-b:+7.2f}   "
              f"{c:9.2f}% {d:9.2f}% {c-d:+7.2f}")
    A = np.array(agg)
    print("-" * 66)
    print(f"{'mean':6} {A[:,0].mean():9.2f}% {A[:,1].mean():9.2f}% "
          f"{A[:,0].mean()-A[:,1].mean():+7.2f}   "
          f"{A[:,2].mean():9.2f}% {A[:,3].mean():9.2f}% "
          f"{A[:,2].mean()-A[:,3].mean():+7.2f}")

    print("\n" + "=" * 74)
    print("PAIRED on (training seed, eval seed)")
    print("=" * 74)
    for p in ("int8", "fp32"):
        x, y = [], []
        for s in SEEDS:
            q, c = qat[(s, p)], ctl[(s, p)]
            for k in sorted(set(q) & set(c)):
                x.append(q[k]); y.append(c[k])
        x, y = np.array(x, float), np.array(y, float)
        t, pv = stats.ttest_rel(x, y)
        w = stats.wilcoxon(x, y, zero_method="zsplit")
        d = x - y
        print(f"  {p.upper()}: QAT {x.mean():6.2f}% vs control {y.mean():6.2f}%  "
              f"({d.mean():+5.2f})  n={len(d)}")
        print(f"       t={t:5.2f} p={pv:.4f}   wilcoxon p={w.pvalue:.4f}   "
              f"up/down/tie {(d>0).sum()}/{(d<0).sum()}/{(d==0).sum()}")
        # per-training-seed sign test: does it hold across seeds, not just pairs?
        per = [np.mean([qat[(s,p)][k]-ctl[(s,p)][k]
                        for k in sorted(set(qat[(s,p)]) & set(ctl[(s,p)]))])
               for s in SEEDS if set(qat[(s,p)]) & set(ctl[(s,p)])]
        print(f"       per-seed deltas: {[round(v,2) for v in per]}  "
              f"({sum(v>0 for v in per)}/{len(per)} positive)")

    print("\n" + "=" * 74)
    print("REFERENCE POINTS (same 12-seed protocol)")
    print("=" * 74)
    dep = cell("Results/fixAC/int8_e*.csv")
    if dep:
        print(f"  deployed bi_s0 INT8          {np.mean(list(dep.values())):6.2f}%")
    print(f"  control (wclip8, 6 seeds)    {A[:,1].mean():6.2f}%")
    print(f"  QAT     (qat8,   6 seeds)    {A[:,0].mean():6.2f}%")
    print(f"  best single QAT cell         {A[:,0].max():6.2f}%")


if __name__ == "__main__":
    main()
