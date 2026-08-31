#!/usr/bin/env python3
"""Does Net2WiderNet capacity (128/64) beat the same policy left at 64/32?

Both arms continue from the SAME qat8_sN parent for the SAME 6000 episodes with
--actor-wclip 8 --actor-fakequant. Only the actor width differs (14,528 vs
5,216 weights), so the pair isolates capacity from the extra training that
widening would otherwise smuggle in.

Pairing is on (training seed, eval seed) -- the arms share both levels, so a
pooled two-proportion z would understate the effect, the error that gave Fix A
a "ns" verdict earlier in this campaign.
"""
import csv, glob, re
import numpy as np
from scipy import stats

SEEDS = [0, 1, 2, 3, 4, 5]


def cell(arm, sd, prec):
    out = {}
    for f in glob.glob(f"Results/widen/eval/{arm}_s{sd}_{prec}_e*.csv"):
        rows = list(csv.DictReader(open(f)))
        if len(rows) == 100:
            out[int(re.search(r"_e(\d+)\.csv$", f).group(1))] = \
                sum(int(r["success"]) for r in rows)
    return out


def main():
    W = {(s, p): cell("wide128", s, p) for s in SEEDS for p in ("fp32", "int8")}
    C = {(s, p): cell("cont64", s, p) for s in SEEDS for p in ("fp32", "int8")}
    inc = [f"{a}_s{s}_{p}" for a, D in (("wide128", W), ("cont64", C))
           for (s, p), v in D.items() if len(v) != 12]
    print("=" * 72)
    print("NET2WIDER 128/64 vs SAME POLICY AT 64/32 (matched parent + training)")
    print("=" * 72)
    if inc:
        print("\nINCOMPLETE CELLS:", ", ".join(inc))

    print(f"\n{'tseed':6} {'wide INT8':>10} {'cont INT8':>10} {'d':>7}   "
          f"{'wide FP32':>10} {'cont FP32':>10} {'d':>7}")
    print("-" * 64)
    A = []
    for s in SEEDS:
        wi, ci = W[(s, "int8")], C[(s, "int8")]
        wf, cf = W[(s, "fp32")], C[(s, "fp32")]
        ki, kf = sorted(set(wi) & set(ci)), sorted(set(wf) & set(cf))
        if not ki:
            continue
        a, b = np.mean([wi[k] for k in ki]), np.mean([ci[k] for k in ki])
        c, d = np.mean([wf[k] for k in kf]), np.mean([cf[k] for k in kf])
        A.append((a, b, c, d))
        print(f"s{s:<5} {a:9.2f}% {b:9.2f}% {a-b:+7.2f}   "
              f"{c:9.2f}% {d:9.2f}% {c-d:+7.2f}")
    A = np.array(A)
    print("-" * 64)
    print(f"{'mean':6} {A[:,0].mean():9.2f}% {A[:,1].mean():9.2f}% "
          f"{A[:,0].mean()-A[:,1].mean():+7.2f}   "
          f"{A[:,2].mean():9.2f}% {A[:,3].mean():9.2f}% "
          f"{A[:,2].mean()-A[:,3].mean():+7.2f}")

    print("\n" + "=" * 72)
    print("PAIRED on (training seed, eval seed)")
    print("=" * 72)
    for p in ("int8", "fp32"):
        x, y = [], []
        for s in SEEDS:
            for k in sorted(set(W[(s, p)]) & set(C[(s, p)])):
                x.append(W[(s, p)][k]); y.append(C[(s, p)][k])
        x, y = np.array(x, float), np.array(y, float)
        t, pv = stats.ttest_rel(x, y)
        wl = stats.wilcoxon(x, y, zero_method="zsplit")
        d = x - y
        per = [np.mean([W[(s,p)][k]-C[(s,p)][k]
                        for k in sorted(set(W[(s,p)]) & set(C[(s,p)]))]) for s in SEEDS]
        print(f"  {p.upper()}: wide {x.mean():6.2f}% vs cont {y.mean():6.2f}%  "
              f"({d.mean():+5.2f})  n={len(d)}")
        print(f"       t={t:5.2f} p={pv:.4f}  wilcoxon p={wl.pvalue:.4f}  "
              f"up/down/tie {(d>0).sum()}/{(d<0).sum()}/{(d==0).sum()}")
        print(f"       per-seed: {[round(v,2) for v in per]}  "
              f"({sum(v>0 for v in per)}/6 positive)")

    print("\n" + "=" * 72)
    print("AGAINST THE 90% TARGET")
    print("=" * 72)
    gap_w = A[:,2].mean() - A[:,0].mean()
    gap_c = A[:,3].mean() - A[:,1].mean()
    print(f"  quantisation gap   wide {gap_w:5.2f}   cont {gap_c:5.2f}")
    print(f"  best FP32 cell     wide {A[:,2].max():5.2f}%  cont {A[:,3].max():5.2f}%")
    print(f"  best INT8 cell     wide {A[:,0].max():5.2f}%  cont {A[:,1].max():5.2f}%")
    print(f"  FP32 needed for INT8>=90 at wide's gap: {90+gap_w:.2f}%")


if __name__ == "__main__":
    main()
