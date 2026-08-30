#!/usr/bin/env python3
"""Grip robustness under scripted disturbance: does jerk break the grip, and
does the monolithic policy hold better than the decomposed one?

Survival is reported CONDITIONAL on lift certification, because that is the
population that actually reaches TRANSPORT in the deployed pipeline. Arms are
compared pairwise on per-seed rates -- all arms share the 12 eval seeds and
spawn sequences, so a pooled two-proportion z understates the effect (that
error cost Fix A a "ns" verdict earlier in this campaign).
"""
import csv, glob, os
from collections import defaultdict
import numpy as np
from scipy import stats

ROOT = "Results/grip_jerk"
ARMS = [("mono", "monolithic v7 FP32"), ("dfp32", "decomposed FP32"),
        ("int8", "decomposed INT8")]
PROFILES = ["hold", "carry", "jerk"]


def load(profile, arm):
    rows = []
    for f in sorted(glob.glob(os.path.join(ROOT, profile, f"{arm}_e*.csv"))):
        with open(f) as fh:
            rows += list(csv.DictReader(fh))
    return rows


def fnum(r, k):
    v = r.get(k, "")
    return float(v) if v not in ("", None) else float("nan")


def main():
    data = {(p, a): load(p, a) for p in PROFILES for a, _ in ARMS}
    for k, v in data.items():
        if not v:
            print(f"MISSING {k}")

    print("=" * 78)
    print("GRIP ROBUSTNESS UNDER SCRIPTED DISTURBANCE (transport policy removed)")
    print("=" * 78)
    print("\nsurvival is conditional on lift certification -- the population that")
    print("reaches TRANSPORT on the real pipeline.\n")
    hdr = f"{'profile':7} {'arm':22} {'n':>5} {'grasp%':>7} {'liftcert%':>10} {'survive%':>9} {'lost_at med':>12}"
    print(hdr); print("-" * len(hdr))
    surv = {}
    for p in PROFILES:
        for a, label in ARMS:
            R = data[(p, a)]
            if not R:
                continue
            n = len(R)
            g = sum(int(r["grasp_ok"]) for r in R)
            lc = [r for r in R if int(r["lift_ok"])]
            sv = sum(int(r["survived"]) for r in lc)
            la = [int(r["lost_at"]) for r in lc
                  if not int(r["survived"]) and int(r["lost_at"]) >= 0]
            surv[(p, a)] = (sv, len(lc))
            print(f"{p:7} {label:22} {n:5d} {g/n*100:6.2f}% {len(lc)/n*100:9.2f}% "
                  f"{sv/max(1,len(lc))*100:8.2f}% "
                  f"{np.median(la) if la else float('nan'):12.1f}")
        print()

    # ---- is jerk the mechanism, or is the grip just weak? -----------------
    print("=" * 78)
    print("IS JERK THE MECHANISM?  (hold is the control: near-zero translation)")
    print("=" * 78)
    for a, label in ARMS:
        line = f"  {label:22}"
        for p in PROFILES:
            s, t = surv.get((p, a), (0, 0))
            line += f"  {p}={s/max(1,t)*100:6.2f}%"
        h = surv.get(("hold", a), (0, 1)); j = surv.get(("jerk", a), (0, 1))
        line += f"   jerk-hold = {j[0]/max(1,j[1])*100 - h[0]/max(1,h[1])*100:+6.2f} pts"
        print(line)

    # ---- paired per-seed comparisons --------------------------------------
    print("\n" + "=" * 78)
    print("PAIRED PER-SEED COMPARISON (12 eval seeds, shared spawns)")
    print("=" * 78)

    def per_seed(p, a):
        d = defaultdict(lambda: [0, 0])
        for r in data[(p, a)]:
            if int(r["lift_ok"]):
                d[r["seed"]][1] += 1
                d[r["seed"]][0] += int(r["survived"])
        return {k: v[0] / v[1] * 100 for k, v in d.items() if v[1]}

    for p in PROFILES:
        print(f"\n  profile = {p}")
        base = per_seed(p, "int8")
        for a, label in ARMS:
            if a == "int8":
                continue
            cur = per_seed(p, a)
            ks = sorted(set(base) & set(cur))
            if len(ks) < 3:
                continue
            x = np.array([cur[k] for k in ks]); y = np.array([base[k] for k in ks])
            t, pv = stats.ttest_rel(x, y)
            up = int((x > y).sum()); dn = int((x < y).sum())
            print(f"    {label:22} vs decomposed INT8: {x.mean():6.2f}% vs "
                  f"{y.mean():6.2f}%  ({x.mean()-y.mean():+6.2f})  "
                  f"t={t:5.2f} p={pv:.4f}  seeds up/down {up}/{dn}")

    # ---- what predicts a drop? -------------------------------------------
    print("\n" + "=" * 78)
    print("WHAT PREDICTS LOSING THE GRIP UNDER JERK")
    print("=" * 78)
    for a, label in ARMS:
        R = [r for r in data[("jerk", a)] if int(r["lift_ok"])]
        if not R:
            continue
        print(f"\n  {label}  (n={len(R)} lift-certified)")
        print("    grip angle from a flat face:")
        for lo, hi in [(0, 15), (15, 30), (30, 46)]:
            b = [r for r in R if lo <= fnum(r, "diag_deg") < hi]
            if b:
                s = sum(int(r["survived"]) for r in b)
                print(f"      {lo:2d}-{hi:2d} deg  n={len(b):4d}  survive {s/len(b)*100:6.2f}%")
        print("    joint margin at handoff:")
        for lo, hi, nm in [(-9, .02, "pinned  <0.02"), (.02, .10, "tight .02-.10"),
                           (.10, 9, "clear   >0.10")]:
            b = [r for r in R if lo <= fnum(r, "margin_hand") < hi]
            if b:
                s = sum(int(r["survived"]) for r in b)
                print(f"      {nm:14} n={len(b):4d}  survive {s/len(b)*100:6.2f}%")
        pin = [r for r in R if fnum(r, "margin_hand") < 0.02]
        print(f"    -> pinned at handoff: {len(pin)/len(R)*100:.2f}% of certified lifts")


if __name__ == "__main__":
    main()
