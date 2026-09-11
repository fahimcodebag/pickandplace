#!/usr/bin/env python3
"""ROT_ANCHOR_EPS on the deployed artifact (c2m512_s1): reproduction, class size, effect.

Reads Results/anchor_c2m512/ (run_anchor_c2m512.sh) and the recorded baselines
Results/holdout4/c2m512s1_{fp32,int8}_e<seed>.csv (random spawn, held-out seeds)
and Results/fixed_c2m512/{fp32,int8}_e<seed>.csv (fixed spawn, protocol seeds).

1. REPRODUCTION  eps-0 CSVs vs recorded, per episode on (success, phase, steps).
2. PASSIVITY     anchor_probe.py's own CSV vs plain fsm_sim.py, per episode.
3. CLASS SIZE    blocked / pinned episodes from the probe; bi_s0 positive control.
4. EFFECT        eps 1e-6 vs eps 0, paired on seed: delta, t(11), up/down/tie, and
                 which episodes flipped -- were they the probe's blocked ones?
"""
import csv, glob, math, os, sys
from collections import Counter

O = "Results/anchor_c2m512"
HELD = [13, 59, 71, 137, 199, 257, 331, 419, 547, 601, 733, 911]
PROT = [7, 31, 47, 89, 101, 123, 211, 307, 401, 503, 555, 2024]
CELLS = {  # cell: (seeds, recorded path template)
    "rand_fp32":  (HELD, "Results/holdout4/c2m512s1_fp32_e{}.csv"),
    "rand_int8":  (HELD, "Results/holdout4/c2m512s1_int8_e{}.csv"),
    "fixed_fp32": (PROT, "Results/fixed_c2m512/fp32_e{}.csv"),
    "fixed_int8": (PROT, "Results/fixed_c2m512/int8_e{}.csv"),
}


def rows(path):
    return list(csv.DictReader(open(path))) if os.path.exists(path) else None


def key(r):
    return (r["success"], r["phase"], r["steps"])


def tstat(d):
    n = len(d); m = sum(d) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in d) / (n - 1)) if n > 1 else 0.0
    return m, (m / (sd / math.sqrt(n)) if sd > 0 else float("inf") if m else 0.0)


print("1. REPRODUCTION of the recorded baselines at --rot-anchor-eps 0 (per episode)")
for cell, (seeds, rec) in CELLS.items():
    ok_new = ok_rec = n = mism = missing = 0
    for s in seeds:
        a, b = rows(f"{O}/{cell}_eps0_e{s}.csv"), rows(rec.format(s))
        if a is None or b is None:
            missing += 1; continue
        n += len(a); ok_new += sum(int(r["success"]) for r in a); ok_rec += sum(int(r["success"]) for r in b)
        mism += sum(key(x) != key(y) for x, y in zip(a, b)) + abs(len(a) - len(b))
    if n:
        print(f"  {cell:11s} new {ok_new}/{n} = {100*ok_new/n:.2f}%   recorded {ok_rec} = {100*ok_rec/n:.2f}%"
              f"   episode mismatches {mism}" + (f"   ({missing} seeds missing)" if missing else ""))
    else:
        print(f"  {cell:11s} not run yet")

print("\n2. PASSIVITY: anchor_probe.py CSV vs plain fsm_sim.py (per episode)")
for cell in ("rand_fp32", "rand_int8"):
    mism = n = 0
    for s in HELD:
        a, b = rows(f"{O}/probe_{cell}_eps0_e{s}.csv"), rows(f"{O}/{cell}_eps0_e{s}.csv")
        if a is None or b is None:
            continue
        n += len(a); mism += sum(key(x) != key(y) for x, y in zip(a, b)) + abs(len(a) - len(b))
    print(f"  {cell:11s} {n} episodes compared, mismatches {mism}")


def probe_summary(stem, seeds, recorded=None):
    tot = Counter(); joints = Counter(); ok = n = 0; blocked_eps = {}
    for s in seeds:
        pr, c = rows(f"{O}/{stem}_e{s}.probe.csv"), rows(f"{O}/{stem}_e{s}.csv")
        if pr is None or c is None:
            continue
        scored = [r for r in c if r["phase"] != "handoff_failed"]
        align = sum((p["final"] == "DONE_OK") != (r["success"] == "1") for p, r in zip(pr, scored))
        tot["align_err"] += align + abs(len(pr) - len(scored))
        n += len(c); ok += sum(int(r["success"]) for r in c)
        for p, r in zip(pr, scored):
            if p["blocked"] == "1":
                tot["blocked"] += 1; blocked_eps.setdefault(s, set()).add(int(r["episode"]))
                # joint_end is the argmin margin; when two joints sit at their stops
                # together it can name the other one.  Joint 5's upper stop is 3.752.
                tot["j5_at_stop"] += float(p["q5_end"]) > 3.732
                if p["pinned"] == "1":
                    tot["pinned"] += 1; joints[int(p["joint_end"])] += 1
            if r["success"] == "0":
                tot["fail"] += 1
                if p["fail_from"] == "TRANSPORT":
                    tot["fail_from_transport"] += 1
    if not n:
        print(f"  {stem}: not run yet"); return {}
    print(f"  {stem:26s} {ok}/{n} = {100*ok/n:.2f}%" + (f" (recorded {recorded})" if recorded else "")
          + f"   failures {tot['fail']}, out of TRANSPORT {tot['fail_from_transport']}"
          f"   BLOCKED {tot['blocked']} = {100*tot['blocked']/n:.2f}%   joint 5 at stop {tot['j5_at_stop']}"
          f"   pinned {tot['pinned']}"
          f" joints {dict(joints)}   probe/CSV alignment errors {tot['align_err']}")
    return blocked_eps


print("\n3. CLASS SIZE (anchor_probe.py at eps 0)")
probe_summary("probe_bi_s0_fp32_eps0", PROT, recorded="1088/1200; blocked class 3.08%")
blocked = {c: probe_summary(f"probe_{c}_eps0", HELD) for c in ("rand_fp32", "rand_int8")}

print("\n4. EFFECT of --rot-anchor-eps 1e-6, paired on seed")
for cell, (seeds, _) in CELLS.items():
    d = []; up = down = tie = 0; gained = lost = 0; gained_blocked = 0
    for s in seeds:
        a, b = rows(f"{O}/{cell}_eps0_e{s}.csv"), rows(f"{O}/{cell}_eps1e-6_e{s}.csv")
        if a is None or b is None:
            continue
        x = sum(int(r["success"]) for r in a); y = sum(int(r["success"]) for r in b)
        d.append(y - x); up += y > x; down += y < x; tie += y == x
        for ra, rb in zip(a, b):
            if ra["success"] == "0" and rb["success"] == "1":
                gained += 1
                gained_blocked += int(ra["episode"]) in blocked.get(cell, {}).get(s, set())
            elif ra["success"] == "1" and rb["success"] == "0":
                lost += 1
    if len(d) == len(seeds):
        m, t = tstat(d)
        print(f"  {cell:11s} {sum(d):+d} episodes = {m:+.2f} pts   t(11)={t:+.2f}   {up}u/{down}d/{tie}t"
              f"   flips: {gained} gained ({gained_blocked} were probe-blocked), {lost} lost")
    else:
        print(f"  {cell:11s} treatment incomplete ({len(d)}/{len(seeds)} seeds)")
