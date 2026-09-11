#!/usr/bin/env python3
"""Cereal, random spawn: trained place policy vs scripted transport vs transfer.

Reads Results/cereal_pairing/csv/<arm>_e<seed>.csv (run_cereal_pairing.sh).
1. CONTROLS   scripted_eps1e-6 and transfer_eps1e-6 against the recorded totals
              1070/1200 (89.17%) and 878/1200 (73.17%), commit 6c4a122.
2. ARMS       scored success (robosuite) and physical rest, per arm.
3. PAIRED     per-seed differences, t(11), seeds up/down/tie.
"""
import csv, math, os

O = "Results/cereal_pairing/csv"
SEEDS = [7, 31, 47, 89, 101, 123, 211, 307, 401, 503, 555, 2024]
ARMS = ["scripted_eps1e-6", "transfer_eps1e-6", "transfer_eps0",
        "trained_eps1e-6", "trained_eps0", "trained_r008_eps1e-6"]
RECORDED = {"scripted_eps1e-6": 1070, "transfer_eps1e-6": 878}


def load(arm):
    ok, rest, n = [], [], 0
    for s in SEEDS:
        p = f"{O}/{arm}_e{s}.csv"
        if not os.path.exists(p):
            return None
        r = list(csv.DictReader(open(p)))
        n += len(r)
        ok.append(sum(int(x["success"]) for x in r))
        rest.append(sum(int(x["rested"] or 0) for x in r))
    return dict(ok=ok, rest=rest, n=n)


def paired(a, b):
    d = [x - y for x, y in zip(a, b)]; m = sum(d) / len(d)
    sd = math.sqrt(sum((x - m) ** 2 for x in d) / (len(d) - 1))
    t = m / (sd / math.sqrt(len(d))) if sd else float("nan")
    return m, t, sum(x > 0 for x in d), sum(x < 0 for x in d), sum(x == 0 for x in d)


D = {a: load(a) for a in ARMS}
print("1. CONTROLS against the recorded CLI totals")
for a, rec in RECORDED.items():
    if D[a]:
        tot = sum(D[a]["ok"])
        print(f"  {a:22s} {tot}/{D[a]['n']} = {100*tot/D[a]['n']:.2f}%   recorded {rec}/1200 = {rec/12:.2f}%"
              f"   {'REPRODUCED' if tot == rec else 'DIFFERS by %+d' % (tot - rec)}")
print("\n2. ARMS (12 seeds x 100)")
for a in ARMS:
    if D[a]:
        print(f"  {a:22s} scored {sum(D[a]['ok'])/12:6.2f}%   came to rest in the compartment {sum(D[a]['rest'])/12:6.2f}%"
              f"   per-seed {D[a]['ok']}")
    else:
        print(f"  {a:22s} incomplete")
print("\n3. PAIRED on seed (percentage points, scored success; rest in brackets)")
for a, b in (("trained_eps1e-6", "scripted_eps1e-6"), ("trained_r008_eps1e-6", "scripted_eps1e-6"),
             ("trained_eps0", "scripted_eps1e-6"), ("trained_eps1e-6", "transfer_eps1e-6"),
             ("trained_eps1e-6", "trained_eps0"), ("transfer_eps1e-6", "transfer_eps0"),
             ("trained_r008_eps1e-6", "trained_eps1e-6")):
    if D[a] and D[b]:
        m, t, u, dn, ti = paired(D[a]["ok"], D[b]["ok"])
        mr, tr, *_ = paired(D[a]["rest"], D[b]["rest"])
        print(f"  {a:22s} - {b:18s} {m:+6.2f}  t(11)={t:+6.2f}  {u}u/{dn}d/{ti}t   [rest {mr:+6.2f}, t(11)={tr:+6.2f}]")
