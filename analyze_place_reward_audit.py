#!/usr/bin/env python3
"""Price the place reward modes from Results/place_reward_audit/csv (run_place_reward_audit.sh).

For each curriculum frac: mean accumulated return per outcome class under
  custom        the incumbent potential-shaped reward
  builtin       robosuite's shaped reward, bare
  builtin - c   built-in minus an idle cost c per policy step, over a grid of c
A mode is WELL-PRICED when every success earns more than every non-success class on
average, by a margin (success minus the best failure class, over |best failure|).
Reports the smallest grid cost that restores that ordering at both fracs, and the
controls: returned rewards == R_custom, built-in replica == robosuite reward().
"""
import collections, csv, glob, sys
import numpy as np

D = sys.argv[1] if len(sys.argv) > 1 else "Results/place_reward_audit"
rows = [r for f in sorted(glob.glob(f"{D}/csv/*.csv")) for r in csv.DictReader(open(f))]
if not rows:
    raise SystemExit("no audit CSVs yet")
for r in rows:
    for k in ("R_custom", "R_builtin", "frac", "ctrl_returned_minus_custom", "ctrl_builtin_max_abs",
              "R_bpot_shape", "R_bpot_sparse"):
        if k in r:
            r[k] = float(r[k])
    for k in ("steps", "success", "ctrl_builtin_mismatch_steps"):
        r[k] = int(r[k])

print(f"episodes: {len(rows)}")
print(f"CONTROL returned == R_custom: max |diff| {max(abs(r['ctrl_returned_minus_custom']) for r in rows):.2e}")
print(f"CONTROL builtin replica == robosuite reward(): mismatched steps {sum(r['ctrl_builtin_mismatch_steps'] for r in rows)}, "
      f"max |diff| {max(r['ctrl_builtin_max_abs'] for r in rows):.2e}")

GRID = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5]


def cls(r):
    return "SUCCESS" if r["success"] else r["reason"]


def pricing(v, value):
    by = collections.defaultdict(list)
    for r in v:
        by[cls(r)].append(value(r))
    return {k: (float(np.mean(x)), len(x)) for k, x in by.items()}


def verdict(p):
    if "SUCCESS" not in p:
        return None, None
    fails = {k: m for k, (m, n) in p.items() if k != "SUCCESS"}
    if not fails:
        return True, None
    worst = max(fails, key=fails.get)
    s, f = p["SUCCESS"][0], fails[worst]
    return s > f, (worst, s - f)


ok_costs = {}
for frac in sorted({r["frac"] for r in rows}):
    v = [r for r in rows if r["frac"] == frac]
    print(f"\n=== curriculum frac {frac}  ({len(v)} episodes) ===")
    print("  outcome by policy:", dict(collections.Counter((r["label"], cls(r)) for r in v)))
    modes = [("custom", lambda r: r["R_custom"]), ("builtin", lambda r: r["R_builtin"])]
    modes += [(f"builtin-{c}", (lambda c: lambda r: r["R_builtin"] - c * r["steps"])(c)) for c in GRID[1:]]
    if "R_bpot_shape" in v[0]:
        modes += [(f"bpot w={w}", (lambda w: lambda r: w * r["R_bpot_shape"] + r["R_bpot_sparse"])(w))
                  for w in (1.0, 0.5, 0.1, 0.0)]
    classes = sorted({cls(r) for r in v}, key=lambda k: (k != "SUCCESS", k))
    print(f"  {'mode':14s} " + " ".join(f"{k[:16]:>17s}" for k in classes) + "   success beats all?  margin (vs best failure)")
    for name, fn in modes:
        p = pricing(v, fn)
        good, det = verdict(p)
        if name.startswith("builtin-") and good:
            ok_costs.setdefault(frac, float(name.split("-", 1)[1]))
        cells = " ".join(f"{p[k][0]:9.2f} (n={p[k][1]:3d})" if k in p else f"{'':>17s}" for k in classes)
        print(f"  {name:14s} {cells}   {str(good):5s}  " + (f"{det[1]:+.2f} vs {det[0]}" if det else ""))
    print("  mean policy steps by class:", {k: round(float(np.mean([r['steps'] for r in v if cls(r) == k])), 1) for k in classes})

print("\nSmallest grid idle cost at which success out-earns every failure class:", ok_costs)
