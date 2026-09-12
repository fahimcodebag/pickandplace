#!/usr/bin/env python3
"""Build the held-out evaluation job list for the three cereal place batches.

For every run of batches A (cerES), B' (cerBP) and C' (cerBPann), three candidates:
  peak    the snapshot with the best end-to-end success on the SELECTION seeds
          (Results/place_es*/eval.tsv, seeds 7 31 x 50 -- the same data batch A's
          early stopping used)
  best    best/ -- the trainer's own selector (frac50 x place50)
  final   actor_td3 -- the weights training ended with
Each is scored on 8 HELD-OUT seeds x 100 that never took part in selection, through
fsm_sim.py with the batches' own protocol (cereal, cereal_alignwarm_s0 grasp,
--regrasp --rc-steps 60 --rot-anchor-eps 1e-6, robosuite criterion, --settle-steps 60).
Reference arms on the same seeds: the warm-start source (cerealscratch_s2/best) and
the bread place actor transferred.

Candidates are DEDUPLICATED BY ACTOR MD5: C' runs whose anneal never triggered are
bit-identical to their B' twin, and peak/best/final often coincide.  Identical
weights are evaluated once and the result is shared, which also makes any difference
between two "identical" rows impossible by construction.

    python make_place_batch_holdout_jobs.py > Results/place_batch_holdout/jobs.txt
"""
import collections, csv, glob, hashlib, os, sys

ROOT = os.path.dirname(os.path.abspath(__file__))
PY = "/home/fahim/Thesis_fahim/venv/bin/python"
OUT = "Results/place_batch_holdout"
HELD = [101, 123, 211, 307, 401, 503, 555, 2024]
GRASP = "checkpoints/td3_grasp_rand_td3_ln_cereal_alignwarm_s0/best"
EVALS = ["Results/place_es/eval.tsv", "Results/place_es_B/eval.tsv", "Results/place_es_C/eval.tsv"]


def md5(path):
    return hashlib.md5(open(path, "rb").read()).hexdigest()[:10]


sel = collections.defaultdict(lambda: [0, 0])
for path in EVALS:
    if not os.path.exists(path):
        continue
    for line in open(path):
        f = line.rstrip("\n").split("\t")
        if len(f) == 6 and f[4] not in ("", "0"):
            sel[(os.path.basename(f[0]), int(f[1]))][0] += int(f[3])
            sel[(os.path.basename(f[0]), int(f[1]))][1] += int(f[4])

candidates = {}          # label -> checkpoint dir
for run in sorted({r for r, _ in sel} | {os.path.basename(d) for d in glob.glob("checkpoints/td3_place_cer*_s1*")}):
    d = os.path.join("checkpoints", run)
    if not os.path.isdir(d):
        continue
    eps = sorted(e for r, e in sel if r == run)
    if eps:
        peak = max(eps, key=lambda e: (sel[(run, e)][0] / sel[(run, e)][1], -e))
        candidates[f"{run}|peak|ep{peak}"] = os.path.join(d, "snapshots", f"ep_{peak:05d}")
    candidates[f"{run}|best|-"] = os.path.join(d, "best")
    candidates[f"{run}|final|-"] = d
candidates["ref_warm_source|best|-"] = "checkpoints/td3_place_cerealscratch_s2/best"
candidates["ref_bread_transfer|best|-"] = "checkpoints/td3_place/best"

by_hash = collections.defaultdict(list)
for label, d in candidates.items():
    actor = os.path.join(d, "actor_td3")
    if os.path.exists(actor):
        by_hash[md5(actor)].append((label, d))
    else:
        print(f"# missing {actor}", file=sys.stderr)

with open(os.path.join(ROOT, OUT, "labels.tsv"), "w") as fh:
    for h, items in sorted(by_hash.items()):
        for label, d in items:
            fh.write(f"{label}\t{h}\t{d}\n")

n = 0
for h, items in sorted(by_hash.items()):
    d = items[0][1]
    for s in HELD:
        out = f"{OUT}/csv/{h}_s{s}.csv"
        if os.path.exists(os.path.join(ROOT, out)):
            continue
        print(f"{PY} fsm_sim.py --grasp-ckpt {GRASP} --place-ckpt {d} --object-type cereal "
              f"--regrasp --rc-steps 60 --rot-anchor-eps 1e-6 --success-criterion robosuite "
              f"--settle-steps 60 --episodes 100 --seed {s} --out {out} > /dev/null 2>&1")
        n += 1
print(f"# {len(candidates)} candidates, {len(by_hash)} unique actors, {n} jobs", file=sys.stderr)
