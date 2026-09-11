#!/usr/bin/env python3
"""Training metric vs end-to-end success, per place-training snapshot.

Reads checkpoints/td3_place_cur{F,R}_s*/snapshots/ep_*/meta.json (written by
train_place.py --snapshot-every) and Results/curriculum_pair/eval.tsv (written
by eval_place_snapshots.sh).

WITHIN-RUN vs POOLED.  The question is whether the signal training optimises and
selects on tracks deployed success AS TRAINING PROCEEDS.  That is a within-run
correlation over time.  Pooling snapshots across seeds also mixes in seed-to-seed
differences (one seed collapsed, another not), which can create or hide a
correlation that no single run has.  Both are reported; within-run is primary.

WHY IT IS KEYED ON TRAINING PROGRESS, NOT EPISODE.  TD3's learn() is a no-op until
mem_cntr >= batch_size * 10 (td3.py), i.e. 5,120 transitions at batch 512, and
the actor only selects actions once time_step >= warmup (10,000).  Fixed-spawn
episodes start inside the release radius, so the scaffolding finishes them in a
few steps and the buffer fills slowly; random-spawn episodes run longer.
Measured: at episode 500 the fixed arm had 3,620 transitions (no learning yet)
while a random seed had 13,105 (already acting).

Snapshots taken before learning started are the seeded INITIALISATION -- their
actor hashes are identical across episodes and across arms at the same seed --
so they are reported but excluded from every correlation.
"""
import collections, glob, hashlib, json, os, re
import numpy as np

R = "Results/curriculum_pair"
LEARN_GATE = 512 * 10      # td3.py learn(): mem_cntr < batch_size * 10 -> return
WARMUP = 10000             # train_place.py: actor acts once time_step >= warmup
KEYS = ("place50", "frac50", "metric", "time_step")
MIN_N = 5

def rank(a):
    a = np.asarray(a, float); r = np.empty(len(a)); r[a.argsort()] = np.arange(len(a))
    for v in np.unique(a):
        i = np.where(a == v)[0]; r[i] = r[i].mean()
    return r

def corr(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3 or x.std() == 0 or y.std() == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])

snaps = {}
for d in glob.glob("checkpoints/td3_place_cur[FR]_s*/snapshots/ep_[0-9][0-9][0-9][0-9][0-9]"):
    run = d.split(os.sep)[1].replace("td3_place_", "")
    m = json.load(open(os.path.join(d, "meta.json")))
    m["md5"] = hashlib.md5(open(os.path.join(d, "actor_td3"), "rb").read()).hexdigest()[:8]
    snaps[(run, m["episode"])] = m

ev = collections.defaultdict(lambda: [0, 0])
if os.path.exists(os.path.join(R, "eval.tsv")):
    for line in open(os.path.join(R, "eval.tsv")):
        f = line.rstrip("\n").split("\t")
        if len(f) != 4 or f[3] in ("", "0"):
            continue
        mo = re.match(r"(cur[FR]_s\d+)_ep(\d+)$", f[0])
        if mo:
            k = (mo.group(1), int(mo.group(2))); ev[k][0] += int(f[2]); ev[k][1] += int(f[3])

for arm, name in (("curF", "FIXED spawn"), ("curR", "RANDOM spawn")):
    print(f"\n=== {name} ===")
    pooled = collections.defaultdict(list); finals = []
    for run in sorted({r for r, _ in snaps if r.startswith(arm)}):
        local = collections.defaultdict(list); last = None
        print(f"  {run}   ep  time_step  state      frac50 place50 metric | end-to-end  actor")
        for (r, e) in sorted(k for k in snaps if k[0] == run):
            m = snaps[(r, e)]; ok, n = ev.get((r, e), (0, 0))
            state = ("init" if m["mem_cntr"] < LEARN_GATE
                     else "acting" if m["time_step"] >= WARMUP else "learning")
            e2e = 100 * ok / n if n else float("nan")
            print(f"          {e:5d}  {m['time_step']:9d}  {state:9s}  {m['frac50']:.2f}  "
                  f"{100*m['place50']:5.1f}  {m['metric']:.3f} |  {e2e:5.1f}% n={n:<3d} {m['md5']}")
            if n and state != "init":
                for dst in (pooled, local):
                    dst["e2e"].append(e2e)
                    for key in KEYS: dst[key].append(m[key])
                last = (e, e2e, m["frac50"])
        k = len(local["e2e"])
        if k >= MIN_N:
            print(f"    within {run} (n={k}) Spearman vs end-to-end: " + "  ".join(
                f"{key} {corr(rank(local[key]), rank(local['e2e'])):+.2f}" for key in KEYS))
        else:
            print(f"    within {run}: n={k} trained+scored snapshots, too few for a correlation")
        if last:
            finals.append((run,) + last)
    k = len(pooled["e2e"])
    print(f"  POOLED over seeds (n={k})" + ("" if k >= MIN_N else "  -- too few to read"))
    if k >= 3:
        for key in KEYS:
            print(f"    {key:9s} vs end-to-end   Spearman {corr(rank(pooled[key]), rank(pooled['e2e'])):+.2f}"
                  f"   Pearson {corr(pooled[key], pooled['e2e']):+.2f}")
    if finals:
        print("  latest scored snapshot per seed: " + "   ".join(
            f"{f[0]}@ep{f[1]} {f[2]:.1f}% (frac {f[3]:.2f})" for f in finals)
            + f"   mean {np.mean([f[2] for f in finals]):.1f}%")
