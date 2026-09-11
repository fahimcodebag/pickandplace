#!/usr/bin/env python3
"""Cereal from scratch vs warm-started cereal place training, keyed on transitions.

QUESTION (Results/place_random_spawn_investigation.txt, section 7).  Every
random-spawn place run that collapsed was cereal, warm-started, or both.
td3_place_cerealscratch_s* is the bread curR arm with one change, the object.
Collapses like the warm-started cereal runs -> the failure is cereal.
Healthy -> the failure was warm-starting.

ARMS
  scratch  cerealscratch_s0-2  cereal, from scratch                  wrapper @ 0fc9a20
  pairfix  pairfix_s0-2        cereal, actor+critics from            wrapper @ 0fc9a20
                               cereal_s2/best, no random warmup      <- code-matched
  default  cereal_s0-4         cereal, actor from bread td3_place,   wrapper BEFORE 0fc9a20
                               critics fresh                         <- code-confounded
  curR     curR_s0-2           bread, from scratch (reference)       wrapper @ 0fc9a20
All four use the default recipe (buffer 200k, batch 512, critic 64x32, LayerNorm).
`default` launched 2026-09-09 13:24-15:35; osc_anchor.py did not exist until
14:01 and place_env_wrapper.py was last edited 2026-09-10 13:35.  So at least
cereal_s0-2 trained with a[3:6] = 0 on the POLICY path (the orientation-anchor
defect) and a grasp-rollout cap of 200 rather than 70.  pairfix and scratch
launched after both edits.

AXIS.  Transitions collected (mem_cntr), not episode: episode length depends on
object, curriculum and policy.  time_step == mem_cntr for every arm except
pairfix, whose --skip-warmup starts time_step at 10,000 with an acting actor.
Exact where recorded -- snapshot meta.json every 250 episodes, "Checkpoint at"
lines every 500 episodes until the buffer caps, final mem_cntr from
replay_buffer.npz -- and linearly interpolated between.  For the warm runs the
stretch from the cap (~episode 4,500) to the end is interpolated, but that lies
beyond the range the scratch arm covers.

Reads logs/train_place_<run>.log, checkpoints/td3_place_<run>/, and
Results/cereal_scratch/eval.tsv (written by eval_cereal_scratch.sh).
"""
import collections, glob, hashlib, json, math, os, re
import numpy as np

R = "Results/cereal_scratch"
BUF_CAP = 200_000
LEARN_GATE = 512 * 10      # td3.py learn(): mem_cntr < batch_size * 10 -> return
WARMUP = 10_000            # train_place.py: actor acts once time_step >= warmup
BIN = 25_000
KEYS = ("place50", "frac50", "metric", "time_step")
MIN_N = 5
ARMS = (
    ("scratch", [f"cerealscratch_s{i}" for i in range(3)], "cereal, from scratch"),
    ("pairfix", [f"pairfix_s{i}" for i in range(3)],
     "cereal, warm actor+critics (cereal_s2/best), no warmup -- code-matched"),
    ("default", [f"cereal_s{i}" for i in range(5)],
     "cereal, warm actor (bread td3_place), fresh critics -- PRE-0fc9a20 wrapper"),
    ("curR", [f"curR_s{i}" for i in range(3)], "bread, from scratch -- reference"),
)

BEST_RE = re.compile(r"^Episode\s+(\d+) \| ★ BEST POLICY! frac50=([\d.]+) place50=\s*([\d.]+)%")
EP_RE = re.compile(r"^Episode\s+(\d+) \| Score:")
OUT_RE = re.compile(r"^\s+last 50 outcomes: (.*)$")
FRAC_RE = re.compile(r"^\s+curriculum frac \(50\): min=[\d.]+ mean=([\d.]+)")
CKPT_RE = re.compile(r"^Checkpoint at (\d+): .*\(buffer: ([\d,]+) transitions\)")
TOTAL_RE = re.compile(r"^Total episodes:\s+(\d+)")


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


def parse_log(run):
    """50-episode records (episode, place50, frac50), buffer counts, total."""
    recs, ckpts, total = [], {}, None
    ep = succ = None
    with open(f"logs/train_place_{run}.log", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            m = BEST_RE.match(line)
            if m:   # a BEST line replaces the normal block and carries both metrics
                recs.append((int(m[1]), float(m[3]) / 100, float(m[2]))); ep = None
                continue
            m = EP_RE.match(line)
            if m:
                ep, succ = int(m[1]), None; continue
            m = OUT_RE.match(line)
            if m and ep is not None:
                tally = dict(kv.split(":", 1) for kv in m[1].split() if ":" in kv)
                succ = int(tally.get("success", 0)); continue
            m = FRAC_RE.match(line)
            if m and ep is not None and succ is not None:
                recs.append((ep, succ / 50, float(m[1]))); ep = None; continue
            m = CKPT_RE.match(line)
            if m:
                ckpts[int(m[1])] = int(m[2].replace(",", "")); continue
            m = TOTAL_RE.match(line)
            if m:
                total = int(m[1])
    return recs, ckpts, total


def snapshots(run):
    out = {}
    for d in glob.glob(f"checkpoints/td3_place_{run}/snapshots/ep_[0-9][0-9][0-9][0-9][0-9]"):
        if not os.path.exists(os.path.join(d, "meta.json")):
            continue
        m = json.load(open(os.path.join(d, "meta.json")))
        m["md5"] = hashlib.md5(open(os.path.join(d, "actor_td3"), "rb").read()).hexdigest()[:8]
        out[m["episode"]] = m
    return out


def final_mem(run):
    p = f"checkpoints/td3_place_{run}/replay_buffer.npz"
    if not os.path.exists(p):
        return None
    with np.load(p) as z:
        return int(z["mem_cntr"][0])


def load_run(run):
    recs, ckpts, total = parse_log(run)
    snaps = snapshots(run)
    anchors, checks = {0: 0}, []
    for e, m in snaps.items():
        anchors[e] = m["mem_cntr"]
    for e, b in ckpts.items():
        if b >= BUF_CAP:
            continue
        if e in snaps:
            checks.append(abs(snaps[e]["mem_cntr"] - b))   # positive control
        else:
            anchors[e] = b
    fm = final_mem(run) if total else None
    if fm:
        anchors[total] = fm
    xs = np.array(sorted(anchors), float); ys = np.array([anchors[x] for x in sorted(anchors)], float)
    rows = [(e, float(np.interp(e, xs, ys)), p, f) for e, p, f in sorted(recs) if e <= xs.max()]
    return dict(run=run, rows=rows, snaps=snaps, total=total, final_mem=fm,
                last_anchor=(int(xs.max()), int(ys.max())), checks=checks,
                exact=len(anchors) - 1)


def fmt(v, w, spec):
    return " " * w if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:{w}{spec}}"


runs = {r: load_run(r) for _, rs, _ in ARMS for r in rs
        if os.path.exists(f"logs/train_place_{r}.log")}

print("TRANSITION AXIS")
for r, d in runs.items():
    c = d["checks"]
    print(f"  {r:18s} anchors {d['exact']:3d}  last at ep {d['last_anchor'][0]:5d} = "
          f"{d['last_anchor'][1]:7,d} transitions  total ep {d['total'] or '-':>5}"
          + (f"  | log-vs-snapshot mem_cntr max |diff| {max(c)} over {len(c)}" if c else ""))

scratch_ends = [runs[r]["last_anchor"][1] for r in ARMS[0][1] if r in runs]
R_COMMON = min(scratch_ends) if scratch_ends else 250_000
NB = R_COMMON // BIN      # full bins only: a partial last bin holds a few episodes of one seed
print(f"\nCommon range = shortest scratch run = {R_COMMON:,} transitions; {NB} full bins of {BIN // 1000}k.")


def binned(d, idx):
    out = []
    for k in range(NB):
        v = [row[idx] for row in d["rows"] if k * BIN <= row[1] < (k + 1) * BIN]
        out.append(float(np.mean(v)) if v else float("nan"))
    return out


for title, idx, w, spec, scale in (("SUCCESS per 50 training episodes (%)", 2, 4, ".0f", 100),
                                   ("CURRICULUM frac, 50-episode mean", 3, 4, ".2f", 1)):
    print(f"\n{title}, by transitions collected (upper bin edge)")
    print("  " + " " * 18 + "".join(f"{(k + 1) * BIN // 1000:>4d}k" for k in range(NB)))
    for arm, rs, desc in ARMS:
        per = []
        for r in rs:
            if r not in runs:
                continue
            b = [x * scale for x in binned(runs[r], idx)]; per.append(b)
            print(f"  {r:18s}" + "".join(" " + fmt(x, w, spec) for x in b))
        if per:
            mean = np.nanmean(np.array(per), axis=0) if len(per) > 1 else per[0]
            print(f"  {'> ' + arm + ' mean':18s}" + "".join(" " + fmt(float(x), w, spec) for x in mean))
        print()

print("SUMMARY.  'learning' = transitions in [10k, common range]; blocks are 50 episodes.")
print(f"  {'run':18s} {'mean succ':>9s} {'min block':>9s} {'max frac':>8s} {'frac@end':>8s} |"
      f" {'whole run: max frac':>19s} {'last 500 ep succ':>16s} {'final frac':>10s}")
arm_sum = collections.defaultdict(list)
for arm, rs, desc in ARMS:
    for r in rs:
        if r not in runs:
            continue
        rows = runs[r]["rows"]
        win = [x for x in rows if WARMUP <= x[1] <= R_COMMON]
        if not win:
            continue
        tail = [x for x in rows if x[0] > rows[-1][0] - 500]
        s = (100 * np.mean([x[2] for x in win]), 100 * min(x[2] for x in win),
             max(x[3] for x in win), win[-1][3],
             max(x[3] for x in rows), 100 * np.mean([x[2] for x in tail]), rows[-1][3])
        arm_sum[arm].append(s)
        print(f"  {r:18s} {s[0]:9.1f} {s[1]:9.0f} {s[2]:8.2f} {s[3]:8.2f} | {s[4]:19.2f} {s[5]:16.1f} {s[6]:10.2f}")
    if arm_sum[arm]:
        m = np.mean(np.array(arm_sum[arm]), axis=0)
        print(f"  {'> ' + arm + ' mean':18s} {m[0]:9.1f} {m[1]:9.0f} {m[2]:8.2f} {m[3]:8.2f} |"
              f" {m[4]:19.2f} {m[5]:16.1f} {m[6]:10.2f}   ({desc})")
    print()

# --- end-to-end -------------------------------------------------------------
ev = collections.defaultdict(lambda: [0, 0, 0])
if os.path.exists(os.path.join(R, "eval.tsv")):
    for line in open(os.path.join(R, "eval.tsv")):
        f = line.rstrip("\n").split("\t")
        if len(f) < 4 or f[3] in ("", "0"):
            continue
        k = ev[f[0]]; k[0] += int(f[2]); k[1] += int(f[3]); k[2] += int(f[4]) if len(f) > 4 else 0

print("END-TO-END through fsm_sim, cereal, random spawn, 4 seeds x 50 per checkpoint.")
print("  e2e = robosuite criterion (comparable with recorded figures); rest = came to rest in the compartment.")
pooled = collections.defaultdict(list); finals = []
for r in ARMS[0][1]:
    if r not in runs:
        continue
    local = collections.defaultdict(list); last = None
    print(f"  {r:18s}   ep  transitions  state     frac50 place50 metric |   e2e    rest   n    actor")
    for e in sorted(runs[r]["snaps"]):
        m = runs[r]["snaps"][e]; ok, n, rest = ev.get(f"{r}_ep{e:05d}", (0, 0, 0))
        state = ("init" if m["mem_cntr"] < LEARN_GATE
                 else "acting" if m["time_step"] >= WARMUP else "learning")
        e2e = 100 * ok / n if n else float("nan")
        print(f"  {'':18s} {e:5d}  {m['mem_cntr']:11,d}  {state:8s}  {m['frac50']:.2f}  "
              f"{100 * m['place50']:5.1f}  {m['metric']:.3f} | {fmt(e2e, 5, '.1f')}% "
              f"{fmt(100 * rest / n if n else float('nan'), 5, '.1f')}% {n:3d}  {m['md5']}")
        if n and state != "init":
            for dst in (pooled, local):
                dst["e2e"].append(e2e)
                for key in KEYS:
                    dst[key].append(m[key])
            last = (e, m["mem_cntr"], e2e, 100 * rest / n)
    k = len(local["e2e"])
    if k >= MIN_N:
        print(f"    within {r} (n={k}) Spearman vs e2e: " + "  ".join(
            f"{key} {corr(rank(local[key]), rank(local['e2e'])):+.2f}" for key in KEYS))
    if last:
        finals.append((r,) + last)
k = len(pooled["e2e"])
if k >= 3:
    print(f"  POOLED scratch (n={k})" + ("" if k >= MIN_N else "  -- too few to read"))
    for key in KEYS:
        print(f"    {key:9s} vs e2e   Spearman {corr(rank(pooled[key]), rank(pooled['e2e'])):+.2f}"
              f"   Pearson {corr(pooled[key], pooled['e2e']):+.2f}")

print("\n  Latest scratch snapshot vs FINAL actor_td3 of the warm runs (same protocol):")
print(f"  {'checkpoint':26s} {'arm':8s} {'episodes':>8s} {'transitions':>11s} {'e2e':>6s} {'rest':>6s} {'n':>4s}")
for r, e, mc, e2e, rest in finals:
    print(f"  {r + f'@ep{e}':26s} {'scratch':8s} {e:8d} {mc:11,d} {e2e:5.1f}% {rest:5.1f}% {200:4d}")
for arm, rs, _ in ARMS[1:3]:
    vals = []
    for r in rs:
        ok, n, rest = ev.get(f"final_{r}", (0, 0, 0))
        if not n or r not in runs:
            continue
        vals.append(100 * ok / n)
        print(f"  {'final_' + r:26s} {arm:8s} {runs[r]['total'] or 0:8d} {runs[r]['final_mem'] or 0:11,d} "
              f"{100 * ok / n:5.1f}% {100 * rest / n:5.1f}% {n:4d}")
    if vals:
        print(f"  {'> ' + arm + ' mean':26s} {'':8s} {'':8s} {'':11s} {np.mean(vals):5.1f}%")
if finals:
    print(f"  {'> scratch mean (latest)':26s} {'':8s} {'':8s} {'':11s} {np.mean([f[3] for f in finals]):5.1f}%")
