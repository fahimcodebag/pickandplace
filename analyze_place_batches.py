#!/usr/bin/env python3
"""Held-out comparison of the three cereal place-training batches.

Reads Results/place_batch_holdout/labels.tsv and csv/<md5>_s<seed>.csv
(make_place_batch_holdout_jobs.py + the job list it writes).

  A  = cerES     custom reward + early stopping on end-to-end evaluation
  B' = cerBP     robosuite's staged reward as a potential + success
  C' = cerBPann  B' with shaping annealed toward sparse once training succeeds

Per run: the PEAK snapshot (chosen on the selection seeds 7/31 x 50), best/ (the
trainer's own metric) and the FINAL weights, each scored on 8 held-out seeds x 100.
Identical actors were evaluated once and share their result, so runs that are
bit-identical (a C' run whose anneal never triggered) cannot differ here.
"""
import collections, csv, math, os

D = "Results/place_batch_holdout"
HELD = [101, 123, 211, 307, 401, 503, 555, 2024]
BATCH = {"cerES": "A custom+ES", "cerBP": "B' potential", "cerBPann": "C' annealed"}


def per_seed(md5):
    out = []
    for s in HELD:
        p = f"{D}/csv/{md5}_s{s}.csv"
        if not os.path.exists(p):
            return None
        r = list(csv.DictReader(open(p)))
        out.append(100 * sum(int(x["success"]) for x in r) / len(r))
    return out


def mean(v):
    return sum(v) / len(v)


def paired(a, b):
    d = [x - y for x, y in zip(a, b)]; m = mean(d)
    sd = math.sqrt(sum((x - m) ** 2 for x in d) / (len(d) - 1))
    return m, (m / (sd / math.sqrt(len(d))) if sd else float("nan")), sum(x > 0 for x in d), sum(x < 0 for x in d)


rows = [l.rstrip("\n").split("\t") for l in open(f"{D}/labels.tsv")]
score, hash_of = {}, {}
for label, h, _d in rows:
    v = per_seed(h)
    if v:
        score[label] = v; hash_of[label] = h
SRC = hash_of.get("ref_warm_source|best|-")   # the warm-start actor itself

refs = {k: score.get(f"ref_{k}|best|-") for k in ("warm_source", "bread_transfer")}
print("HELD-OUT end-to-end success, 8 seeds x 100 (cereal, random spawn)")
for k, v in refs.items():
    if v:
        print(f"  reference {k:15s} {mean(v):6.2f}%")

BATCHES = ("cerES", "cerBP", "cerBPann")
table = collections.defaultdict(dict)
for label in score:
    if label.startswith("ref_"):
        continue
    run, kind, ep = label.split("|")
    tag = run.replace("td3_place_", "")
    if tag.split("_")[0] not in BATCHES:      # older cereal runs the glob also matched
        continue
    table[tag][kind] = (score[label], ep, hash_of[label])

print(f"\n  {'run':22s} {'peak':>18s} {'best/':>14s} {'final':>8s}")
per_batch = collections.defaultdict(lambda: collections.defaultdict(list))
unchanged = []
for tag in sorted(table):
    batch = tag.split("_")[0]; kind = "warm" if "_warm_" in tag else "scratch"
    c = table[tag]
    cells = []
    for k in ("peak", "best", "final"):
        if k in c:
            v, ep, h = c[k]
            if h == SRC:
                # best/ never improved on the warm start: this is the STARTING policy,
                # not a training result.  Flagged and excluded from the batch means.
                unchanged.append(f"{tag} {k}")
                cells.append(f"{mean(v):6.2f}%=start")
            else:
                per_batch[batch][k].append(mean(v))
                per_batch[batch + "/" + kind][k].append(mean(v))
                cells.append(f"{mean(v):6.2f}%" + (f" ({ep})" if k == "peak" else ""))
        else:
            cells.append("      --")
    print(f"  {tag:22s} {cells[0]:>18s} {cells[1]:>14s} {cells[2]:>8s}")

if unchanged:
    print(f"  =start: identical to the warm-start actor, excluded from the means below -- {', '.join(unchanged)}")

print(f"\n  {'batch':22s} {'peak':>8s} {'best/':>8s} {'final':>8s}   (best/ over trained checkpoints only)")
for b in ("cerES", "cerBP", "cerBPann"):
    for sub in (b, b + "/scratch", b + "/warm"):
        if per_batch[sub]["peak"]:
            name = BATCH[b] if sub == b else "   " + sub.split("/")[1]

            def cell(k, sub=sub):
                v = per_batch[sub][k]
                if not v:                      # every checkpoint was the warm start
                    return "     -- "
                n = f"(n={len(v)})" if len(v) != len(per_batch[sub]["peak"]) else ""
                return f"{mean(v):7.2f}%{n}"

            print(f"  {name:22s} " + " ".join(cell(k) for k in ("peak", "best", "final")))

print("\nPAIRED on seed, peak snapshots (same seeds, same protocol)")
for a, b in (("cerBP", "cerES"), ("cerBPann", "cerES"), ("cerBPann", "cerBP")):
    for kind in ("scr", "warm"):
        va, vb = [], []
        for s in (10, 11, 12):
            la, lb = f"{a}_{kind}_s{s}", f"{b}_{kind}_s{s}"
            if la in table and lb in table and "peak" in table[la] and "peak" in table[lb]:
                va += table[la]["peak"][0]; vb += table[lb]["peak"][0]
        if va:
            m, t, u, d = paired(va, vb)
            print(f"  {a:10s} - {b:10s} {kind:5s} {m:+7.2f}  t({len(va)-1})={t:+6.2f}  {u}u/{d}d  (3 runs x 8 seeds)")

print("\nSELECTION vs TRAINING METRIC vs LAST WEIGHTS, mean over the 18 batch runs")
for k in ("peak", "best", "final"):
    v = [mean(table[t][k][0]) for t in table if k in table[t] and table[t][k][2] != SRC]
    print(f"  {k:6s} {mean(v):6.2f}%  (n={len(v)} runs)")
