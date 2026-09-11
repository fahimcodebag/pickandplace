#!/usr/bin/env python3
"""Held-out test of place-checkpoint selection (run_place_selection.sh).

For each run: the snapshot with the best SELECTION-seed end-to-end success (peak),
the trainer's best/, and the final weights, all re-scored on 8 held-out seeds x 100.
Selection-seed numbers come from the 4 x 50 snapshot evals; a max over 16 noisy
estimates is optimistic, so the held-out column is the one to quote.
"""
import collections, csv, glob, math, os, re

O = "Results/place_selection"
HELD = [101, 123, 211, 307, 401, 503, 555, 2024]


def heldout(label, col="success"):
    per = []
    for s in HELD:
        p = f"{O}/csv/{label}_s{s}.csv"
        if not os.path.exists(p):
            return None
        r = list(csv.DictReader(open(p)))
        per.append(sum(int(x[col] or 0) for x in r) / len(r) * 100)
    return per


def mean(v):
    return sum(v) / len(v)


def paired(a, b):
    d = [x - y for x, y in zip(a, b)]; m = mean(d)
    sd = math.sqrt(sum((x - m) ** 2 for x in d) / (len(d) - 1))
    return m, (m / (sd / math.sqrt(len(d))) if sd else float("nan")), sum(x > 0 for x in d), sum(x < 0 for x in d)


sel = collections.defaultdict(lambda: [0, 0])
for f in ("Results/curriculum_pair/eval.tsv", "Results/cereal_scratch/eval.tsv"):
    for line in open(f):
        x = line.rstrip("\n").split("\t")
        m = re.match(r"((?:cur[FR]|cerealscratch)_s\d)_ep(\d+)$", x[0])
        if m and x[3] not in ("", "0"):
            k = (m[1], int(m[2])); sel[k][0] += int(x[2]); sel[k][1] += int(x[3])

peaks = [l.split() for l in open(f"{O}/peaks.txt")]
refs = {"bread": heldout("ref_bread_td3place"), "cereal": heldout("ref_cereal_transfer")}
print("HELD-OUT end-to-end success, 8 seeds x 100 (selection-seed value in brackets)")
print(f"  {'run':18s} {'peak ep':>7s} {'peak':>15s} {'best/':>7s} {'final':>7s} | peak-best  peak-final  reference")
rows = collections.defaultdict(list)
for run, ep in peaks:
    ep = int(ep); obj = "cereal" if run.startswith("cereal") else "bread"
    pk, bs, fn = heldout(f"{run}_peak"), heldout(f"{run}_best"), heldout(f"{run}_final")
    s_ok, s_n = sel[(run, ep)]
    if not (pk and bs and fn):
        print(f"  {run:18s} incomplete"); continue
    ref = refs[obj]
    rows[obj].append((pk, bs, fn))
    print(f"  {run:18s} {ep:7d} {mean(pk):6.2f} [{100*s_ok/s_n:5.1f}] {mean(bs):7.2f} {mean(fn):7.2f} |"
          f" {mean(pk)-mean(bs):+8.2f}  {mean(pk)-mean(fn):+9.2f}   "
          + (f"{mean(ref):.2f} ({obj})" if ref else "--"))
for obj, v in rows.items():
    pk = [mean(x[0]) for x in v]; bs = [mean(x[1]) for x in v]; fn = [mean(x[2]) for x in v]
    print(f"  mean {obj:13s} {'':7s} {mean(pk):6.2f} {'':7s} {mean(bs):7.2f} {mean(fn):7.2f} |"
          f" {mean(pk)-mean(bs):+8.2f}  {mean(pk)-mean(fn):+9.2f}")
for obj, ref in refs.items():
    if not ref:
        continue
    for run, ep in peaks:
        pk = heldout(f"{run}_peak")
        if pk and (run.startswith("cereal") == (obj == "cereal")):
            m, t, u, d = paired(pk, ref)
            print(f"  {run}_peak - reference {obj}: {m:+.2f}  t(7)={t:+.2f}  {u}u/{d}d")
cer = heldout("cerealscratch_s2_peak", "rested")
if cer:
    print(f"  cerealscratch_s2 peak, came to rest in the compartment: {mean(cer):.2f}%")
