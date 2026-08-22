#!/usr/bin/env python3
"""Live training dashboard — reads the CSVs, not the buffered stdout logs.

`train_rand.py` flushes episodes.csv every 50 episodes and probe.csv on every
probe, whereas its stdout is block-buffered when redirected to a file, so the
CSVs are the reliable progress source for a run already in flight.

Usage (from any terminal, in the repo root):
    python3 monitor_training.py                      # auto-detect active runs
    python3 monitor_training.py --interval 5
    python3 monitor_training.py --runs "logs/grasp_rand_*_phaseA_s0"
    python3 monitor_training.py --once                # single snapshot, no loop
"""

import argparse
import glob
import json
import os
import time
from datetime import timedelta

import pandas as pd

REASONS = ["success", "timeout_flicker", "timeout_touched", "timeout_no_reach"]
SHORT = {"success": "succ", "timeout_flicker": "flick",
         "timeout_touched": "touch", "timeout_no_reach": "noreach"}

BOLD, DIM, GREEN, YELLOW, RED, RESET = (
    "\033[1m", "\033[2m", "\033[32m", "\033[33m", "\033[31m", "\033[0m")


def fmt_eta(done, total, elapsed_s):
    if done <= 0 or elapsed_s <= 0:
        return "?"
    rate = done / elapsed_s
    if rate <= 0:
        return "?"
    return str(timedelta(seconds=int((total - done) / rate)))


def snapshot(run_dirs):
    rows = []
    for d in run_dirs:
        ep_path = os.path.join(d, "episodes.csv")
        if not os.path.exists(ep_path):
            continue
        try:
            df = pd.read_csv(ep_path)
        except Exception:
            continue
        if df.empty:
            continue
        man = {}
        mp = os.path.join(d, "manifest.json")
        if os.path.exists(mp):
            try:
                man = json.load(open(mp))
            except Exception:
                pass

        last = df.iloc[-1]
        target = int(man.get("episodes", 0)) or int(last["episode"])
        wall = float(last.get("wall_s", 0) or 0)
        n = len(df)
        w = df.tail(100)
        succ100 = w["grasp_success"].mean() * 100 if n else 0.0
        prev = df.iloc[-200:-100]["grasp_success"].mean() * 100 if n >= 200 else None
        trend = "" if prev is None else (
            f"{GREEN}+{succ100 - prev:.0f}{RESET}" if succ100 >= prev
            else f"{RED}{succ100 - prev:.0f}{RESET}")

        tally = ""
        if "done_reason" in df.columns:
            w50 = df.tail(50)["done_reason"]
            tally = " ".join(f"{SHORT[r]}={int((w50 == r).sum())}"
                             for r in REASONS if (w50 == r).sum())

        steps_s = (float(last["env_steps"]) / wall) if wall > 0 else 0.0
        age = time.time() - os.path.getmtime(ep_path)
        rows.append(dict(
            algo=man.get("algo", os.path.basename(d)),
            ep=int(last["episode"]), target=target,
            succ=succ100, trend=trend,
            best=float(last.get("best_metric", 0) or 0),
            score=float(w["score"].mean()),
            sps=steps_s,
            eta=fmt_eta(int(last["episode"]), target, wall),
            tally=tally, age=age))
    return rows


def render(rows, runs_pattern):
    out = []
    out.append(f"{BOLD}Random-spawn grasp training{RESET}   "
               f"{DIM}{time.strftime('%H:%M:%S')}   {runs_pattern}{RESET}")
    out.append("")
    if not rows:
        out.append(f"  {YELLOW}no runs with data yet{RESET} "
                   f"{DIM}(CSVs flush every 50 episodes){RESET}")
        return "\n".join(out)

    hdr = (f"  {'algo':<8} {'episode':>13} {'succ100':>9} {'best':>7} "
           f"{'score':>8} {'st/s':>7} {'eta':>10}  outcomes(50)")
    out.append(BOLD + hdr + RESET)
    out.append("  " + "-" * (len(hdr) - 2 + 14))
    for r in sorted(rows, key=lambda x: -x["succ"]):
        stale = f" {YELLOW}(stale {r['age']:.0f}s){RESET}" if r["age"] > 180 else ""
        pct = f"{r['ep']}/{r['target']}"
        col = GREEN if r["succ"] >= 70 else (YELLOW if r["succ"] >= 40 else "")
        end = RESET if col else ""      # no stray reset when uncoloured
        trend = f"{r['trend']:>4}" if r["trend"] else " " * 4
        out.append(
            f"  {r['algo']:<8} {pct:>13} {col}{r['succ']:>8.1f}%{end}"
            f"{trend} {r['best']:>7.3f} {r['score']:>8.1f} "
            f"{r['sps']:>7.0f} {r['eta']:>10}  {DIM}{r['tally']}{RESET}{stale}")
    out.append("")
    out.append(f"  {DIM}succ100 = rolling success over last 100 episodes; "
               f"trend vs the 100 before that{RESET}")
    out.append(f"  {DIM}best    = difficulty-weighted rolling metric "
               f"(the save criterion){RESET}")
    return "\n".join(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", default="logs/grasp_rand_*")
    p.add_argument("--interval", type=float, default=10.0)
    p.add_argument("--once", action="store_true")
    p.add_argument("--no-color", action="store_true",
                   help="plain text (for files synced to Drive / phone viewing)")
    args = p.parse_args()

    if args.no_color:
        global BOLD, DIM, GREEN, YELLOW, RED, RESET
        BOLD = DIM = GREEN = YELLOW = RED = RESET = ""

    while True:
        dirs = sorted(glob.glob(args.runs))
        text = render(snapshot(dirs), args.runs)
        if args.once:
            print(text)
            return
        print("\033[H\033[J" + text, flush=True)   # clear + home
        try:
            time.sleep(args.interval)
        except KeyboardInterrupt:
            print()
            return


if __name__ == "__main__":
    main()
