#!/usr/bin/env bash
# Mirror training progress into Google Drive so it can be checked from a phone.
#
# Writes three things into <DEST>:
#   status.txt    plain-text dashboard (open directly in the Drive app)
#   progress.png  learning curves — the fastest thing to read on a phone
#   runs/<algo>/  episodes.csv, probe.csv, manifest.json for later analysis
#
# Google Drive's G: is a virtual drive; WSL does not auto-mount it. Mount once:
#   sudo mkdir -p /mnt/g && sudo mount -t drvfs G: /mnt/g
# This script waits for the mount rather than failing if it is not ready.
#
# Usage:  ./sync_to_drive.sh [interval_seconds]        (default 60)
#         DEST="/mnt/g/My Drive/thesis_training" ./sync_to_drive.sh
set -uo pipefail
cd "$(dirname "$0")"

DEST=${DEST:-"/mnt/g/My Drive/thesis_training"}
INTERVAL=${1:-60}
RUNS=${RUNS:-"logs/grasp_rand_*phaseA_s0"}
PLOT_EVERY=${PLOT_EVERY:-5}          # regenerate the figure every N cycles

echo "syncing '$RUNS' -> '$DEST' every ${INTERVAL}s (Ctrl-C to stop)"

waited=0
until [ -d "$(dirname "$DEST")" ]; do
  [ $((waited % 300)) -eq 0 ] && echo "waiting for Drive mount at $(dirname "$DEST") ..."
  sleep 15; waited=$((waited + 15))
done
mkdir -p "$DEST/runs" || { echo "cannot create $DEST"; exit 1; }

cycle=0
while true; do
  cycle=$((cycle + 1))

  # 1. Plain-text dashboard, written atomically so the phone never reads a
  #    half-written file mid-sync.
  if python3 monitor_training.py --once --no-color --runs "$RUNS" \
        > /tmp/_status.txt 2>/dev/null; then
    { echo "updated $(date '+%Y-%m-%d %H:%M:%S %Z')"; echo;
      cat /tmp/_status.txt; } > /tmp/_status2.txt
    cp /tmp/_status2.txt "$DEST/status.txt" 2>/dev/null
  fi

  # 2. Raw CSVs per run (small; safe to copy every cycle).
  for d in $RUNS; do
    [ -d "$d" ] || continue
    algo=$(python3 -c "import json,sys;print(json.load(open('$d/manifest.json'))['algo'])" 2>/dev/null) \
      || algo=$(basename "$d")
    mkdir -p "$DEST/runs/$algo"
    for f in episodes.csv probe.csv manifest.json; do
      [ -f "$d/$f" ] && cp "$d/$f" "$DEST/runs/$algo/$f" 2>/dev/null
    done
  done

  # 3. Learning-curve figure — the most readable artifact on a small screen.
  if [ $((cycle % PLOT_EVERY)) -eq 1 ]; then
    if python3 plot_thesis.py --runs "$RUNS" --outdir /tmp/_figs >/dev/null 2>&1; then
      [ -f /tmp/_figs/fig1_learning_curves.png ] && \
        cp /tmp/_figs/fig1_learning_curves.png "$DEST/progress.png" 2>/dev/null
      [ -f /tmp/_figs/fig2_done_reasons.png ] && \
        cp /tmp/_figs/fig2_done_reasons.png "$DEST/outcomes.png" 2>/dev/null
    fi
  fi

  sleep "$INTERVAL"
done
