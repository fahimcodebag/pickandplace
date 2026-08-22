#!/usr/bin/env bash
# Mirror training progress into Google Drive so it can be checked from a phone.
#
# Google Drive's G: is a virtual drive created by the Drive client. WSL does not
# auto-mount it, and `mount -t drvfs G: /mnt/g` needs root. Neither is required:
# Windows can read WSL's filesystem over UNC (\\wsl.localhost\<distro>\...), so
# this stages files locally and has PowerShell copy them across. No sudo, no
# mount, nothing for the user to set up.
#
# Writes into "<DRIVE_DIR>":
#   status.txt    plain-text dashboard (opens directly in the Drive app)
#   progress.png  learning curves — fastest thing to read on a phone
#   outcomes.png  done-reason composition
#   runs/<algo>/  episodes.csv, probe.csv, manifest.json
#
# Usage:  ./sync_to_drive.sh [interval_seconds]      (default 60)
set -uo pipefail
cd "$(dirname "$0")"

INTERVAL=${1:-60}
RUNS=${RUNS:-"logs/grasp_rand_*phaseA_s0"}
DRIVE_DIR=${DRIVE_DIR:-'G:\My Drive\thesis_training'}
PLOT_EVERY=${PLOT_EVERY:-5}
DISTRO=${DISTRO:-$(wsl.exe -l -q 2>/dev/null | tr -d '\r\0' | head -1)}
DISTRO=${DISTRO:-Ubuntu}
STAGE=/tmp/drive_stage
PS=/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe
UNC="\\\\wsl.localhost\\${DISTRO}\\tmp\\drive_stage"

[ -x "$PS" ] || { echo "powershell.exe not found at $PS"; exit 1; }
echo "syncing '$RUNS' -> '$DRIVE_DIR' every ${INTERVAL}s  (distro=$DISTRO)"
echo "Ctrl-C to stop"

"$PS" -NoProfile -Command "New-Item -ItemType Directory -Path '$DRIVE_DIR' -Force | Out-Null" 2>/dev/null

cycle=0
while true; do
  cycle=$((cycle + 1))
  rm -rf "$STAGE"; mkdir -p "$STAGE/runs"

  # 1. Plain-text dashboard.
  if python3 monitor_training.py --once --no-color --runs "$RUNS" > /tmp/_st.txt 2>/dev/null; then
    { echo "updated $(date '+%Y-%m-%d %H:%M:%S %Z')"; echo; cat /tmp/_st.txt; } \
      > "$STAGE/status.txt"
  fi

  # 2. Raw CSVs per run.
  for d in $RUNS; do
    [ -d "$d" ] || continue
    algo=$(python3 -c "import json;print(json.load(open('$d/manifest.json'))['algo'])" 2>/dev/null) \
      || algo=$(basename "$d")
    mkdir -p "$STAGE/runs/$algo"
    for f in episodes.csv probe.csv manifest.json; do
      [ -f "$d/$f" ] && cp "$d/$f" "$STAGE/runs/$algo/$f" 2>/dev/null
    done
  done

  # 3. Figures — the most readable artifact on a small screen.
  if [ $((cycle % PLOT_EVERY)) -eq 1 ]; then
    python3 plot_thesis.py --runs "$RUNS" --outdir /tmp/_figs >/dev/null 2>&1 || true
  fi
  [ -f /tmp/_figs/fig1_learning_curves.png ] && cp /tmp/_figs/fig1_learning_curves.png "$STAGE/progress.png"
  [ -f /tmp/_figs/fig2_done_reasons.png ]    && cp /tmp/_figs/fig2_done_reasons.png    "$STAGE/outcomes.png"

  # 4. One PowerShell call copies the whole staging tree across.
  "$PS" -NoProfile -Command "Copy-Item -Path '$UNC\\*' -Destination '$DRIVE_DIR\\' -Recurse -Force -ErrorAction SilentlyContinue" 2>/dev/null

  printf '[%s] cycle %d synced\n' "$(date '+%H:%M:%S')" "$cycle"
  sleep "$INTERVAL"
done
