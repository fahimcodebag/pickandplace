#!/bin/bash
# Run batches B' and C' behind batch A without exhausting memory.
#
# Launching all 18 place runs at once OOM-crashed the 98 GB WSL VM
# (Results/place_training_batches.txt).  Measured after the restart: batch A alone
# (6 runs x 9 processes) uses ~42 GB, ~7 GB per run, and each fsm_sim.py evaluator
# ~1 GB.  This queue keeps at most MAX_RUNS place trainers alive, and needs
# MIN_FREE_GB of MemAvailable before each launch.  Runs start in the order below as
# batch A's early-stopped runs free their slots.
#
#   ./run_place_queue.sh [MAX_RUNS=6] [MIN_FREE_GB=20]
#
# The scoring-only watchers for B' and C' start up front and wait for snapshots.
# Live trainers are counted by /proc comm (python*) plus cmdline, never pgrep -f.
set -u
cd "$(dirname "$0")"
MAX=${1:-6}; MINFREE=${2:-20}
PY=/home/fahim/Thesis_fahim/venv/bin/python
LOG=Results/place_queue.log
QUEUE="B:scr_s10 B:scr_s11 B:scr_s12 B:warm_s10 B:warm_s11 B:warm_s12 C:scr_s10 C:scr_s11 C:scr_s12 C:warm_s10 C:warm_s11 C:warm_s12"

alive_runs() {   # distinct place runs with a live python trainer
  for p in /proc/[0-9]*; do
    case "$(cat $p/comm 2>/dev/null)" in
      # sed, not grep -o '[^/ ]*$': that also emits an empty match per line, which
      # sort -u turns into one phantom run (caught by the control before launch).
      # /proc cmdline has no trailing newline: add one, or sed's output runs every
      # run name together onto one line (the second control caught that).
      python*) { tr '\0' ' ' < $p/cmdline 2>/dev/null; echo; } \
                 | sed -n 's|.*train_place\.py.*--place-chkpt-dir [^ ]*/\([^/ ][^/ ]*\).*|\1|p' ;;
    esac
  done | sort -u | wc -l
}
free_gb() { awk '/MemAvailable/ {print int($2 / 1048576)}' /proc/meminfo; }
tag_of()  { case $1 in B) echo cerBP ;; C) echo cerBPann ;; esac; }
runs_of() { for k in scr warm; do for s in 10 11 12; do printf 'checkpoints/td3_place_%s_%s_s%s ' "$(tag_of $1)" $k $s; done; done; }

echo "$(date '+%F %T') queue start: at most $MAX live runs, at least ${MINFREE} GB available per launch" >> $LOG
for B in B C; do
  nohup $PY -u es_watch.py --no-stop --parallel 4 --runs $(runs_of $B) --out Results/place_es_$B \
    > Results/place_es_watch_$B.log 2>&1 &
done
echo "$(date '+%F %T') scoring-only watchers started for B' and C'" >> $LOG

for item in $QUEUE; do
  B=${item%%:*}; R=${item#*:}
  while [ "$(alive_runs)" -ge "$MAX" ] || [ "$(free_gb)" -lt "$MINFREE" ]; do
    sleep 120
  done
  echo "$(date '+%F %T') launch $B $R (live runs $(alive_runs), available $(free_gb) GB)" >> $LOG
  ./run_place_es_batch.sh $B $R >> $LOG 2>&1
  sleep 240   # let the new run's workers load before measuring again
done
echo "$(date '+%F %T') queue empty" >> $LOG
