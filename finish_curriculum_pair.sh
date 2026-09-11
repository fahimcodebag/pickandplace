#!/usr/bin/env bash
# Unattended completion of the curriculum pair:
#   1. wait for every curF/curR trainer to exit
#   2. wait for any in-flight evaluation pass to drain (no double-queued jobs)
#   3. evaluate every remaining complete snapshot on the freed machine
#   4. write Results/curriculum_pair/correlation_report.txt
# Process checks filter on /proc/PID/comm, never on command-line text alone:
# pgrep -f also matches the shell that runs it, which cost several self-kills
# in this project.  A power cut kills this script; rerun it -- every step is
# resumable (eval_place_snapshots.sh skips rows already in eval.tsv).
set -u
cd "$(dirname "$0")"
R=Results/curriculum_pair; LOG=$R/finish.log
n_trainers() { for p in $(pgrep -f td3_place_cur 2>/dev/null); do
  case "$(cat /proc/$p/comm 2>/dev/null)" in python*) echo x;; esac; done | wc -l; }
n_eval()     { for p in $(pgrep -f eval_place_snap 2>/dev/null); do
  case "$(cat /proc/$p/comm 2>/dev/null)" in eval_place*) echo x;; esac; done | wc -l; }
echo "$(date '+%F %T') waiting for $(n_trainers) trainer processes" >> $LOG
while [ "$(n_trainers)" -gt 0 ]; do sleep 60; done
echo "$(date '+%F %T') trainers finished; draining in-flight evaluation" >> $LOG
while [ "$(n_eval)" -gt 0 ]; do sleep 30; done
echo "$(date '+%F %T') final evaluation pass" >> $LOG
./eval_place_snapshots.sh 24 >> $R/eval_driver.log 2>&1
echo "$(date '+%F %T') writing report" >> $LOG
{ echo "Curriculum pair: training metric vs end-to-end success -- generated $(date '+%F %T')"
  echo "eval rows: $(wc -l < $R/eval.tsv)"; echo
  /home/fahim/Thesis_fahim/venv/bin/python analyze_place_snapshots.py; } > $R/correlation_report.txt 2>&1
echo "$(date '+%F %T') done -> $R/correlation_report.txt" >> $LOG
