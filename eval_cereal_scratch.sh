#!/bin/bash
# End-to-end evaluation for the CEREAL from-scratch place arm
# (td3_place_cerealscratch_s*, launched by run_cereal_scratch.sh).  Separate
# from eval_place_snapshots.sh, which globs td3_place_cur[FR]_s* and hard-codes
# bread.  Two job sets, one protocol (eval_cereal_scratch_job.sh, 4 seeds x 50):
#   1. every COMPLETE snapshot (ep_XXXXX, never ep_XXXXX.tmp) -- write-once,
#      so safe while its trainer runs
#   2. the FINAL actor_td3 of each warm-started cereal run, so both sides of the
#      cereal-vs-warm-start comparison are scored identically.  actor_td3 is
#      overwritten in place, so a run whose trainer is still alive is skipped.
# Resumable: skips (label, seed) rows already in eval.tsv with n > 0.
set -u
cd "$(dirname "$0")"
P=${1:-24}
SEEDS="7 31 47 89"
FINALS="pairfix_s0 pairfix_s1 pairfix_s2 cereal_s0 cereal_s1 cereal_s2 cereal_s3 cereal_s4"
R=Results/cereal_scratch; mkdir -p $R; touch $R/eval.tsv
J=$R/eval_jobs.txt; : > $J
# A live trainer writing checkpoints/td3_place_$1?  Filter on /proc comm, never
# on command-line text alone (pgrep -f matches the calling shell).
trainer_alive() {
  for p in /proc/[0-9]*; do
    case "$(cat $p/comm 2>/dev/null)" in
      python*) tr '\0' ' ' < $p/cmdline 2>/dev/null \
                 | grep -q "train_place\.py.*td3_place_$1\b" && return 0;;
    esac
  done
  return 1
}
queue() {  # label ckpt_dir
  for s in $SEEDS; do
    grep -qP "^$1\t${s}\t\d+\t[1-9]" $R/eval.tsv && continue
    echo "$1 $2 $s" >> $J
  done
}
for d in checkpoints/td3_place_cerealscratch_s*/snapshots/ep_[0-9][0-9][0-9][0-9][0-9]; do
  [ -f "$d/actor_td3" ] || continue
  run=$(basename "$(dirname "$(dirname "$d")")"); run=${run#td3_place_}
  ep=$(basename "$d"); queue "${run}_ep${ep#ep_}" "$d"
done
for r in $FINALS; do
  [ -f checkpoints/td3_place_$r/actor_td3 ] || continue
  if trainer_alive "$r"; then echo "skip final_$r: trainer still running"; continue; fi
  queue "final_$r" checkpoints/td3_place_$r
done
echo "jobs: $(wc -l < $J)"
xargs -a $J -P $P -L 1 ./eval_cereal_scratch_job.sh >> $R/eval.tsv
