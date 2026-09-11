#!/usr/bin/env bash
# Evaluate every COMPLETE snapshot (ep_XXXXX, never ep_XXXXX.tmp) of the
# curriculum pair. Resumable: skips (label, seed) rows already in eval.tsv.
# Snapshots are write-once, so this is safe to run while training continues.
set -u
cd "$(dirname "$0")"
P=${1:-20}
SEEDS="7 31 47 89"
R=Results/curriculum_pair; mkdir -p $R; touch $R/eval.tsv
J=$R/eval_jobs.txt; : > $J
for d in checkpoints/td3_place_cur[FR]_s*/snapshots/ep_[0-9][0-9][0-9][0-9][0-9]; do
  [ -f "$d/actor_td3" ] || continue
  run=$(basename "$(dirname "$(dirname "$d")")"); run=${run#td3_place_}
  ep=$(basename "$d"); ep=${ep#ep_}
  label="${run}_ep${ep}"
  for s in $SEEDS; do
    grep -qP "^${label}\t${s}\t" $R/eval.tsv && continue
    echo "$label $d $s" >> $J
  done
done
echo "jobs: $(wc -l < $J)"
xargs -a $J -P $P -L 1 ./eval_place_snapshot_job.sh >> $R/eval.tsv
