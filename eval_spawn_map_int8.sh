#!/bin/bash
# Fig 8, INT8 variant: success vs spawn pose for the artifact that actually ships,
# on the holdout4 seeds the 95.33% headline used -- so the two are like-for-like.
set -u
cd "$(dirname "$0")"
P=${1:-12}
R=Results/spawn_map_int8; mkdir -p $R/csv; touch $R/eval.tsv
for SD in 13 59 71 137 199 257 331 419 547 601 733 911; do
  awk -F'\t' -v s="$SD" '$2==s && $4>0 {f=1} END{exit !f}' $R/eval.tsv && continue
  echo "$SD 100"
done | xargs -P "$P" -L1 ./eval_spawn_map_int8_job.sh >> $R/eval.tsv
echo "done: $(cat $R/csv/*.csv 2>/dev/null | grep -vc '^episode') scored episodes"
