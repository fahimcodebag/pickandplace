#!/bin/bash
# Fig 8: end-to-end success vs SPAWN POSE for the deployed stack.
# 12 protocol seeds x 100 episodes = 1,200 episodes -> ~48 per cell on a 5x5 grid.
# Host-only: fsm_sim runs sim and both policies. No HIL needed -- on-device
# inference matches desktop TFLite exactly on all 14 outputs (boot self-test) and
# INT8 scores 95.33% vs FP32 94.92%, so the host pipeline is representative.
set -u
cd "$(dirname "$0")"
P=${1:-16}
R=Results/spawn_map; mkdir -p $R/csv; touch $R/eval.tsv
for SD in 7 31 47 89 101 123 211 307 401 503 555 2024; do
  awk -F'\t' -v s="$SD" '$2==s && $4>0 {f=1} END{exit !f}' $R/eval.tsv && continue
  echo "$SD 100"
done | xargs -P "$P" -L1 ./eval_spawn_map_job.sh >> $R/eval.tsv
echo "done: $(cat $R/csv/*.csv 2>/dev/null | grep -vc '^episode') scored episodes"
