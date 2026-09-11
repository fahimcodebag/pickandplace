#!/bin/bash
# Price the place-stage reward modes on recorded episodes BEFORE training them
# (thesis_context 9.10).  Cereal, random spawn, cereal_alignwarm_s0 grasp, curriculum
# pinned at 0.3 (the training regime) and 1.0 (full transport), 2 seeds x 20 episodes
# per (policy, frac).  One trajectory is priced under every mode at once
# ("Decomposed state training/place_reward_audit.py"); analyze_place_reward_audit.py
# reads the CSVs.  Resumable: skips existing CSVs.
set -u
cd "$(dirname "$0")"
P=${1:-20}
O=${2:-Results/place_reward_audit}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/home/fahim/Thesis_fahim/venv/bin/python
mkdir -p $O/csv
J=$O/jobs.txt; : > $J
declare -A POL=(
  [good]="actor:../checkpoints/td3_place_cerealscratch_s2/best"
  [collapsed]="actor:../checkpoints/td3_place_cerealscratch_s2"
  [zero]="zero"
  [linger]="linger"
  [creep]="creep"
)
for label in good collapsed zero linger creep; do
  for frac in 0.3 1.0; do
    for seed in 0 1; do
      out=$O/csv/${label}_f${frac}_s${seed}.csv
      [ -s "$out" ] && continue
      echo "cd 'Decomposed state training' && $PY place_reward_audit.py --policy ${POL[$label]} --label $label --frac $frac --episodes 20 --seed $seed --out ../$out > ../$O/csv/${label}_f${frac}_s${seed}.log 2>&1" >> $J
    done
  done
done
echo "jobs: $(wc -l < $J) at -P $P"
xargs -a $J -d '\n' -P $P -I{} bash -c '{}'
echo "done"
