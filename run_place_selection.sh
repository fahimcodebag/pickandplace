#!/bin/bash
# Does selecting place checkpoints on END-TO-END evaluation rescue a usable policy?
#
# Selection data already exists: every snapshot of curF/curR (bread) and
# cerealscratch (cereal) was scored on seeds 7 31 47 89 x 50
# (Results/curriculum_pair/eval.tsv, Results/cereal_scratch/eval.tsv).  A maximum
# over 16 noisy 200-episode estimates is optimistic, so every candidate is re-scored
# on 8 HELD-OUT seeds x 100 never used to choose it:
#   peak   the snapshot with the highest selection-seed end-to-end success
#   best   best/ -- the trainer's own selector (frac50 x place50)
#   final  actor_td3 -- the weights training ended with
# plus references on the same seeds and protocol:
#   bread  checkpoints/td3_place/best (the deployed place actor), same bi_s0 grasp
#   cereal checkpoints/td3_place/best transferred, cereal grasp
# Same protocol as the selection evals (eval_place_snapshot_job.sh /
# eval_cereal_scratch_job.sh): --regrasp --rc-steps 60 --rot-anchor-eps 1e-6,
# robosuite criterion; cereal also records physical rest (--settle-steps 60).
# All trainers exited 2026-09-11.  Resumable: skips existing CSVs.
set -u
cd "$(dirname "$0")"
P=${1:-24}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/home/fahim/Thesis_fahim/venv/bin/python
O=Results/place_selection; mkdir -p $O/csv
HELD="101 123 211 307 401 503 555 2024"
PEAK_FILE=$O/peaks.txt          # "run episode": argmax selection-seed e2e per run, ties -> earliest
[ -s $PEAK_FILE ] || { echo "missing $PEAK_FILE"; exit 1; }
J=$O/jobs.txt; : > $J
job() {  # label, object, place ckpt dir
  local obj=$2 g flags
  if [ $obj = bread ]; then
    g=checkpoints/td3_grasp_rand_td3_ln_bi_s0/best; flags=""
  else
    g=checkpoints/td3_grasp_rand_td3_ln_cereal_alignwarm_s0/best; flags="--settle-steps 60"
  fi
  [ -f "$3/actor_td3" ] || { echo "missing $3/actor_td3"; return; }
  for s in $HELD; do
    [ -s $O/csv/$1_s$s.csv ] && continue
    echo "$PY fsm_sim.py --grasp-ckpt $g --place-ckpt $3 --object-type $obj --regrasp --rc-steps 60 --rot-anchor-eps 1e-6 --success-criterion robosuite $flags --episodes 100 --seed $s --out $O/csv/$1_s$s.csv > /dev/null 2>&1" >> $J
  done
}
while read run ep; do
  case $run in cerealscratch*) obj=cereal;; *) obj=bread;; esac
  d=checkpoints/td3_place_$run
  job ${run}_peak  $obj $d/snapshots/ep_$(printf %05d $ep)
  job ${run}_best  $obj $d/best
  job ${run}_final $obj $d
done < $PEAK_FILE
job ref_bread_td3place  bread  checkpoints/td3_place/best
job ref_cereal_transfer cereal checkpoints/td3_place/best
echo "jobs: $(wc -l < $J) at -P $P"
xargs -a $J -d '\n' -P $P -I{} bash -c '{}'
echo "done"
