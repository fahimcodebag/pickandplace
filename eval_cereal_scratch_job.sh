#!/bin/bash
# One CEREAL place checkpoint x one eval seed, END-TO-END through fsm_sim.
# Copy of eval_place_snapshot_job.sh (the curriculum pair's, bread) with three
# changes: --object-type cereal; the grasp policy every cereal place run trained
# against (td3_grasp_rand_td3_ln_cereal_alignwarm_s0/best); and --settle-steps 60,
# so each row also records whether the box came to REST in its compartment.
# robosuite's z window scores a cereal box lying flat (z = 0.90) as a miss
# (Results/orientation_anchor.txt).  `success` stays robosuite's criterion --
# comparable with every recorded cereal place figure -- and `rested` is the
# physical one.  Settling runs after the episode is scored and cannot change it.
cd "$(dirname "$0")"
LABEL=$1; PCK=$2; SEED=$3; EP=${4:-50}
OUT=Results/cereal_scratch/csv; mkdir -p $OUT
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
/home/fahim/Thesis_fahim/venv/bin/python fsm_sim.py \
  --grasp-ckpt checkpoints/td3_grasp_rand_td3_ln_cereal_alignwarm_s0/best \
  --place-ckpt "$PCK" --object-type cereal \
  --regrasp --rc-steps 60 --rot-anchor-eps 1e-6 --success-criterion robosuite \
  --settle-steps 60 \
  --episodes $EP --seed $SEED --out $OUT/${LABEL}_s${SEED}.csv > /dev/null 2>&1
OK=$(tail -n +2 $OUT/${LABEL}_s${SEED}.csv 2>/dev/null | awk -F, '{s+=$3} END{print s+0}')
RS=$(tail -n +2 $OUT/${LABEL}_s${SEED}.csv 2>/dev/null | awk -F, '{s+=$4} END{print s+0}')
N=$(tail -n +2 $OUT/${LABEL}_s${SEED}.csv 2>/dev/null | wc -l)
echo -e "${LABEL}\t${SEED}\t${OK}\t${N}\t${RS}"
