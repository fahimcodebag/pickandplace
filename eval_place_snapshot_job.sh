#!/bin/bash
# One place-training snapshot x one eval seed, END-TO-END through fsm_sim
# (full transport distance, no curriculum), random spawn, robosuite criterion,
# ROT_ANCHOR_EPS on, Fix A + C -- the same protocol as every end-to-end figure
# in Results/orientation_anchor.txt.
cd "$(dirname "$0")"
LABEL=$1; PCK=$2; SEED=$3; EP=${4:-50}
OUT=Results/curriculum_pair/csv; mkdir -p $OUT
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
/home/fahim/Thesis_fahim/venv/bin/python fsm_sim.py \
  --grasp-ckpt checkpoints/td3_grasp_rand_td3_ln_bi_s0/best \
  --place-ckpt "$PCK" --object-type bread \
  --regrasp --rc-steps 60 --rot-anchor-eps 1e-6 --success-criterion robosuite \
  --episodes $EP --seed $SEED --out $OUT/${LABEL}_s${SEED}.csv > /dev/null 2>&1
OK=$(tail -n +2 $OUT/${LABEL}_s${SEED}.csv 2>/dev/null | awk -F, '{s+=$3} END{print s+0}')
N=$(tail -n +2 $OUT/${LABEL}_s${SEED}.csv 2>/dev/null | wc -l)
echo -e "${LABEL}\t${SEED}\t${OK}\t${N}"
