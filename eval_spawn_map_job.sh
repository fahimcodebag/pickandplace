#!/bin/bash
# One eval seed of the DEPLOYED stack, end-to-end, recording the spawn pose of the
# scored attempt (Fig 8: success vs spawn). Anchor OFF: it costs c2m512_s1 2.75
# FP32 / 2.50 INT8 points (thesis_context 9.17.1), so the deployed configuration
# is eps 0 -- fsm_sim's own default.
cd "$(dirname "$0")"
SEED=$1; EP=${2:-100}
OUT=Results/spawn_map/csv; mkdir -p $OUT
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
/home/fahim/Thesis_fahim/venv/bin/python fsm_sim.py \
  --grasp-ckpt checkpoints/td3_grasp_rand_td3_ln_c2m512_s1/best \
  --place-ckpt checkpoints/td3_place/best --object-type bread \
  --regrasp --rc-steps 60 --rot-anchor-eps 0 --success-criterion robosuite \
  --episodes $EP --seed $SEED --out $OUT/dep_s${SEED}.csv > /dev/null 2>&1
OK=$(tail -n +2 $OUT/dep_s${SEED}.csv 2>/dev/null | awk -F, '{s+=$3} END{print s+0}')
N=$(tail -n +2 $OUT/dep_s${SEED}.csv 2>/dev/null | wc -l)
echo -e "dep\t${SEED}\t${OK}\t${N}"
