#!/bin/bash
# One holdout4 seed of the DEPLOYED INT8 artifact, end-to-end, recording spawn pose.
# Directly comparable to the 95.33% headline (Results/orientation_anchor_c2m512.txt
# 1: random INT8 1144/1200 = 95.33% at --rot-anchor-eps 0, same seeds, same flags).
# INT8 needs the TensorFlow venv; the torch venv cannot load a .tflite interpreter.
cd "$(dirname "$0")"
SEED=$1; EP=${2:-100}
OUT=Results/spawn_map_int8/csv; mkdir -p $OUT
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
/home/fahim/Thesis_fahim/convert_venv/bin/python fsm_sim.py \
  --int8 --grasp-tflite Results/big2m/c2m512_s1_int8.tflite \
  --place-tflite qat_output_bi/place_orig_int8.tflite \
  --grasp-ckpt checkpoints/td3_grasp_rand_td3_ln_c2m512_s1/best \
  --place-ckpt checkpoints/td3_place/best --object-type bread \
  --regrasp --rc-steps 60 --rot-anchor-eps 0 --success-criterion robosuite \
  --episodes $EP --seed $SEED --out $OUT/dep_int8_s${SEED}.csv > /dev/null 2>&1
OK=$(tail -n +2 $OUT/dep_int8_s${SEED}.csv 2>/dev/null | awk -F, '{s+=$3} END{print s+0}')
N=$(tail -n +2 $OUT/dep_int8_s${SEED}.csv 2>/dev/null | wc -l)
echo -e "dep_int8\t${SEED}\t${OK}\t${N}"
