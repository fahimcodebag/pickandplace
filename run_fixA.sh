#!/usr/bin/env bash
# Fix A: lost-grip recovery in TRANSPORT. Measured against the same 12 eval
# seeds as the tuned baseline (FP32 89.58% / INT8 76.08%).
set -u
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
G=checkpoints/td3_grasp_rand_td3_ln_bi_s0/best
P=checkpoints/td3_place/best
GT=qat_output_fix/grasp_bi_noqat_int8.tflite
PT=qat_output_bi/place_orig_int8.tflite
TFPY=/home/fahim/Thesis_fahim/convert_venv/bin/python
O=Results/fixA
for s in 7 31 47 89 101 123 211 307 401 503 555 2024; do
  setsid nohup python fsm_sim.py --grasp-ckpt $G --place-ckpt $P --regrasp \
    --episodes 100 --seed $s --tag fixA_fp32 \
    --out $O/fp32_e$s.csv > $O/fp32_e$s.log 2>&1 < /dev/null &
  setsid nohup $TFPY fsm_sim.py --int8 --grasp-tflite $GT --place-tflite $PT \
    --grasp-ckpt $G --place-ckpt $P --regrasp \
    --episodes 100 --seed $s --tag fixA_int8 \
    --out $O/int8_e$s.csv > $O/int8_e$s.log 2>&1 < /dev/null &
done
disown -a
echo "launched 24 Fix A shards"
