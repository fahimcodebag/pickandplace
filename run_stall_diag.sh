#!/usr/bin/env bash
# Instrument the carry trajectory to split TRANSPORT horizon-outs into causes.
set -u
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
G=checkpoints/td3_grasp_rand_td3_ln_bi_s0/best
P=checkpoints/td3_place/best
GT=qat_output_fix/grasp_bi_noqat_int8.tflite
PT=qat_output_bi/place_orig_int8.tflite
O=Results/stall_diag
for s in 7 31 47 89 101 123 211 307 401 503 555 2024; do
  setsid nohup python stall_diag.py --grasp-ckpt $G --place-ckpt $P \
    --episodes 100 --seed $s --tag fp32_rand \
    --out $O/fp32_e$s.csv > $O/fp32_e$s.log 2>&1 < /dev/null &
  setsid nohup /home/fahim/Thesis_fahim/convert_venv/bin/python stall_diag.py --int8 --grasp-tflite $GT --place-tflite $PT \
    --episodes 100 --seed $s --tag int8_rand \
    --out $O/int8_e$s.csv > $O/int8_e$s.log 2>&1 < /dev/null &
done
disown -a
echo "launched 24 shards"
