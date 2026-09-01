#!/usr/bin/env bash
# Held-out validation of buf500k_s1 (INT8 95.17% on selection seeds) against
# the incumbent smallc64_s5, on the 12 seeds never used for selection.
set -u
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/home/fahim/Thesis_fahim/venv/bin/python
TFPY=/home/fahim/Thesis_fahim/convert_venv/bin/python
P=checkpoints/td3_place/best; PT=qat_output_bi/place_orig_int8.tflite
O=Results/holdout3; mkdir -p $O
SEEDS="13 59 71 137 199 257 331 419 547 601 733 911"
add () {
  for s in $SEEDS; do
    echo "$PY fsm_sim.py --grasp-ckpt $2 --place-ckpt $P --regrasp --rc-steps 60 --episodes 100 --seed $s --tag ${1}_fp32 --out $O/${1}_fp32_e$s.csv > $O/${1}_fp32_e$s.log 2>&1" >> $O/jobs.txt
    echo "$TFPY fsm_sim.py --int8 --grasp-tflite $3 --place-tflite $PT --grasp-ckpt $2 --place-ckpt $P --regrasp --rc-steps 60 --episodes 100 --seed $s --tag ${1}_int8 --out $O/${1}_int8_e$s.csv > $O/${1}_int8_e$s.log 2>&1" >> $O/jobs.txt
  done
}
: > $O/jobs.txt
add buf500ks1  checkpoints/td3_grasp_rand_td3_ln_buf500k_s1/best  Results/buffer/buf500k_s1_int8.tflite
add buf1000ks0 checkpoints/td3_grasp_rand_td3_ln_buf1000k_s0/best Results/buffer/buf1000k_s0_int8.tflite
add smallc64s5 checkpoints/td3_grasp_rand_td3_ln_smallc64_s5/best Results/bigcritic/smallc64_s5_int8.tflite
n=$(wc -l < $O/jobs.txt); rm -f $O/DONE
setsid nohup bash -c "xargs -a $O/jobs.txt -d '\n' -P 24 -I{} bash -c '{}'; touch $O/DONE" > $O/queue.log 2>&1 < /dev/null &
disown -a; echo "queued $n held-out shards"
