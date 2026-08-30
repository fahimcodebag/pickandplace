#!/usr/bin/env bash
# Confirm the wclip candidates on eval seeds NEVER used in the sweep.
# wclip8_s0 was the best of 12 cells, so its +6.33 is a selected maximum;
# held-out seeds are the only way to tell a real effect from that selection.
# wclip6_s0 rides along so one lucky cell cannot carry the conclusion.
# Deployed bi_s0 is re-measured on the SAME held-out seeds as the paired baseline.
set -u
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/home/fahim/Thesis_fahim/venv/bin/python
TFPY=/home/fahim/Thesis_fahim/convert_venv/bin/python
P=checkpoints/td3_place/best
PT=qat_output_bi/place_orig_int8.tflite
O=Results/wclip/holdout
# disjoint from the sweep's 7 31 47 89 101 123 211 307 401 503 555 2024
SEEDS="13 59 71 137 199 257 331 419 547 601 733 911"
mkdir -p $O
: > $O/jobs.txt
add () {  # name  ckpt  tflite
  for s in $SEEDS; do
    echo "$PY fsm_sim.py --grasp-ckpt $2 --place-ckpt $P --regrasp --rc-steps 60 --episodes 100 --seed $s --tag ${1}_fp32 --out $O/${1}_fp32_e$s.csv > $O/${1}_fp32_e$s.log 2>&1" >> $O/jobs.txt
    echo "$TFPY fsm_sim.py --int8 --grasp-tflite $3 --place-tflite $PT --grasp-ckpt $2 --place-ckpt $P --regrasp --rc-steps 60 --episodes 100 --seed $s --tag ${1}_int8 --out $O/${1}_int8_e$s.csv > $O/${1}_int8_e$s.log 2>&1" >> $O/jobs.txt
  done
}
add deployed checkpoints/td3_grasp_rand_td3_ln_bi_s0/best qat_output_fix/grasp_bi_noqat_int8.tflite
add wclip8s0 checkpoints/td3_grasp_rand_td3_ln_wclip8_s0/best Results/wclip/grasp_wclip8_s0_int8.tflite
add wclip6s0 checkpoints/td3_grasp_rand_td3_ln_wclip6_s0/best Results/wclip/grasp_wclip6_s0_int8.tflite
n=$(wc -l < $O/jobs.txt); rm -f $O/DONE
setsid nohup bash -c "xargs -a $O/jobs.txt -d '\n' -P 32 -I{} bash -c '{}'; touch $O/DONE" \
  > $O/queue.log 2>&1 < /dev/null &
disown -a
echo "queued $n held-out shards (3 models x {fp32,int8} x 12 fresh seeds)"
