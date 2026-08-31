#!/usr/bin/env bash
# Held-out validation of the campaign's two best cells, on the 12 eval seeds
# never used for selection. Both are best-of-6 maxima; wclip8_s0 previously
# went +6.33 (selected) -> +2.92 (held-out), so shrinkage is expected.
# cont64_s0 rides along as the incumbent reference, re-measured on the SAME
# fresh seeds so a shift in seed difficulty cannot masquerade as an effect.
set -u
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/home/fahim/Thesis_fahim/venv/bin/python
TFPY=/home/fahim/Thesis_fahim/convert_venv/bin/python
P=checkpoints/td3_place/best
PT=qat_output_bi/place_orig_int8.tflite
O=Results/holdout2; mkdir -p $O
SEEDS="13 59 71 137 199 257 331 419 547 601 733 911"
add () {  # name  ckptdir  tflite
  for s in $SEEDS; do
    echo "$PY fsm_sim.py --grasp-ckpt $2 --place-ckpt $P --regrasp --rc-steps 60 --episodes 100 --seed $s --tag ${1}_fp32 --out $O/${1}_fp32_e$s.csv > $O/${1}_fp32_e$s.log 2>&1" >> $O/jobs.txt
    echo "$TFPY fsm_sim.py --int8 --grasp-tflite $3 --place-tflite $PT --grasp-ckpt $2 --place-ckpt $P --regrasp --rc-steps 60 --episodes 100 --seed $s --tag ${1}_int8 --out $O/${1}_int8_e$s.csv > $O/${1}_int8_e$s.log 2>&1" >> $O/jobs.txt
  done
}
: > $O/jobs.txt
add smallc64s5 checkpoints/td3_grasp_rand_td3_ln_smallc64_s5/best Results/bigcritic/smallc64_s5_int8.tflite
add bigc512s0  checkpoints/td3_grasp_rand_td3_ln_bigc512_s0/best  Results/bigcritic/bigc512_s0_int8.tflite
add cont64s0   checkpoints/td3_grasp_rand_td3_ln_cont64_s0/best   Results/widen/cont64_s0_int8.tflite
n=$(wc -l < $O/jobs.txt); rm -f $O/DONE
setsid nohup bash -c "xargs -a $O/jobs.txt -d '\n' -P 24 -I{} bash -c '{}'; touch $O/DONE" > $O/queue.log 2>&1 < /dev/null &
disown -a
echo "queued $n held-out shards at -P 24"
