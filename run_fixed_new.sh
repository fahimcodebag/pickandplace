#!/usr/bin/env bash
# Re-measure the FIXED-SPAWN cells on the validated artifact (smallc64_s5).
# The results table currently reports bi_s0's 100.00%/96.67% for these cells,
# which no longer matches the model in the random-spawn row.
# Same protocol as Results/fixed_ac: 12 eval seeds x 100, A+C rule layer.
set -u
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/home/fahim/Thesis_fahim/venv/bin/python
TFPY=/home/fahim/Thesis_fahim/convert_venv/bin/python
C=checkpoints/td3_grasp_rand_td3_ln_smallc64_s5/best
GT=Results/bigcritic/smallc64_s5_int8.tflite
P=checkpoints/td3_place/best
PT=qat_output_bi/place_orig_int8.tflite
O=Results/fixed_new; mkdir -p $O
SEEDS="7 31 47 89 101 123 211 307 401 503 555 2024"
: > $O/jobs.txt
for s in $SEEDS; do
  echo "$PY fsm_sim.py --fixed-spawn --grasp-ckpt $C --place-ckpt $P --regrasp --rc-steps 60 --episodes 100 --seed $s --tag new_fixed_fp32 --out $O/fp32_e$s.csv > $O/fp32_e$s.log 2>&1" >> $O/jobs.txt
  echo "$TFPY fsm_sim.py --fixed-spawn --int8 --grasp-tflite $GT --place-tflite $PT --grasp-ckpt $C --place-ckpt $P --regrasp --rc-steps 60 --episodes 100 --seed $s --tag new_fixed_int8 --out $O/int8_e$s.csv > $O/int8_e$s.log 2>&1" >> $O/jobs.txt
done
n=$(wc -l < $O/jobs.txt); rm -f $O/DONE
setsid nohup bash -c "xargs -a $O/jobs.txt -d '\n' -P 24 -I{} bash -c '{}'; touch $O/DONE" > $O/queue.log 2>&1 < /dev/null &
disown -a
echo "queued $n fixed-spawn shards at -P 24"
