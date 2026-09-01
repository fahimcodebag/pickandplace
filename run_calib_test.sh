#!/usr/bin/env bash
# CONFOUND TEST: is the buffer gain policy or CALIBRATION?
# qat_and_convert.py calibrates INT8 on the run's own replay_buffer.npz, so the
# 1M arm also got 5x more diverse calibration states. Cross the two factors:
# convert each 200k-trained actor using a 1M run's buffer, and vice versa.
# If calibration is the driver, the 200k actor should recover most of the gain.
set -u
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/home/fahim/Thesis_fahim/venv/bin/python
TFPY=/home/fahim/Thesis_fahim/convert_venv/bin/python
P=checkpoints/td3_place/best; PT=qat_output_bi/place_orig_int8.tflite
O=Results/calibtest; mkdir -p $O/eval
SEEDS="7 31 47 89 101 123 211 307 401 503 555 2024"
# actor from arm A, calibration data from arm B
cross () {  # name actor_arm calib_arm seed
  A=checkpoints/td3_grasp_rand_td3_ln_buf${2}_s${4}
  B=checkpoints/td3_grasp_rand_td3_ln_buf${3}_s${4}
  $TFPY qat_and_convert.py --actor_path $A/best/actor_td3 \
    --replay_buffer_path $B/replay_buffer.npz --skip_qat --per_tensor \
    --input_dims 46 --fc1 64 --fc2 32 --n_actions 7 \
    --output_path $O/${1}_s${4}_int8.tflite > $O/conv_${1}_s${4}.log 2>&1
  echo "  ${1}_s${4}: actor=buf${2} calib=buf${3} -> $(stat -c%s $O/${1}_s${4}_int8.tflite 2>/dev/null) B"
  for s in $SEEDS; do
    echo "$TFPY fsm_sim.py --int8 --grasp-tflite $O/${1}_s${4}_int8.tflite --place-tflite $PT --grasp-ckpt $A/best --place-ckpt $P --regrasp --rc-steps 60 --episodes 100 --seed $s --tag ${1}_s${4} --out $O/eval/${1}_s${4}_e$s.csv > $O/eval/${1}_s${4}_e$s.log 2>&1" >> $O/eval/jobs.txt
  done
}
: > $O/eval/jobs.txt
for sd in 0 1 2; do
  cross a200_c1000 200k 1000k $sd     # weak actor, rich calibration
  cross a1000_c200 1000k 200k $sd     # strong actor, poor calibration
done
n=$(wc -l < $O/eval/jobs.txt); rm -f $O/eval/DONE
setsid nohup bash -c "xargs -a $O/eval/jobs.txt -d '\n' -P 24 -I{} bash -c '{}'; touch $O/eval/DONE" > $O/eval/queue.log 2>&1 < /dev/null &
disown -a; echo "queued $n calibration-crossover shards"
