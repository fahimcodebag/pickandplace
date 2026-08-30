#!/usr/bin/env bash
# Grip robustness under scripted disturbance, transport policy removed.
# 3 arms x 3 profiles x 12 seeds x 100 = 10800 episodes.
# Job-queued at 32 concurrent (32 cores) rather than launched all at once:
# 108 simultaneous shards would oversubscribe 3.4x and thrash.
set -u
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/home/fahim/Thesis_fahim/venv/bin/python
TFPY=/home/fahim/Thesis_fahim/convert_venv/bin/python
MONO=/home/fahim/Thesis_fahim/checkpoints/td3_v7
G=checkpoints/td3_grasp_rand_td3_ln_bi_s0/best
GT=qat_output_fix/grasp_bi_noqat_int8.tflite
O=Results/grip_jerk
SEEDS="7 31 47 89 101 123 211 307 401 503 555 2024"

mkdir -p $O
: > $O/jobs.txt
for pr in hold carry jerk; do
  mkdir -p $O/$pr
  for s in $SEEDS; do
    echo "$PY grip_jerk_test.py --mono-ckpt $MONO --profile $pr --episodes 100 --seed $s --tag mono_$pr --out $O/$pr/mono_e$s.csv > $O/$pr/mono_e$s.log 2>&1" >> $O/jobs.txt
    echo "$PY grip_jerk_test.py --grasp-ckpt $G --profile $pr --episodes 100 --seed $s --tag dfp32_$pr --out $O/$pr/dfp32_e$s.csv > $O/$pr/dfp32_e$s.log 2>&1" >> $O/jobs.txt
    echo "$TFPY grip_jerk_test.py --grasp-tflite $GT --profile $pr --episodes 100 --seed $s --tag int8_$pr --out $O/$pr/int8_e$s.csv > $O/$pr/int8_e$s.log 2>&1" >> $O/jobs.txt
  done
done
n=$(wc -l < $O/jobs.txt)
setsid nohup bash -c "xargs -a $O/jobs.txt -d '\n' -P 32 -I{} bash -c '{}'; touch $O/DONE" \
  > $O/queue.log 2>&1 < /dev/null &
disown -a
echo "queued $n shards at 32 concurrent -> $O (touch $O/DONE when finished)"
