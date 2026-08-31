#!/usr/bin/env bash
# Evaluate bigc512 (Net2Wider 128/64) against smallc64 (same parent, same 6000
# extra episodes, unchanged width). The pair isolates WIDTH from extra training.
# Both arms carry --actor-wclip 8 --actor-fakequant, inherited from qat8.
set -u
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/home/fahim/Thesis_fahim/venv/bin/python
TFPY=/home/fahim/Thesis_fahim/convert_venv/bin/python
P=checkpoints/td3_place/best
PT=qat_output_bi/place_orig_int8.tflite
O=Results/bigcritic; mkdir -p $O/eval
SEEDS="7 31 47 89 101 123 211 307 401 503 555 2024"
conv () {  # tag fc1 fc2 seed
  C=checkpoints/td3_grasp_rand_td3_ln_${1}_s${4}
  [ -f "$C/best/actor_td3" ] || { echo "MISSING $C"; return; }
  $TFPY qat_and_convert.py --actor_path $C/best/actor_td3 \
    --replay_buffer_path $C/replay_buffer.npz --skip_qat --per_tensor \
    --input_dims 46 --fc1 $2 --fc2 $3 --n_actions 7 \
    --output_path $O/${1}_s${4}_int8.tflite > $O/convert_${1}_s${4}.log 2>&1
  echo "  ${1}_s${4} -> $(stat -c%s $O/${1}_s${4}_int8.tflite 2>/dev/null) bytes"
}
for sd in 0 1 2 3 4 5; do conv bigc512 64 32 $sd; conv smallc64 64 32 $sd; done
: > $O/eval/jobs.txt
for sd in 0 1 2 3 4 5; do
  for arm in bigc512 smallc64; do
    W=""
    C=checkpoints/td3_grasp_rand_td3_ln_${arm}_s${sd}
    GT=$O/${arm}_s${sd}_int8.tflite
    for s in $SEEDS; do
      echo "$PY fsm_sim.py --grasp-ckpt $C/best $W --place-ckpt $P --regrasp --rc-steps 60 --episodes 100 --seed $s --tag ${arm}_s${sd}_fp32 --out $O/eval/${arm}_s${sd}_fp32_e$s.csv > $O/eval/${arm}_s${sd}_fp32_e$s.log 2>&1" >> $O/eval/jobs.txt
      echo "$TFPY fsm_sim.py --int8 --grasp-tflite $GT --place-tflite $PT --grasp-ckpt $C/best $W --place-ckpt $P --regrasp --rc-steps 60 --episodes 100 --seed $s --tag ${arm}_s${sd}_int8 --out $O/eval/${arm}_s${sd}_int8_e$s.csv > $O/eval/${arm}_s${sd}_int8_e$s.log 2>&1" >> $O/eval/jobs.txt
    done
  done
done
n=$(wc -l < $O/eval/jobs.txt); rm -f $O/eval/DONE
setsid nohup bash -c "xargs -a $O/eval/jobs.txt -d '\n' -P 16 -I{} bash -c '{}'; touch $O/eval/DONE" \
  > $O/eval/queue.log 2>&1 < /dev/null &
disown -a
echo "queued $n eval shards at 16 concurrent"
