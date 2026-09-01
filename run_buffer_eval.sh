#!/usr/bin/env bash
# End-to-end evaluation of the buffer sweep. Convert each run's best checkpoint
# to per-tensor INT8 and measure on the standard 12-seed protocol, against the
# validated incumbent smallc64_s5 (INT8 93.08% held-out).
set -u
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/home/fahim/Thesis_fahim/venv/bin/python
TFPY=/home/fahim/Thesis_fahim/convert_venv/bin/python
P=checkpoints/td3_place/best
PT=qat_output_bi/place_orig_int8.tflite
O=Results/buffer; mkdir -p $O/eval
SEEDS="7 31 47 89 101 123 211 307 401 503 555 2024"
for b in 200k 500k 1000k; do for sd in 0 1 2; do
  C=checkpoints/td3_grasp_rand_td3_ln_buf${b}_s${sd}
  [ -f "$C/best/actor_td3" ] || { echo "MISSING $C"; continue; }
  $TFPY qat_and_convert.py --actor_path $C/best/actor_td3 \
    --replay_buffer_path $C/replay_buffer.npz --skip_qat --per_tensor \
    --input_dims 46 --fc1 64 --fc2 32 --n_actions 7 \
    --output_path $O/buf${b}_s${sd}_int8.tflite > $O/convert_${b}_s${sd}.log 2>&1
  echo "  buf${b}_s${sd} -> $(stat -c%s $O/buf${b}_s${sd}_int8.tflite 2>/dev/null) bytes (deployed is 8088)"
done; done
: > $O/eval/jobs.txt
for b in 200k 500k 1000k; do for sd in 0 1 2; do
  C=checkpoints/td3_grasp_rand_td3_ln_buf${b}_s${sd}; GT=$O/buf${b}_s${sd}_int8.tflite
  for s in $SEEDS; do
    echo "$PY fsm_sim.py --grasp-ckpt $C/best --place-ckpt $P --regrasp --rc-steps 60 --episodes 100 --seed $s --tag buf${b}_s${sd}_fp32 --out $O/eval/${b}_s${sd}_fp32_e$s.csv > $O/eval/${b}_s${sd}_fp32_e$s.log 2>&1" >> $O/eval/jobs.txt
    echo "$TFPY fsm_sim.py --int8 --grasp-tflite $GT --place-tflite $PT --grasp-ckpt $C/best --place-ckpt $P --regrasp --rc-steps 60 --episodes 100 --seed $s --tag buf${b}_s${sd}_int8 --out $O/eval/${b}_s${sd}_int8_e$s.csv > $O/eval/${b}_s${sd}_int8_e$s.log 2>&1" >> $O/eval/jobs.txt
  done
done; done
n=$(wc -l < $O/eval/jobs.txt); rm -f $O/eval/DONE
setsid nohup bash -c "xargs -a $O/eval/jobs.txt -d '\n' -P 24 -I{} bash -c '{}'; touch $O/eval/DONE" > $O/eval/queue.log 2>&1 < /dev/null &
disown -a
echo "queued $n eval shards at -P 24"
