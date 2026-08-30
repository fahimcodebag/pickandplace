#!/usr/bin/env bash
# Convert the QAT arm to PER-TENSOR INT8 (the deployed granularity -- omitting
# --per_tensor silently yields per-channel, which measured nothing last time)
# and evaluate on the standard protocol against the existing wclip8 control.
# Control INT8/FP32 cells already exist in Results/wclip/eval/w8_s*_{int8,fp32}.
# THERMAL: eval capped at 16 concurrent (~50%), not 32.
set -u
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/home/fahim/Thesis_fahim/venv/bin/python
TFPY=/home/fahim/Thesis_fahim/convert_venv/bin/python
P=checkpoints/td3_place/best
PT=qat_output_bi/place_orig_int8.tflite
O=Results/qat8; mkdir -p $O/eval
SEEDS="7 31 47 89 101 123 211 307 401 503 555 2024"
for sd in 0 1 2 3 4 5; do
  C=checkpoints/td3_grasp_rand_td3_ln_qat8_s${sd}
  [ -f "$C/best/actor_td3" ] || { echo "MISSING $C/best/actor_td3"; continue; }
  $TFPY qat_and_convert.py --actor_path $C/best/actor_td3 \
    --replay_buffer_path $C/replay_buffer.npz --skip_qat --per_tensor \
    --input_dims 46 --fc1 64 --fc2 32 --n_actions 7 \
    --output_path $O/grasp_qat8_s${sd}_int8.tflite > $O/convert_s${sd}.log 2>&1
  echo "converted s$sd -> $(stat -c%s $O/grasp_qat8_s${sd}_int8.tflite 2>/dev/null) bytes (deployed artifact is 8088)"
done
: > $O/eval/jobs.txt
for sd in 0 1 2 3 4 5; do
  C=checkpoints/td3_grasp_rand_td3_ln_qat8_s${sd}
  GT=$O/grasp_qat8_s${sd}_int8.tflite
  for s in $SEEDS; do
    echo "$PY fsm_sim.py --grasp-ckpt $C/best --place-ckpt $P --regrasp --rc-steps 60 --episodes 100 --seed $s --tag qat8_s${sd}_fp32 --out $O/eval/q${sd}_fp32_e$s.csv > $O/eval/q${sd}_fp32_e$s.log 2>&1" >> $O/eval/jobs.txt
    echo "$TFPY fsm_sim.py --int8 --grasp-tflite $GT --place-tflite $PT --grasp-ckpt $C/best --place-ckpt $P --regrasp --rc-steps 60 --episodes 100 --seed $s --tag qat8_s${sd}_int8 --out $O/eval/q${sd}_int8_e$s.csv > $O/eval/q${sd}_int8_e$s.log 2>&1" >> $O/eval/jobs.txt
  done
done
# control seeds 3-5 are new and need the same treatment (s0-s2 are in Results/wclip)
for sd in 3 4 5; do
  C=checkpoints/td3_grasp_rand_td3_ln_wclip8_s${sd}
  [ -f "$C/best/actor_td3" ] || { echo "MISSING control $C"; continue; }
  $TFPY qat_and_convert.py --actor_path $C/best/actor_td3 \
    --replay_buffer_path $C/replay_buffer.npz --skip_qat --per_tensor \
    --input_dims 46 --fc1 64 --fc2 32 --n_actions 7 \
    --output_path Results/wclip/grasp_wclip8_s${sd}_int8.tflite \
    > Results/wclip/convert_wclip8_s${sd}.log 2>&1
  echo "control s$sd -> $(stat -c%s Results/wclip/grasp_wclip8_s${sd}_int8.tflite 2>/dev/null) bytes"
  GT=Results/wclip/grasp_wclip8_s${sd}_int8.tflite
  for s in $SEEDS; do
    echo "$PY fsm_sim.py --grasp-ckpt $C/best --place-ckpt $P --regrasp --rc-steps 60 --episodes 100 --seed $s --tag wclip8_s${sd}_fp32 --out Results/wclip/eval/w8_s${sd}_fp32_e$s.csv > Results/wclip/eval/w8_s${sd}_fp32_e$s.log 2>&1" >> $O/eval/jobs.txt
    echo "$TFPY fsm_sim.py --int8 --grasp-tflite $GT --place-tflite $PT --grasp-ckpt $C/best --place-ckpt $P --regrasp --rc-steps 60 --episodes 100 --seed $s --tag wclip8_s${sd}_int8 --out Results/wclip/eval/w8_s${sd}_int8_e$s.csv > Results/wclip/eval/w8_s${sd}_int8_e$s.log 2>&1" >> $O/eval/jobs.txt
  done
done
n=$(wc -l < $O/eval/jobs.txt); rm -f $O/eval/DONE
setsid nohup bash -c "xargs -a $O/eval/jobs.txt -d '\n' -P 16 -I{} bash -c '{}'; touch $O/eval/DONE" \
  > $O/eval/queue.log 2>&1 < /dev/null &
disown -a
echo "queued $n eval shards at 16 concurrent"
