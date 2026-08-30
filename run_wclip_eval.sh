#!/usr/bin/env bash
# Convert each wclip arm to per-tensor INT8 (no-QAT path, the deployed one) and
# measure FP32 + INT8 end-to-end on the standard protocol: 12 eval seeds x 100,
# Fix A + Fix C active (the adopted rule layer).
# The point of the sweep is the FP32-vs-INT8 TRADE: lower k should shrink the
# quantisation gap but may cost FP32 quality. Both cells are needed per arm.
set -u
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/home/fahim/Thesis_fahim/venv/bin/python
TFPY=/home/fahim/Thesis_fahim/convert_venv/bin/python
P=checkpoints/td3_place/best
PT=qat_output_bi/place_orig_int8.tflite
O=Results/wclip
SEEDS="7 31 47 89 101 123 211 307 401 503 555 2024"
mkdir -p $O/eval

# --- 1. convert, sequentially: the converter is memory-hungry and short -----
for k in 0 4 6 8; do for sd in 0 1 2; do
  C=checkpoints/td3_grasp_rand_td3_ln_wclip${k}_s${sd}
  if [ ! -f "$C/best/actor_td3" ]; then echo "MISSING $C/best/actor_td3"; continue; fi
  # Calibrated on each arm's OWN replay buffer: the calibration distribution is
  # a known confound (Results/int8_deployment.txt), so it must track the policy.
  $TFPY qat_and_convert.py --actor_path $C/best/actor_td3 \
    --replay_buffer_path $C/replay_buffer.npz --skip_qat --per_tensor \
    --input_dims 46 --fc1 64 --fc2 32 --n_actions 7 \
    --output_path $O/grasp_wclip${k}_s${sd}_int8.tflite \
    > $O/convert_wclip${k}_s${sd}.log 2>&1
  echo "converted wclip$k s$sd -> $(stat -c%s $O/grasp_wclip${k}_s${sd}_int8.tflite 2>/dev/null) bytes"
done; done

# --- 2. queue the behavioural evaluation ------------------------------------
: > $O/eval/jobs.txt
for k in 0 4 6 8; do for sd in 0 1 2; do
  C=checkpoints/td3_grasp_rand_td3_ln_wclip${k}_s${sd}
  GT=$O/grasp_wclip${k}_s${sd}_int8.tflite
  for s in $SEEDS; do
    echo "$PY fsm_sim.py --grasp-ckpt $C/best --place-ckpt $P --regrasp --rc-steps 60 --episodes 100 --seed $s --tag wclip${k}_s${sd}_fp32 --out $O/eval/w${k}_s${sd}_fp32_e$s.csv > $O/eval/w${k}_s${sd}_fp32_e$s.log 2>&1" >> $O/eval/jobs.txt
    echo "$TFPY fsm_sim.py --int8 --grasp-tflite $GT --place-tflite $PT --grasp-ckpt $C/best --place-ckpt $P --regrasp --rc-steps 60 --episodes 100 --seed $s --tag wclip${k}_s${sd}_int8 --out $O/eval/w${k}_s${sd}_int8_e$s.csv > $O/eval/w${k}_s${sd}_int8_e$s.log 2>&1" >> $O/eval/jobs.txt
  done
done; done
n=$(wc -l < $O/eval/jobs.txt)
rm -f $O/eval/DONE
setsid nohup bash -c "xargs -a $O/eval/jobs.txt -d '\n' -P 32 -I{} bash -c '{}'; touch $O/eval/DONE" \
  > $O/eval/queue.log 2>&1 < /dev/null &
disown -a
echo "queued $n eval shards at 32 concurrent"
