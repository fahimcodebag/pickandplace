#!/usr/bin/env bash
# Handoff-pose diagnostic across three arms with contrasting failure profiles:
#   gripfix_s2  -- fewest stalls (13/400)
#   gripfix_s0  -- most stalls   (73/400)
#   baseline    -- most drops    (73/400)
# 4 seeds x 100 episodes = 400 samples per arm. Sharded 12-wide (24 threads of
# 32); the ladder used only 8 of 32 cores.
set -u
cd "$(dirname "$0")"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
OUT=Results/handoff_diag
declare -A ARMS=(
  [gripfix_s2]=../checkpoints/td3_grasp_rand_td3_ln_gripfix_s2/best
  [gripfix_s0]=../checkpoints/td3_grasp_rand_td3_ln_gripfix_s0/best
  [baseline]=../checkpoints/td3_grasp_rand_td3_ln_liftcert_phaseB_s2/best
)
for arm in "${!ARMS[@]}"; do
  for s in 7 123 2024 555; do
    python handoff_diag.py --episodes 100 --seed "$s" \
      --grasp-chkpt-dir "${ARMS[$arm]}" \
      --out "$OUT/${arm}_s${s}.csv" > "$OUT/${arm}_s${s}.log" 2>&1 &
  done
done
wait
echo "HANDOFF DIAG COMPLETE"
