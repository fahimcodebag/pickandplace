#!/usr/bin/env bash
# Round 2: does ARM POSTURE (joint config, gripper width, eef position) at
# handoff explain gripfix_s0's stalls? Round 1 showed object-in-gripper pose
# does not: on flat grips only, s0 stalls 16.0% vs s2's 2.3% with the same
# transport policy and statistically identical handoff object pose.
set -u
cd "$(dirname "$0")"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
declare -A ARMS=(
  [gripfix_s0]=../checkpoints/td3_grasp_rand_td3_ln_gripfix_s0/best
  [gripfix_s2]=../checkpoints/td3_grasp_rand_td3_ln_gripfix_s2/best
)
for arm in "${!ARMS[@]}"; do for s in 7 123 2024 555; do
  python handoff_diag.py --episodes 100 --seed "$s" \
    --grasp-chkpt-dir "${ARMS[$arm]}" \
    --out "Results/handoff_diag2/${arm}_s${s}.csv" \
    > "Results/handoff_diag2/${arm}_s${s}.log" 2>&1 &
done; done
wait
echo "POSTURE DIAG COMPLETE"
