#!/usr/bin/env bash
# End-to-end ladder for the alignment arm, run through handoff_diag.py rather
# than test_place.py so a single pass yields BOTH the end-to-end success rate
# and the corner-grip prevalence the term was designed to reduce.
set -u
cd "$(dirname "$0")"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
declare -A ARMS=(
  [align_s0]=../checkpoints/td3_grasp_rand_td3_ln_align_s0/best
  [align_s1]=../checkpoints/td3_grasp_rand_td3_ln_align_s1/best
  [align_s2]=../checkpoints/td3_grasp_rand_td3_ln_align_s2/best
  [gripfix_s2]=../checkpoints/td3_grasp_rand_td3_ln_gripfix_s2/best
)
for arm in "${!ARMS[@]}"; do for s in 7 123 2024 555; do
  python handoff_diag.py --episodes 100 --seed "$s" \
    --grasp-chkpt-dir "${ARMS[$arm]}" \
    --out "Results/align_ladder/${arm}_s${s}.csv" \
    > "Results/align_ladder/${arm}_s${s}.log" 2>&1 &
done; done
wait
echo "ALIGN LADDER COMPLETE"
