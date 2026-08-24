#!/usr/bin/env bash
# Paired reset ablation for the PLACE/transport stage.
#
# The first random-spawn transport run (no resets, plain td3.Agent) showed the
# same decay as the grasp stage -- 79% at ep 1500, 8% by 2500, 17% by 4500 --
# and its best checkpoint evaluated at 75% end-to-end, BELOW the 82% of the
# original fixed-spawn transport policy it was seeded from. Resets were worth
# +20.3 (level 1.0) and +30.0 (level 2.0) on the grasp stage; this tests
# whether they carry to transport.
#
# Seed 0 of the reset arm is already running. This adds seeds 1-2 of the reset
# arm and all three no-reset controls, so the claim is a paired 3-seed result.
# Note the killed seed-0 no-reset run is NOT a valid control: it was
# interrupted by a power cut mid-collapse, so it is rerun here.
set -euo pipefail
cd "$(dirname "$0")/Decomposed state training"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2

GRASP=../checkpoints/td3_grasp_rand_td3_ln_liftcert_phaseB_s2/best
WARM=../checkpoints/td3_place/best
EPISODES=${EPISODES:-8000}
RESET=${RESET:-20000}
LOGDIR=../logs/place_matrix
mkdir -p "$LOGDIR"

launch () {  # launch <seed> <reset|control>
  local seed=$1 arm=$2 extra=()
  [ "$arm" = reset ] && extra=(--critic-reset-every "$RESET")
  echo "launching place seed $seed $arm"
  nohup python3 -u train_place.py --n-envs 5 --episodes "$EPISODES" \
      --random-spawn --grasp-chkpt-dir "$GRASP" --warm-start-from "$WARM" \
      --place-chkpt-dir "../checkpoints/td3_place_rs_${arm}_s${seed}" \
      "${extra[@]}" > "$LOGDIR/${arm}_s${seed}.log" 2>&1 &
  echo "  pid $!"
  sleep 3
}

launch 1 reset
launch 2 reset
launch 0 control
launch 1 control
launch 2 control

echo
echo "five launched; seed-0 reset arm already running as td3_place_randspawn_reset"
