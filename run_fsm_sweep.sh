#!/usr/bin/env bash
# Stage 1: coordinate sweep of the FSM rule layer around the current baseline.
# Sec 5 precedent: rule-layer parameters alone were worth 78% -> 92% at FIXED
# spawn with identical weights. They have never been re-tuned for random spawn,
# where the dominant failure is a transport horizon-out (7.5%), not a policy
# error -- the grasp stage now reaches transport in 400/400 episodes.
set -u
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
G=checkpoints/td3_grasp_rand_td3_ln_bi_s0/best
P=checkpoints/td3_place/best
run () { # name, extra args...
  local n=$1; shift
  setsid nohup python fsm_sim.py --grasp-ckpt $G --place-ckpt $P \
    --episodes 100 --seed 7 --tag "$n" "$@" \
    --out Results/fsm_sweep/${n}.csv > Results/fsm_sweep/${n}.log 2>&1 < /dev/null &
}
run baseline
for v in 0.10 0.12 0.16 0.18 0.20; do run "nearxy_$v" --near-target-xy $v; done
for v in 0.4 0.6 0.7 0.8 1.0;   do run "tscale_$v" --translate-scale $v; done
for v in 200 400 500;            do run "horizon_$v" --place-horizon $v; done
for v in 1 2 5;                  do run "trighold_$v" --release-trig-hold $v; done
for v in 0.02 0.04;              do run "rctol_$v" --rc-tol $v; done
for v in 20 45;                  do run "rcsteps_$v" --rc-steps $v; done
for v in 20 45;                  do run "dssteps_$v" --ds-steps $v; done
for v in 3.0 6.0;                do run "carrygain_$v" --carry-gain $v; done
wait
echo "FSM SWEEP STAGE 1 COMPLETE"
