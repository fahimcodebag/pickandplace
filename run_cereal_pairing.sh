#!/bin/bash
# CEREAL, RANDOM SPAWN: a TRAINED place policy vs scripted transport vs the bread
# actor transferred, paired on seed, 12 protocol seeds x 100, through fsm_sim.py.
#
# thesis_context §10 "a learned transport policy is not required" is cost-matched
# on cereal only: its learned arm is the bread place actor transferred
# (73.17% vs scripted 89.17%, Results/orientation_anchor.txt, commit 6c4a122).
# td3_place_cerealscratch_s2/best is the first trained cereal place policy that
# works (89.00% on 8 held-out seeds x 100, Results/place_selection).  Its best/ was
# chosen by the trainer's metric, never on these seeds.
#
# Arms, all with the cereal_alignwarm_s0 grasp the place runs trained against and
# that produced the recorded 73.17 / 89.17 (Results/transport_diagnosis.txt):
#   scripted_eps1e-6     --waypoint-transport --near-target-xy 0.08, anchor on
#                        CONTROL: must reproduce the recorded 1070/1200
#   transfer_eps1e-6     td3_place/best, anchor on
#                        CONTROL: must reproduce the recorded 878/1200
#   transfer_eps0        td3_place/best, anchor off (fsm_sim default since 2026-09-11)
#   trained_eps1e-6      cerealscratch_s2/best, anchor on (as it was trained)
#   trained_eps0         cerealscratch_s2/best, anchor off
#   trained_r008_eps1e-6 cerealscratch_s2/best with the scripted arm's 0.08 radius
# Every arm: --regrasp --rc-steps 60 (Fix A + C), robosuite criterion, and
# --settle-steps 60 so physical rest is recorded beside the scored success.
# Resumable: skips existing CSVs.
set -u
cd "$(dirname "$0")"
P=${1:-24}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/home/fahim/Thesis_fahim/venv/bin/python
G=checkpoints/td3_grasp_rand_td3_ln_cereal_alignwarm_s0/best
T=checkpoints/td3_place/best
S2=checkpoints/td3_place_cerealscratch_s2/best
SEEDS="7 31 47 89 101 123 211 307 401 503 555 2024"
O=Results/cereal_pairing; mkdir -p $O/csv
BASE="--grasp-ckpt $G --object-type cereal --regrasp --rc-steps 60 --success-criterion robosuite --settle-steps 60 --episodes 100"
J=$O/jobs.txt; : > $J
arm() {  # name, then extra flags
  local name=$1; shift
  for s in $SEEDS; do
    [ -s $O/csv/${name}_e$s.csv ] && continue
    echo "$PY fsm_sim.py $BASE $* --seed $s --out $O/csv/${name}_e$s.csv > $O/csv/${name}_e$s.log 2>&1" >> $J
  done
}
arm scripted_eps1e-6     --place-ckpt $T  --waypoint-transport --near-target-xy 0.08 --rot-anchor-eps 1e-6
arm transfer_eps1e-6     --place-ckpt $T  --rot-anchor-eps 1e-6
arm transfer_eps0        --place-ckpt $T  --rot-anchor-eps 0
arm trained_eps1e-6      --place-ckpt $S2 --rot-anchor-eps 1e-6
arm trained_eps0         --place-ckpt $S2 --rot-anchor-eps 0
arm trained_r008_eps1e-6 --place-ckpt $S2 --near-target-xy 0.08 --rot-anchor-eps 1e-6
echo "jobs: $(wc -l < $J) at -P $P"
xargs -a $J -d '\n' -P $P -I{} bash -c '{}'
echo "done"
