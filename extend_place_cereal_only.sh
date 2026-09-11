#!/usr/bin/env bash
set -u
cd "$(dirname "$0")/Decomposed state training"
PY=/home/fahim/Thesis_fahim/venv/bin/python
for SD in 0 1 2; do
  nohup $PY -u train_place.py --random-spawn --layer-norm --batch-size 1024 \
    --buffer-size 2000000 --critic-fc1 512 --critic-fc2 256 --n-envs 8 \
    --episodes 47000 --object-type cereal --seed $SD \
    --grasp-chkpt-dir ../checkpoints/td3_grasp_rand_td3_ln_cereal_alignwarm_s0/best \
    --place-chkpt-dir ../checkpoints/td3_place_cerealbig_s$SD \
    >> ../logs/train_place_cerealbig_s$SD.log 2>&1 &
  sleep 4
done
for SD in 0 1 2; do
  nohup $PY -u train_place.py --random-spawn --layer-norm --batch-size 1024 \
    --buffer-size 200000 --critic-fc1 512 --critic-fc2 256 --n-envs 8 \
    --episodes 8000 --object-type cereal --seed $SD \
    --warm-start-from ../checkpoints/td3_place_cereal_s2/best \
    --grasp-chkpt-dir ../checkpoints/td3_grasp_rand_td3_ln_cereal_alignwarm_s0/best \
    --place-chkpt-dir ../checkpoints/td3_place_cerealsmall_s$SD \
    > ../logs/train_place_cerealsmall_s$SD.log 2>&1 &
  sleep 4
done
echo "relaunched 3 cerealbig (resume) + 3 cerealsmall (fresh)"
