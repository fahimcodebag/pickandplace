#!/bin/bash
# ROT_ANCHOR_EPS on the DEPLOYED artifact: c2m512_s1 grasp + checkpoints/td3_place/best,
# FP32 and INT8, random and fixed spawn, 12 seeds x 100, through fsm_sim.py itself.
#
#   ./run_anchor_c2m512.sh baseline  [P]   --rot-anchor-eps 0.  Must reproduce the
#        recorded CSVs to the episode: random on run_holdout4.sh's held-out seeds
#        (Results/holdout4, 94.92 / 95.33), fixed on run_fixed_c2m.sh's protocol seeds
#        (Results/fixed_c2m512, 100.0 / 99.83).  Also runs anchor_probe.py at eps 0:
#        on c2m512_s1 random (sizes the blocked class -> the prediction) and on bi_s0
#        FP32 protocol seeds (positive control: recorded 1088/1200, blocked ~3.08%).
#   ./run_anchor_c2m512.sh treatment [P]   --rot-anchor-eps 1e-6, same seeds.  Refuses
#        to run until Results/anchor_c2m512/PREDICTION.txt exists.
# Resumable: skips any job whose CSV already exists.
set -u
cd "$(dirname "$0")"
PHASE=${1:?baseline or treatment}; P=${2:-24}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/home/fahim/Thesis_fahim/venv/bin/python
TFPY=/home/fahim/Thesis_fahim/convert_venv/bin/python
C=checkpoints/td3_grasp_rand_td3_ln_c2m512_s1/best
GT=Results/big2m/c2m512_s1_int8.tflite
B=checkpoints/td3_grasp_rand_td3_ln_bi_s0/best
PL=checkpoints/td3_place/best
PT=qat_output_bi/place_orig_int8.tflite
HELD="13 59 71 137 199 257 331 419 547 601 733 911"
PROT="7 31 47 89 101 123 211 307 401 503 555 2024"
O=Results/anchor_c2m512; mkdir -p $O
COMMON="--place-ckpt $PL --regrasp --rc-steps 60 --episodes 100"
case $PHASE in
  baseline)  E=0 ;;
  treatment) E=1e-6
             [ -s $O/PREDICTION.txt ] || { echo "record $O/PREDICTION.txt first"; exit 1; } ;;
  *) echo "phase must be baseline or treatment"; exit 1 ;;
esac
J=$O/jobs_$PHASE.txt; : > $J
job() {  # stem, then the command
  local stem=$1; shift
  [ -s $O/$stem.csv ] && return
  echo "$* --out $O/$stem.csv > $O/$stem.log 2>&1" >> $J
}
INT8="--int8 --grasp-tflite $GT --place-tflite $PT"
for s in $HELD; do
  job rand_fp32_eps${E}_e$s  $PY   fsm_sim.py        --grasp-ckpt $C $COMMON --rot-anchor-eps $E --seed $s
  job rand_int8_eps${E}_e$s  $TFPY fsm_sim.py  $INT8 --grasp-ckpt $C $COMMON --rot-anchor-eps $E --seed $s
done
for s in $PROT; do
  job fixed_fp32_eps${E}_e$s $PY   fsm_sim.py --fixed-spawn       --grasp-ckpt $C $COMMON --rot-anchor-eps $E --seed $s
  job fixed_int8_eps${E}_e$s $TFPY fsm_sim.py --fixed-spawn $INT8 --grasp-ckpt $C $COMMON --rot-anchor-eps $E --seed $s
done
if [ $PHASE = baseline ]; then
  for s in $HELD; do
    job probe_rand_fp32_eps0_e$s $PY   anchor_probe.py --probe-out $O/probe_rand_fp32_eps0_e$s.probe.csv       --grasp-ckpt $C $COMMON --rot-anchor-eps 0 --seed $s
    job probe_rand_int8_eps0_e$s $TFPY anchor_probe.py --probe-out $O/probe_rand_int8_eps0_e$s.probe.csv $INT8 --grasp-ckpt $C $COMMON --rot-anchor-eps 0 --seed $s
  done
  for s in $PROT; do
    job probe_bi_s0_fp32_eps0_e$s $PY anchor_probe.py --probe-out $O/probe_bi_s0_fp32_eps0_e$s.probe.csv --grasp-ckpt $B $COMMON --rot-anchor-eps 0 --seed $s
  done
fi
echo "$PHASE: $(wc -l < $J) jobs at -P $P"
xargs -a $J -d '\n' -P $P -I{} bash -c '{}'
echo "$PHASE: done"
