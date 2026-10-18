#!/usr/bin/env bash
# INT8 actors on the DEPLOYED perception route, paired on seed with the FP32
# row of Table 5 (Results/tag_e2e7/final_*, 538/600 = 89.67%).
#
# Every earlier perception row ran the FP32 grasp actor; the only INT8 x
# perception number (89.50%, apriltag_perception.txt) used the agentview /
# OpenCV / period-5 route. This measures what the board runs: both INT8 tflite
# actors, the firmware's own C detector (libat32.so) on the 180x160 wrist crop
# with bits_corrected=0, the wrist32 corrector, one latched detection.
#
# Steps:
#   1. rebuild libat32.so with bits_corrected=0 (firmware setting); the old
#      library is kept as libat32_prev.so
#   2. GATE: rerun the FP32 command on seed 7 and require a byte-identical CSV
#      to Results/tag_e2e7/final_s7.csv. If it differs, the flags (or the
#      library) are not the ones the 89.67% row used and the pairing would be
#      invalid -- fix FLAGS below and rerun. Nothing else is launched.
#   3. 12 seeds x 50 episodes INT8, in parallel, into Results/tag_e2e_int8/
#   4. paired comparison against the FP32 row
#
# Usage:  ./run_int8_perception.sh [parallel_jobs]     (default 12)
#         SKIP_GATE=1 ./run_int8_perception.sh          (not for the paper)
set -u
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MUJOCO_GL=${MUJOCO_GL:-egl}
J=${1:-12}
PY=/home/fahim/Thesis_fahim/venv/bin/python             # FP32 runs (tag_e2e7)
TFPY=/home/fahim/Thesis_fahim/convert_venv/bin/python   # has tensorflow, as holdout4
G=checkpoints/td3_grasp_rand_td3_ln_c2m512_s1/best
P=checkpoints/td3_place/best
GT=Results/big2m/c2m512_s1_int8.tflite                  # hil_hardware_validation.txt:8
PT=qat_output_bi/place_orig_int8.tflite                 # hil_hardware_validation.txt:9
REF=Results/tag_e2e7
O=Results/tag_e2e_int8; mkdir -p $O
SEEDS="7 31 47 89 101 123 211 307 401 503 555 2024"

# Perception flags of the FP32 row. Reconstructed -- the original command line
# was not logged -- which is exactly what the gate in step 2 checks.
FLAGS="--mode latch --camera robot0_eye_in_hand --width 320 --height 240 \
--backend esp32 --residual-model esp32_apriltag/tag_residual_wrist32.json \
--detect-until-first --episodes 50"

for f in $G/actor_td3 $P/actor_td3 $GT $PT $REF/final_s7.csv \
         esp32_apriltag/tag_residual_wrist32.json; do
  [ -e "$f" ] || { echo "missing: $f"; exit 1; }
done

# 1. detector library, bits_corrected=0 ---------------------------------------
if [ ! -e esp32_apriltag/libat32_prev.so ]; then
  cp esp32_apriltag/libat32.so esp32_apriltag/libat32_prev.so
fi
gcc -O2 -fPIC -shared -I esp32_apriltag/src -I esp32_apriltag/src/common \
    -DHOST_BUILD -DAT_CLUSTERMAP_FRAC=0.1 -DAT_MEM_CHUNK=1024 \
    -DAT_BITS_CORRECTED=0 \
    -o esp32_apriltag/libat32.so esp32_apriltag/host/at32_shim.c \
    esp32_apriltag/src/*.c esp32_apriltag/src/common/*.c -lm -lpthread \
  || { echo "build failed"; exit 1; }
echo "built esp32_apriltag/libat32.so (bits_corrected=0)"

# 2. gate: reproduce one FP32 seed exactly ------------------------------------
if [ "${SKIP_GATE:-0}" != 1 ]; then
  echo "gate: FP32 seed 7 vs $REF/final_s7.csv ..."
  $PY tag_e2e.py --grasp-ckpt $G --place-ckpt $P $FLAGS --seed 7 \
      --out $O/gate_fp32_s7.csv > $O/gate_fp32_s7.log 2>&1
  if cmp -s $O/gate_fp32_s7.csv $REF/final_s7.csv; then
    echo "gate PASSED: FP32 seed 7 reproduces the 89.67% row exactly"
  else
    echo "gate FAILED: $O/gate_fp32_s7.csv differs from $REF/final_s7.csv"
    diff <(cut -d, -f1-7 $REF/final_s7.csv) <(cut -d, -f1-7 $O/gate_fp32_s7.csv) | head -20
    echo "Check FLAGS (e.g. --period, --detect-until-first, --latch-median) and"
    echo "the bits setting (try -DAT_BITS_CORRECTED=1). No INT8 jobs launched."
    exit 1
  fi
fi

# 3. INT8, 12 seeds x 50 -------------------------------------------------------
: > $O/jobs.txt
for s in $SEEDS; do
  echo "$TFPY tag_e2e.py --int8 --grasp-tflite $GT --place-tflite $PT --grasp-ckpt $G --place-ckpt $P $FLAGS --seed $s --out $O/int8_s$s.csv > $O/int8_s$s.log 2>&1" >> $O/jobs.txt
done
echo "running $(wc -l < $O/jobs.txt) INT8 shards, $J at a time ..."
xargs -a $O/jobs.txt -d '\n' -P "$J" -I{} bash -c '{}'

# 4. paired comparison ---------------------------------------------------------
$PY - "$REF" "$O" $SEEDS <<'EOF' | tee $O/summary.txt
import csv, sys, math
from scipy import stats
ref, out, seeds = sys.argv[1], sys.argv[2], sys.argv[3:]
def rate(path):
    rows = list(csv.DictReader(open(path)))
    return sum(int(r["success"]) for r in rows), len(rows), \
           sum(int(r["truth_fallback_steps"]) > 0 for r in rows)
fp, i8, diffs = [], [], []
print("INT8 actors on the deployed perception route vs FP32 (Table 5)")
print(f"{'seed':>6} {'FP32':>8} {'INT8':>8} {'INT8 fallback eps':>18}")
for s in seeds:
    a, na, _ = rate(f"{ref}/final_s{s}.csv")
    b, nb, fb = rate(f"{out}/int8_s{s}.csv")
    assert na == nb == 50, (s, na, nb)
    fp.append(a); i8.append(b); diffs.append(100 * (b - a) / 50)
    print(f"{s:>6} {100*a/50:7.1f}% {100*b/50:7.1f}% {fb:>18}")
n = 50 * len(seeds)
print(f"{'total':>6} {100*sum(fp)/n:7.2f}% {100*sum(i8)/n:7.2f}%   "
      f"({sum(fp)}/{n} vs {sum(i8)}/{n})")
m = sum(diffs) / len(diffs)
sd = math.sqrt(sum((d - m) ** 2 for d in diffs) / (len(diffs) - 1))
se = sd / math.sqrt(len(diffs))
t = m / se if se else float("nan")
p = 2 * stats.t.sf(abs(t), len(diffs) - 1) if se else float("nan")
h = stats.t.ppf(0.975, len(diffs) - 1) * se
print(f"INT8 - FP32: {m:+.2f} pts, SD {sd:.2f}, 95% CI [{m-h:+.2f}, {m+h:+.2f}], "
      f"t({len(diffs)-1}) = {t:.2f}, p = {p:.3f}")
lo, hi = stats.beta.ppf(0.025, sum(i8), n - sum(i8) + 1), \
         stats.beta.ppf(0.975, sum(i8) + 1, n - sum(i8))
print(f"INT8 pooled Clopper-Pearson 95%: [{100*lo:.2f}, {100*hi:.2f}]")
EOF
touch $O/DONE
