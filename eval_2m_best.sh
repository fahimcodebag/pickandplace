#!/bin/bash
# Headline scoring of the finished 2M TD3 batch: 12 seeds x 100 episodes per
# checkpoint, paired on seed, END-TO-END through fsm_sim.py.
#
# WHAT IS SCORED.  The four surviving runs' best/ policies, plus the warm-start
# SOURCE (td3_place_cerealscratch_s2/best, 89.00% on 8 held-out seeds) as the
# reference every one of them has to beat -- they were all initialised from it,
# so "did training help?" is exactly this comparison, paired on seed.
#
# SAFE TO READ: the four TD3 trainers exited at 55,000 episodes. The three SAC
# runs are still training, so their best/ is NOT scored here (it is rewritten in
# place); their write-once snapshots are what es_watch scores meanwhile.
#
# Anchor stays --rot-anchor-eps 1e-6 (in the job script), matching every recorded
# cereal figure including the 89.00% reference. fsm_sim's own default is 0.0
# because the anchor hurts the deployed c2m512_s1 grasp (9.17.1); for cereal
# place actors it helps, and comparability is what matters here.
set -u
cd "$(dirname "$0")"
P=${1:-16}
SEEDS="7 31 47 89 101 123 211 307 401 503 555 2024"
EP=100
R=Results/place_2m_best; mkdir -p $R/csv; touch $R/eval.tsv
J=$R/eval_jobs.txt; : > $J

trainer_alive() {   # /proc comm + cmdline, never pgrep -f
  local n=$1 p
  for p in /proc/[0-9]*; do
    case "$(cat $p/comm 2>/dev/null)" in python*) ;; *) continue ;; esac
    tr '\0' ' ' < $p/cmdline 2>/dev/null | grep -q "train_place\.py.*checkpoints/$n " && return 0
  done
  return 1
}

queue() {  # label  ckpt_dir
  local label=$1 ck=$2 sd
  [ -e "$ck/actor_td3" ] || { echo "  MISSING $label ($ck)"; return; }
  for sd in $SEEDS; do
    if awk -F'\t' -v l="$label" -v s="$sd" '$1==l && $2==s && $4>0 {f=1} END{exit !f}' $R/eval.tsv; then
      continue                      # already scored; resumable
    fi
    echo -e "$label\t$ck\t$sd\t$EP" >> $J
  done
}

for r in cer2Mnoise_s20 cer2Mnoise_s21 cer2Mnoise_s22 cer2Mplain_s20; do
  if trainer_alive "td3_place_$r"; then echo "  SKIP $r -- trainer still alive"; continue; fi
  queue "$r" "checkpoints/td3_place_$r/best"
done
queue "warmsrc_cerealscratch_s2" "checkpoints/td3_place_cerealscratch_s2/best"

echo "queued $(wc -l < $J) jobs (12 seeds x 100 episodes each), parallel $P"
[ -s $J ] || { echo "nothing to do"; exit 0; }
awk -F'\t' '{print $1, $2, $3, $4}' $J \
  | xargs -P "$P" -L1 ./eval_2m_best_job.sh >> $R/eval.tsv
echo "done: $(wc -l < $R/eval.tsv) rows in $R/eval.tsv"
