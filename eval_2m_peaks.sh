#!/bin/bash
# Headline scoring of the PEAK SNAPSHOTS of the finished 2M TD3 batch.
#
# WHY NOT best/.  Scoring best/ (Results/place_2m_best) showed it is the wrong
# artifact: best/ matches NO snapshot in any run, because it is written whenever
# metric = frac50 x place50 improves, and that metric is measured UNDER THE
# CURRICULUM. A policy can score well at handoff fraction 0.5 and fail at the
# true full-distance spawn fsm_sim evaluates -- cer2Mplain_s20's best/ scored
# 1.42% while its ep_54000 snapshot scored 82% on the watcher's 2x50.
#
# The peaks below come from that 2x50 screen and need the full protocol before
# any of them can be quoted: 12 seeds x 100 episodes, paired, end-to-end.
# The reference (td3_place_cerealscratch_s2/best, 88.00%) is already scored at
# this protocol on these seeds in Results/place_2m_best/eval.tsv -- same job
# script, same anchor, same seeds -- so it is not re-run.
#
# Snapshots are write-once (ep_XXXXX.tmp -> rename), so these are safe to read
# even for a run whose trainer is alive. All four TD3 trainers have exited anyway.
set -u
cd "$(dirname "$0")"
P=${1:-16}
SEEDS="7 31 47 89 101 123 211 307 401 503 555 2024"
EP=100
R=Results/place_2m_best; mkdir -p $R/csv; touch $R/eval.tsv
J=$R/peak_jobs.txt; : > $J

# run:peak_episode, from the 2x50 screen in Results/place_long2m/eval.tsv
PEAKS="cer2Mnoise_s20:26000 cer2Mnoise_s21:38000 cer2Mnoise_s22:4000 cer2Mplain_s20:54000"

for pk in $PEAKS; do
  run=${pk%%:*}; ep=${pk##*:}
  ck=$(printf "checkpoints/td3_place_%s/snapshots/ep_%05d" "$run" "$ep")
  label=$(printf "%s_ep%05d" "$run" "$ep")
  [ -e "$ck/actor_td3" ] || { echo "  MISSING $label ($ck)"; continue; }
  for sd in $SEEDS; do
    awk -F'\t' -v l="$label" -v s="$sd" '$1==l && $2==s && $4>0 {f=1} END{exit !f}' $R/eval.tsv && continue
    echo -e "$label\t$ck\t$sd\t$EP" >> $J
  done
done
echo "queued $(wc -l < $J) jobs, parallel $P"
[ -s $J ] || { echo "nothing to do"; exit 0; }
awk -F'\t' '{print $1, $2, $3, $4}' $J | xargs -P "$P" -L1 ./eval_2m_best_job.sh >> $R/eval.tsv
echo "done"
