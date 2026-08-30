#!/usr/bin/env bash
# Resume the wclip evaluation, skipping shards whose CSV is already complete.
# A power outage killed the first launch at 85/288; shards write their CSV only
# at the end, so "exists AND has 101 lines" is a sound completion test and a
# partial file is deleted rather than trusted.
set -u
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
O=Results/wclip
: > $O/eval/todo.txt
while IFS= read -r job; do
  out=$(sed -n 's/.*--out \([^ ]*\.csv\).*/\1/p' <<<"$job")
  if [ -n "$out" ] && [ -f "$out" ] && [ "$(wc -l < "$out")" -eq 101 ]; then continue; fi
  rm -f "$out"
  echo "$job" >> $O/eval/todo.txt
done < $O/eval/jobs.txt
n=$(wc -l < $O/eval/todo.txt)
rm -f $O/eval/DONE
setsid nohup bash -c "xargs -a $O/eval/todo.txt -d '\n' -P 32 -I{} bash -c '{}'; touch $O/eval/DONE" \
  > $O/eval/queue.log 2>&1 < /dev/null &
disown -a
echo "resumed: $n shards remaining of $(wc -l < $O/eval/jobs.txt)"
