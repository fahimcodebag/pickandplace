#!/usr/bin/env bash
# WARNING -- DO NOT USE. kill -- -PGID stopped ALL SIX can runs, not the four
# intended, because run_can.sh launched them from one shell and they share a
# process group. Kept only as a record. Use stop_runs.sh, which filters on
# /proc comm and kills by exact PID.
# Stop selected can grasp seeds. Lives in a file so the caller's command line
# never contains the match pattern -- pkill -f matched its own shell twice in
# this project's history.
set -u
for sd in "$@"; do
  for pid in $(ps -eo pid,args | grep "train_rand.py" | grep -- "--seed $sd --tag can_warm" | grep -v grep | awk '{print $1}'); do
    kill -TERM -- -"$(ps -o pgid= "$pid" | tr -d ' ')" 2>/dev/null || kill -TERM "$pid" 2>/dev/null
  done
done
sleep 3
echo "can procs remaining: $(pgrep -fc 'train_rand.py' || echo 0)"
