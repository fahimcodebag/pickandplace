#!/usr/bin/env bash
# Kill can grasp runs. Matches the pattern but then filters to processes whose
# executable is actually python -- pkill/pgrep -f otherwise matches the shell
# that INVOKED this script, because that shell's command line contains the
# pattern too. That footgun cost three accidental self-kills in this project.
n=0
for p in $(pgrep -f "train_rand\.py" 2>/dev/null); do
  case "$(cat /proc/$p/comm 2>/dev/null)" in
    python*) kill -TERM "$p" 2>/dev/null; n=$((n+1)) ;;
  esac
done
sleep 3
echo "killed $n python procs; remaining: $(pgrep -f 'train_rand\.py' 2>/dev/null | while read p; do case "$(cat /proc/$p/comm 2>/dev/null)" in python*) echo x;; esac; done | wc -l)"
