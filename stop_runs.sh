#!/usr/bin/env bash
# Stop training runs matching $1, filtering to actual python processes so this
# cannot kill the shell that invoked it (pgrep -f matches the caller's own
# command line, which cost three self-kills in this project).
set -u
pat=$1; n=0
for p in $(pgrep -f "$pat" 2>/dev/null); do
  case "$(cat /proc/$p/comm 2>/dev/null)" in
    python*) kill -TERM "$p" 2>/dev/null; n=$((n+1)) ;;
  esac
done
sleep 4
echo "signalled $n python procs matching '$pat'"
