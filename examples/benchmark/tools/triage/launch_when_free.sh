#!/bin/bash
# Poll every 20 s for a GPU with under 3 GB in use and start the given protocol script on it at once (a 60 s
# poll lost a free GPU to another user within the minute). A GPU is never shared.
# usage: launch_when_free.sh <run script> [its arguments after the gpu index...]
#   e.g. nohup bash launch_when_free.sh run_triage18_phase3.sh > launcher.log 2>&1 &
set -u
SCRIPT=${1:?run script}; shift
HERE=$(cd "$(dirname "$0")" && pwd)
[ -f "$SCRIPT" ] || SCRIPT=$HERE/$SCRIPT
for _ in $(seq 1 4320); do   # up to 24 h
  free=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' '$2 < 3000 {print $1}' | head -1)
  if [ -n "$free" ]; then
    echo "LAUNCH $(basename "$SCRIPT") on GPU $free at $(date '+%m-%d %H:%M')"
    bash "$SCRIPT" "$free" "$@"
    echo "RUN_DONE rc=$? at $(date '+%m-%d %H:%M')"
    exit 0
  fi
  sleep 20
done
echo "NO_GPU after 24h"; exit 1
