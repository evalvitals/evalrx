#!/usr/bin/env bash
# M2+M3 (+ held-out confirm) over the M1 records produced by run_m1.py.
# Usage: bash run_explore.sh   [env: BACKEND=claude_code HOLDOUT=0.4 OUT=outputs/explore]
set -euo pipefail
cd "$(dirname "$0")"
REPO="$(cd ../../.. && pwd)"

BACKEND="${BACKEND:-claude_code}"
HOLDOUT="${HOLDOUT:-0.4}"
OUT="${OUT:-outputs/explore}"

TEMPLATE="$("$REPO/.venv/bin/python" -c 'from evalvitals.analysis.trajectory_records import AGENT_QUESTION_TEMPLATE as t; print(t)')"
QUESTION="The agent under test is Qwen3-VL-2B solving VTC-Bench 'counting' questions \
(four-way multiple choice, dense small-object counting) with image_zoom_in and \
image_detect tools under a ~1MP per-view resolution budget. ${TEMPLATE}"

exec "$REPO/.venv/bin/python" -m evalvitals.cli explore outputs/records.json \
  -q "$QUESTION" \
  --outcome-col label \
  --backend "$BACKEND" \
  --timeout-sec "${TIMEOUT_SEC:-3600}" \
  --holdout-frac "$HOLDOUT" --holdout-confirm --holdout-seed 0 \
  --out "$OUT"
