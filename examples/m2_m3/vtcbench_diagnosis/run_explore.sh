#!/usr/bin/env bash
# M2+M3 (+ held-out confirm) over the M1 records produced by run_m1.py.
# Usage: bash run_explore.sh   [env: BACKEND=claude_code HOLDOUT=0.4 OUT=outputs/explore]
set -euo pipefail
cd "$(dirname "$0")"
BACKEND="${BACKEND:-claude_code}"
HOLDOUT="${HOLDOUT:-0.4}"
RECORDS="${RECORDS:-outputs/records.json}"
OUT="${OUT:-$(dirname "$RECORDS")/explore}"
MODEL_DESC="${MODEL_DESC:-Qwen3-VL-2B}"

# python3 off $PATH, not a hardcoded "$REPO/.venv/bin/python" -- that assumed
# every contributor's evalrx checkout uses a .venv named exactly that,
# which breaks both in Docker (system Python, no venv at all) and for anyone
# using a differently-named venv or a global install.
TEMPLATE="$(python3 -c 'from evalrx.analysis.trajectory_records import AGENT_QUESTION_TEMPLATE as t; print(t)')"
QUESTION="The agent under test is ${MODEL_DESC} solving VTC-Bench 'counting' questions \
(four-way multiple choice, dense small-object counting) with image_zoom_in and \
image_detect tools under a ~1MP per-view resolution budget. ${TEMPLATE}"

exec python3 -m evalrx.cli explore "$RECORDS" \
  -q "$QUESTION" \
  --outcome-col label \
  --backend "$BACKEND" \
  --timeout-sec "${TIMEOUT_SEC:-3600}" \
  --holdout-frac "$HOLDOUT" --holdout-confirm --holdout-seed 0 \
  --out "$OUT"
