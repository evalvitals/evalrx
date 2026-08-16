#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

CODER_PROVIDER="${CODER_PROVIDER:-claude_code}"
CODER_MODEL="${CODER_MODEL:-}"
OUT_DIR="${OUT_DIR:-outputs}"

python3 generate_data.py

cmd=(
  evalvitals
  explore
  data/batches.json
  --backend "$CODER_PROVIDER"
  --outcome-col yield_pct
  --out "$OUT_DIR"
  -q "What predicts yield_pct? Explore how temperature, pressure, and catalyst each relate to yield, how they relate to each other, and produce a variety of chart types."
  --timeout-sec "${TIMEOUT_SEC:-300}"
)

if [[ -n "$CODER_MODEL" ]]; then
  cmd+=(--model "$CODER_MODEL")
fi

exec "${cmd[@]}"
