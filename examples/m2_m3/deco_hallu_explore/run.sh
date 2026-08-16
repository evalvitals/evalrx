#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

CODER_PROVIDER="${CODER_PROVIDER:-claude_code}"
CODER_MODEL="${CODER_MODEL:-}"
OUT_DIR="${OUT_DIR:-outputs}"

# The raw M1 output directory: one JSON file per model, each a run-metadata
# dict with a "cases" list. It is handed to the M2 agent as-is (no
# pre-processing script) — the agent's own generated code loads and organizes
# whatever shape it finds. `source_cases/` is where the Dockerfile copies it
# in; a bare repo checkout falls back to the sibling example's data directory.
if [[ -z "${DATA_DIR:-}" ]]; then
  if [[ -d source_cases ]]; then
    DATA_DIR="source_cases"
  else
    DATA_DIR="../../diagnosis_loops/deco_hallu/data/cases"
  fi
fi

cmd=(
  evalvitals
  explore
  "$DATA_DIR"
  --backend "$CODER_PROVIDER"
  --outcome-col label
  --out "$OUT_DIR"
  -q "What predicts hallucination failures (label=fail) in this VLM object-presence probe? Compare across model size, probe_type (adversarial vs present), and object. Produce a variety of chart types."
  --max-attempts 3
  --timeout-sec "${TIMEOUT_SEC:-300}"
)

if [[ -n "$CODER_MODEL" ]]; then
  cmd+=(--model "$CODER_MODEL")
fi

exec "${cmd[@]}"
