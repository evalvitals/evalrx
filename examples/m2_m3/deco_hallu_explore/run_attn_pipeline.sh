#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

# Held-out hypothesis pipeline over the attention-enriched data:
#   phase 0  prepare_splits.py     split==explore (365) / split==validate (241)
#   phase 1  evalvitals explore    M2+M3 on the explore half -> hypotheses + dashboard
#   phase 2  test_hypotheses.py    frozen-recipe re-eval + e-BH on the validate half
#                                  + LLM judge grades each hypothesis
#   phase 3  run_surgery.py        M5 confirm -> M4 -> tiered fix (L1..L3b) on GPU
#                                  (needs ../diagnosis_loops/deco_hallu/outputs/m1_state.pkl)
#   report   confirm_report.json + fix_report.json land next to the exploratory
#            report; the dashboard renders proposal + verdicts + fix in one page.
#
# Env overrides: CODER_PROVIDER / CODER_MODEL / JUDGE_MODEL / OUT_ROOT /
# TIMEOUT_SEC / SKIP_FIX=1 (stop after phase 2) / DEVICE (default cuda).

CODER_PROVIDER="${CODER_PROVIDER:-claude_code}"
CODER_MODEL="${CODER_MODEL:-}"
JUDGE_MODEL="${JUDGE_MODEL:-claude-opus-4-8}"
OUT_ROOT="${OUT_ROOT:-outputs_pipeline}"
DEVICE="${DEVICE:-cuda}"
PY="${PY:-python3}"

echo "=== phase 0: split preparation ==="
"$PY" prepare_splits.py

echo "=== phase 1: explore (M2+M3) on the explore half ==="
cmd=(
  evalvitals explore data_attn_explore
  --backend "$CODER_PROVIDER"
  --outcome-col label
  --out "$OUT_ROOT/1_explore"
  -q "What predicts hallucination failures (label=fail) in this VLM object-presence probe? The per-case attention-geometry scalars cover every case across three checkpoints (2B/4B/8B). This is the EXPLORATION HALF of a held-out design (a validate half exists and will test whatever you propose): compare the signals' distributions between FAIL and PASS within adversarial probes, check checkpoint and object structure, and make every candidate signal a FROZEN, threshold-explicit recipe so it can be re-evaluated verbatim on the held-out half. Propose hypotheses precise enough to be graded against held-out statistics."
  --max-attempts 3
  --timeout-sec "${TIMEOUT_SEC:-1200}"
)
if [[ -n "$CODER_MODEL" ]]; then cmd+=(--model "$CODER_MODEL"); fi
"${cmd[@]}"

echo "=== phase 2: held-out hypothesis testing (validate half) ==="
"$PY" test_hypotheses.py --pipeline-root "$OUT_ROOT" --judge-model "$JUDGE_MODEL"

if [[ "${SKIP_FIX:-0}" == "1" ]]; then
  echo "SKIP_FIX=1 — stopping after phase 2."
else
  echo "=== phase 3: surgery + tiered fix (GPU) ==="
  "$PY" run_surgery.py --pipeline-root "$OUT_ROOT" --device "$DEVICE"
fi

echo
echo "View the combined report (proposal + held-out verdicts + fix):"
echo "  evalvitals dashboard $OUT_ROOT/1_explore"
