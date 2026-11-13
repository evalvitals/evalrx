#!/usr/bin/env bash
#
# deco_hallu — DECOUPLED flow, PHASE 2: confirm the proposed hypotheses, then fix.
#
# The deferred half of ./run_analysis.sh. Assumes PHASE 1 already ran
# (outputs/analysis/analysis_state.pkl exists). Runs, with GPU + claude:
#   1) run_confirm_fix.py   M4 confirm (reuses PHASE 1's hypotheses + stats)
#                           → M5 surgery → tiered Fix (validated vs the baseline)
#   2) report            the FULL story — analysis + the M4/M5/Fix verdicts merged
#
# It reuses outputs/analysis/ so the hypotheses confirmed here are exactly the ones
# the analysis report showed (not a fresh re-proposal). The VLM is loaded here
# because the fix module validates candidate repairs against the live model.
#
# Overridable via env vars:
#   MODEL=qwen3-vl-2b-instruct  DEVICE=cuda  BACKEND=claude
#   PORT=8501  REPORT=1  (REPORT=0 just prints the report command)
#   PY=/path/to/python  (defaults to the repo's .venv, then `python`)
#   MODEL should match the model M1 was frozen with.
#
# Usage:  ./run_confirm_fix.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../../.." && pwd)"
cd "$HERE"   # the run_*.py scripts do `import run`, so CWD must be this dir

MODEL="${MODEL:-qwen3-vl-2b-instruct}"
DEVICE="${DEVICE:-cuda}"
BACKEND="${BACKEND:-claude}"
PORT="${PORT:-8501}"
REPORT="${REPORT:-1}"

if [[ -z "${PY:-}" ]]; then
  if [[ -x "$REPO_ROOT/.venv/bin/python" ]]; then PY="$REPO_ROOT/.venv/bin/python"; else PY="python"; fi
fi

step() { printf '\n\033[1;36m== %s ==\033[0m\n' "$*"; }

if [[ ! -f "outputs/analysis/analysis_state.pkl" ]]; then
  echo "error: outputs/analysis/analysis_state.pkl not found." >&2
  echo "Run the analysis phase first:  ./run_analysis.sh" >&2
  exit 1
fi

step "deco_hallu — confirm + fix phase  (model=$MODEL device=$DEVICE backend=$BACKEND)"
echo "python : $PY"
echo "reuse  : $HERE/outputs/analysis/  (proposed hypotheses + frozen M2 stats)"

step "1/2  run_confirm_fix.py  — M4 confirm → M5 + tiered Fix [GPU + claude]"
"$PY" run_confirm_fix.py --model "$MODEL" --device "$DEVICE" --backend "$BACKEND"

step "2/2  report  (full story — analysis + verdicts merged)"
if [[ "$REPORT" == "1" ]]; then
  echo "serving on http://localhost:$PORT  (Ctrl-C to stop)"
  exec "$PY" -m evalrx.cli serve outputs/logs_confirm_fix --port "$PORT"
else
  echo "skipped (REPORT=0). Launch it with:"
  echo "  $PY -m evalrx.cli serve outputs/logs_confirm_fix --port $PORT"
fi
