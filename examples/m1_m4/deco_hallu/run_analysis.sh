#!/usr/bin/env bash
#
# deco_hallu — DECOUPLED flow, PHASE 1: analyse + propose → the dashboard.
#
# Assumes M1 already ran (outputs/m1_state.pkl exists). Runs, with NO GPU:
#   1) run_fused.py       Step 1: explore + held-out confirm (claude)
#   2) run_analysis.py    PHASE 1: M1(replay)→M2 stats+charts→M3 propose, NO M5/Fix
#   3) dashboard          the analysis story — hypotheses PROPOSED, not yet confirmed
#
# This is the "look before you repair" half: it stops at the dashboard so you can
# read what was found and which hypotheses were proposed BEFORE spending GPU/claude
# on confirmation + repair. When you're ready, run ./run_confirm_fix.sh — it reuses
# this phase's artifacts (outputs/analysis/), so the SAME hypotheses get confirmed.
#
# Overridable via env vars:
#   MODEL=qwen3-vl-2b-instruct  BACKEND=claude
#   PORT=8501  DASHBOARD=1  (DASHBOARD=0 just prints the dashboard command)
#   PY=/path/to/python  (defaults to the repo's .venv, then `python`)
#
# Usage:  ./run_analysis.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../../.." && pwd)"
cd "$HERE"   # the run_*.py scripts do `import run`, so CWD must be this dir

MODEL="${MODEL:-qwen3-vl-2b-instruct}"
BACKEND="${BACKEND:-claude}"
PORT="${PORT:-8501}"
DASHBOARD="${DASHBOARD:-1}"

if [[ -z "${PY:-}" ]]; then
  if [[ -x "$REPO_ROOT/.venv/bin/python" ]]; then PY="$REPO_ROOT/.venv/bin/python"; else PY="python"; fi
fi

NF="$REPO_ROOT/evalvitals/agent_assets/skills/nature-figure"
SKILL_ARGS=()
[[ -d "$NF" ]] && SKILL_ARGS=(--skill "$NF")

step() { printf '\n\033[1;36m== %s ==\033[0m\n' "$*"; }

if [[ ! -f "outputs/m1_state.pkl" ]]; then
  echo "error: outputs/m1_state.pkl not found." >&2
  echo "Run M1 first:  $PY run_m1.py --model $MODEL --device cuda" >&2
  echo "       or the full pipeline:  ./run_all.sh" >&2
  exit 1
fi

step "deco_hallu — analysis phase (no GPU)  (model=$MODEL backend=$BACKEND)"
echo "python : $PY"
echo "M1     : $HERE/outputs/m1_state.pkl"

step "1/3  run_fused.py  — explore + held-out confirm [claude]"
"$PY" run_fused.py --backend "$BACKEND" "${SKILL_ARGS[@]}"

step "2/3  run_analysis.py  — M2 stats+charts → M3 propose (NO M5/Fix) [claude]"
"$PY" run_analysis.py \
  --model "$MODEL" --backend "$BACKEND" \
  --recipes        outputs/fused/confirmed_recipes.json \
  --explore-report outputs/fused/fused_report.json

step "3/3  dashboard  (analysis story — proposed hypotheses, no verdicts yet)"
echo "next, when ready to confirm + repair:  ./run_confirm_fix.sh"
if [[ "$DASHBOARD" == "1" ]]; then
  echo "serving on http://localhost:$PORT  (Ctrl-C to stop)"
  exec "$PY" -m evalvitals.cli dashboard outputs --port "$PORT"
else
  echo "skipped (DASHBOARD=0). Launch it with:"
  echo "  $PY -m evalvitals.cli dashboard outputs --port $PORT"
fi
