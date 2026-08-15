#!/usr/bin/env bash
#
# deco_hallu — from a FROZEN M1 → the dashboard.
#
# Assumes M1 already ran and outputs/m1_state.pkl exists (produced by run_m1.py
# or run_all.sh). Skips the GPU-heavy M1 forward passes and runs only:
#   2) run_fused.py      Step 1: explore + held-out confirm (claude)
#   3) run_m2-5.py       Step 2: M2→M3→M5→Fix             (GPU + claude)
#   4) dashboard         the connected Analysis report
#
# Use this to iterate on the analysis/repair stages against a fixed M1 result.
#
# Overridable via env vars:
#   MODEL=qwen3-vl-2b-instruct  DEVICE=cuda  BACKEND=claude
#   PORT=8501  DASHBOARD=1  (DASHBOARD=0 just prints the dashboard command)
#   PY=/path/to/python  (defaults to the repo's .venv, then `python`)
#   MODEL should match the model M1 was frozen with (run_m2-5 warns on mismatch).
#
# Usage:  ./run_from_m1.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../../.." && pwd)"
cd "$HERE"   # the run_*.py scripts do `import run`, so CWD must be this dir

MODEL="${MODEL:-qwen3-vl-2b-instruct}"
DEVICE="${DEVICE:-cuda}"
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

# Require the frozen M1 — this script deliberately does NOT regenerate it.
if [[ ! -f "outputs/m1_state.pkl" ]]; then
  echo "error: outputs/m1_state.pkl not found." >&2
  echo "Run M1 first:  $PY run_m1.py --model $MODEL --device $DEVICE" >&2
  echo "       or the full pipeline:  ./run_all.sh" >&2
  exit 1
fi

step "deco_hallu from frozen M1  (model=$MODEL device=$DEVICE backend=$BACKEND)"
echo "python : $PY"
echo "M1     : $(cd "$HERE" && pwd)/outputs/m1_state.pkl"

step "1/3  run_fused.py  — explore + held-out confirm [claude]"
"$PY" run_fused.py --backend "$BACKEND" "${SKILL_ARGS[@]}"

step "2/3  run_m2-5.py  — M2→M3→M5→Fix [GPU + claude]"
"$PY" run_m2-5.py \
  --model "$MODEL" --device "$DEVICE" --backend "$BACKEND" \
  --recipes        outputs/fused/confirmed_recipes.json \
  --explore-report outputs/fused/fused_report.json

step "3/3  dashboard"
if [[ "$DASHBOARD" == "1" ]]; then
  echo "serving on http://localhost:$PORT  (Ctrl-C to stop)"
  exec "$PY" -m evalvitals.cli dashboard outputs --port "$PORT"
else
  echo "skipped (DASHBOARD=0). Launch it with:"
  echo "  $PY -m evalvitals.cli dashboard outputs --port $PORT"
fi
