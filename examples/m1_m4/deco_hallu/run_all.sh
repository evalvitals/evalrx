#!/usr/bin/env bash
#
# deco_hallu — ONE-SHOT: from nothing → the dashboard.
#
# Runs the whole pipeline end to end:
#   0) build_cases.py    balanced FAIL/PASS batch        (offline, no GPU)
#   1) run_m1.py         M1 analyzers, frozen to a pickle (GPU)
#   2) run_fused.py      Step 1: explore + held-out confirm (claude)
#   3) run_m2-5.py       Step 2: M2→M3→M5→Fix             (GPU + claude)
#   4) dashboard         the connected Analysis report
#
# The expensive M1 forward passes run here. If you already have
# outputs/m1_state.pkl, use run_from_m1.sh instead (skips steps 0-1).
#
# Overridable via env vars:
#   MODEL=qwen3-vl-2b-instruct  DEVICE=cuda  BACKEND=claude
#   PORT=8501  DASHBOARD=1  (DASHBOARD=0 just prints the dashboard command)
#   PY=/path/to/python  (defaults to the repo's .venv, then `python`)
#
# Usage:  ./run_all.sh            # or:  MODEL=qwen3-vl-4b-instruct ./run_all.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../../.." && pwd)"
cd "$HERE"   # the run_*.py scripts do `import run`, so CWD must be this dir

MODEL="${MODEL:-qwen3-vl-2b-instruct}"
DEVICE="${DEVICE:-cuda}"
BACKEND="${BACKEND:-claude}"
PORT="${PORT:-8501}"
DASHBOARD="${DASHBOARD:-1}"

# Python: explicit $PY, else the repo venv, else whatever `python` resolves to.
if [[ -z "${PY:-}" ]]; then
  if [[ -x "$REPO_ROOT/.venv/bin/python" ]]; then PY="$REPO_ROOT/.venv/bin/python"; else PY="python"; fi
fi

# Bundled figure-styling skill — passed to the explorer when present.
NF="$REPO_ROOT/evalvitals/agent_assets/skills/nature-figure"
SKILL_ARGS=()
[[ -d "$NF" ]] && SKILL_ARGS=(--skill "$NF")

step() { printf '\n\033[1;36m== %s ==\033[0m\n' "$*"; }

step "deco_hallu full run  (model=$MODEL device=$DEVICE backend=$BACKEND)"
echo "python : $PY"
echo "repo   : $REPO_ROOT"

step "0/4  build_cases.py  — balanced FAIL/PASS batch (offline)"
"$PY" build_cases.py

step "1/4  run_m1.py  — M1 analyzers, frozen [GPU]"
"$PY" run_m1.py --model "$MODEL" --device "$DEVICE"

step "2/4  run_fused.py  — explore + held-out confirm [claude]"
"$PY" run_fused.py --backend "$BACKEND" "${SKILL_ARGS[@]}"

step "3/4  run_m2-5.py  — M2→M3→M5→Fix [GPU + claude]"
"$PY" run_m2-5.py \
  --model "$MODEL" --device "$DEVICE" --backend "$BACKEND" \
  --recipes        outputs/fused/confirmed_recipes.json \
  --explore-report outputs/fused/fused_report.json

step "4/4  dashboard"
if [[ "$DASHBOARD" == "1" ]]; then
  echo "serving on http://localhost:$PORT  (Ctrl-C to stop)"
  exec "$PY" -m evalvitals.cli dashboard outputs --port "$PORT"
else
  echo "skipped (DASHBOARD=0). Launch it with:"
  echo "  $PY -m evalvitals.cli dashboard outputs --port $PORT"
fi
