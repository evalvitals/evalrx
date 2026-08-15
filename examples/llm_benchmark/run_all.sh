#!/usr/bin/env bash
# One command, whole chain: serve -> wait -> build cases -> M1..M4 -> release GPU.
#
#   ./run_all.sh qwen3.5-9b supergpqa_law            # full chain, ALL items
#   ./run_all.sh qwen3.5-2b cruxeval_output 60       # cap at 60 items
#   ANALYSIS_ONLY=1 ./run_all.sh qwen3.5-9b bamboogle 40   # M1->M3, no M5/M4
#
# Written for unattended/agent execution: absolute interpreter paths (no shell
# variables carried between steps), an explicit readiness wait, a free-GPU probe,
# and a cleanup trap so a failure anywhere still releases the card.
set -uo pipefail

MODEL="${1:-qwen3.5-9b}"
DATASET="${2:-supergpqa_law}"
NCASES="${3:-0}"   # 0 = every item in the slice
PORT="${PORT:-8020}"
ANALYSIS_ONLY="${ANALYSIS_ONLY:-0}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG_ROOT="$(cd "$HERE/../.." && pwd)"   # the dir holding pyproject.toml

# Resolve interpreters WITHOUT hardcoding a machine. Order: explicit override ->
# a venv inside the checkout -> whatever is on PATH. Serving and evaluating are
# allowed to be different interpreters because vLLM pins versions that the rest
# of the toolchain does not want.
resolve() {  # resolve <override> <candidate>...
  local override="$1"; shift
  if [ -n "$override" ]; then echo "$override"; return; fi
  for c in "$@"; do [ -x "$c" ] && { echo "$c"; return; }; done
  echo ""
}
EVAL_PY="$(resolve "${EVALVITALS_PYTHON:-}" \
  "$PKG_ROOT/.venv/bin/python" "$PKG_ROOT/../.venv/bin/python" \
  "$(command -v python3 || true)")"
VLLM_BIN="$(resolve "${VLLM_BIN:-}" \
  "$PKG_ROOT/.venv-vllm/bin/vllm" "$PKG_ROOT/../.venv-vllm/bin/vllm" \
  "$(command -v vllm || true)")"

if [ -z "$EVAL_PY" ]; then
  echo "no python found. Set EVALVITALS_PYTHON=/path/to/python" >&2; exit 6
fi
if [ -z "$VLLM_BIN" ]; then
  echo "no vllm found. Set VLLM_BIN=/path/to/vllm (see README section 2)" >&2; exit 6
fi

declare -A REPO=(
  [qwen3.5-2b]=Qwen/Qwen3.5-2B
  [qwen3.5-4b]=Qwen/Qwen3.5-4B
  [qwen3.5-9b]=Qwen/Qwen3.5-9B
)
HF_REPO="${REPO[$MODEL]:-}"
if [ -z "$HF_REPO" ]; then
  echo "unknown model '$MODEL'. Use one of: ${!REPO[*]}" >&2
  exit 2
fi

stamp() { echo "[$(date -Iseconds)] $*"; }

# torch orders devices by capability, so CUDA_VISIBLE_DEVICES=0 can resolve to a
# different card than nvidia-smi's 0 -- which shows up as an inexplicable OOM.
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export VLLM_USE_FLASHINFER_SAMPLER=0

GPU="${GPU:-}"
if [ -z "$GPU" ]; then
  GPU=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader \
        | awk -F', ' '$2+0 < 1000 {print $1; exit}')
fi
if [ -z "$GPU" ]; then
  echo "no free GPU (all cards have >1GB in use). Set GPU=<index> to override." >&2
  exit 3
fi
export CUDA_VISIBLE_DEVICES="$GPU"
N_LABEL=$([ "$NCASES" -le 0 ] && echo "ALL" || echo "$NCASES")
stamp "model=$MODEL ($HF_REPO)  dataset=$DATASET  n=$N_LABEL  gpu=$GPU  port=$PORT"

# export so the preflight child sees the SAME interpreters this script resolved,
# otherwise it reports vllm missing while run_all is about to use it
export VLLM_BIN EVALVITALS_PYTHON="$EVAL_PY"

if [ "${SKIP_PREFLIGHT:-0}" != "1" ]; then
  stamp "preflight"
  "$EVAL_PY" "$HERE/preflight.py" --model "$MODEL" --dataset "$DATASET" || exit 7
fi

VLLM_PID=""
cleanup() {
  if [ -n "$VLLM_PID" ] && kill -0 "$VLLM_PID" 2>/dev/null; then
    stamp "stopping vllm pid $VLLM_PID"
    kill "$VLLM_PID" 2>/dev/null
    for _ in $(seq 1 30); do kill -0 "$VLLM_PID" 2>/dev/null || break; sleep 2; done
    kill -9 "$VLLM_PID" 2>/dev/null
  fi
  stamp "GPU released"
}
trap cleanup EXIT INT TERM

LOG_DIR="$HERE/outputs/$MODEL/$DATASET"
mkdir -p "$LOG_DIR"

stamp "launching vllm (weights load takes 3-6 min on a cold page cache)"
"$VLLM_BIN" serve "$HF_REPO" \
  --served-model-name "$MODEL" --port "$PORT" \
  --max-model-len 32768 --max-num-seqs 32 --gpu-memory-utilization 0.92 \
  > "$LOG_DIR/vllm.log" 2>&1 &
VLLM_PID=$!

for _ in $(seq 1 120); do
  if curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
    stamp "endpoint ready"; READY=1; break
  fi
  # a dead server must fail fast, not sit here for 20 minutes
  if ! kill -0 "$VLLM_PID" 2>/dev/null; then
    stamp "FATAL: vllm exited during startup. Last lines:"; tail -25 "$LOG_DIR/vllm.log"
    exit 4
  fi
  sleep 10
done
if [ "${READY:-0}" != "1" ]; then
  stamp "FATAL: endpoint never came up within 20 min"; tail -25 "$LOG_DIR/vllm.log"; exit 5
fi

BASE_URL="http://127.0.0.1:$PORT/v1"

stamp "STAGE 0 build_cases (slowest step; a full census of a 650+ item slice runs 4-6 h — see README)"
"$EVAL_PY" "$HERE/build_cases.py" \
  --model "$MODEL" --dataset "$DATASET" --n "$NCASES" --base-url "$BASE_URL"
rc=$?
if [ $rc -ne 0 ]; then
  # exit 1 here is usually the deliberate out-of-band refusal, not a crash
  stamp "build_cases exited $rc — if it refused on band position, pick a dataset"
  stamp "that sits mid-band for THIS model size (see README section 1)."
  exit $rc
fi

if [ "$ANALYSIS_ONLY" = "1" ]; then
  stamp "STAGE 1 run_pipeline --analysis-only (M1->M2->M3)"
  "$EVAL_PY" "$HERE/run_pipeline.py" \
    --model "$MODEL" --dataset "$DATASET" --base-url "$BASE_URL" --analysis-only
else
  stamp "STAGE 2 run_pipeline (M1->M2->M3->M5->M4)"
  "$EVAL_PY" "$HERE/run_pipeline.py" \
    --model "$MODEL" --dataset "$DATASET" --base-url "$BASE_URL"
fi
rc=$?

stamp "done (rc=$rc). Results in $LOG_DIR"
echo
echo "  dashboard:"
echo "    cd $PKG_ROOT"
echo "    $EVAL_PY -m evalvitals.cli dashboard examples/llm_benchmark/outputs/$MODEL/$DATASET"
exit $rc
