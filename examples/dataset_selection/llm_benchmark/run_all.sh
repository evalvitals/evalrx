#!/usr/bin/env bash
# One command, whole chain: serve -> wait -> build cases -> M1..M4 -> release GPU.
#
#   ./run_all.sh qwen3.5-9b supergpqa_law            # full chain, ALL items
#   ./run_all.sh qwen3.5-2b cruxeval_output 60       # cap at 60 items
#   ANALYSIS_ONLY=1 ./run_all.sh qwen3.5-9b bamboogle 40   # M1->M3, no M5/M4
#   SKIP_STAGE0=1 CONFIRM_ONLY=1 ./run_all.sh qwen3.5-2b bbh_word_sorting
#                                     # M5->M4->fix only, on the last run's M2/M3
#   EXPLORE=0 ./run_all.sh qwen3.5-2b bbh_word_sorting   # no explore step (catalog M2 only)
#   SKIP_STAGE0=1 ANALYSIS_ONLY=1 RUN_TAG=smoke MAX_CASES=60 ./run_all.sh qwen3.5-2b bbh_word_sorting
#                                     # smoke run: 60-case subsample of the frozen batch,
#                                     # M1->explore->M2->M3, everything under <dataset>.smoke/
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
CONFIRM_ONLY="${CONFIRM_ONLY:-0}"    # reuse logs/ M2+M3; needs the frozen batch (SKIP_STAGE0=1)
EXPLORE="${EXPLORE:-1}"              # 0 = skip the in-cycle explore step (free-form EDA beside M2)
RUN_TAG="${RUN_TAG:-}"               # set = write to outputs/<model>/<dataset>.<tag>/ (smoke runs; needs SKIP_STAGE0=1)
MAX_CASES="${MAX_CASES:-0}"          # >0 = label-stratified subsample of the frozen batch (smoke runs)
PIPELINE_ARGS="${PIPELINE_ARGS:-}"   # extra run_pipeline.py flags, e.g. "--analyzer-max-cases 16"
BUILD_ARGS="${BUILD_ARGS:-}"         # extra build_cases.py flags, e.g. "--force" (write an out-of-band batch)

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Walk up to the checkout root instead of counting directories: this example
# moved one level deeper in an examples/ reorg, and a fixed "../.." silently
# resolved to examples/ — which still exists, so nothing errored.
PKG_ROOT="$HERE"
while [ "$PKG_ROOT" != "/" ] && [ ! -f "$PKG_ROOT/pyproject.toml" ]; do
  PKG_ROOT="$(dirname "$PKG_ROOT")"
done
if [ ! -f "$PKG_ROOT/pyproject.toml" ]; then
  echo "cannot find the checkout root (no pyproject.toml above $HERE)" >&2; exit 6
fi

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
BASE_DIR="$LOG_DIR"
if [ -n "$RUN_TAG" ]; then
  # A tagged run keeps a real run's logs/ (append-only) and explore/ untouched.
  # Stage 0 always writes the UNTAGGED batch, so a tagged run must reuse one.
  if [ "${SKIP_STAGE0:-0}" != "1" ]; then
    stamp "RUN_TAG=$RUN_TAG requires SKIP_STAGE0=1 (build_cases writes the untagged batch)"; exit 1
  fi
  LOG_DIR="$HERE/outputs/$MODEL/$DATASET.$RUN_TAG"
fi
mkdir -p "$LOG_DIR"

# The server's context has to be sized BEFORE it starts, and some datasets need
# more than the default. minervamath's 0.500 reference was measured at a 65k
# generation budget; at 40k the same slice reads 0.320 and budget_limited, so a
# 32768 context would not shorten the run, it would change what it measures.
DS_TOKENS=$("$EVAL_PY" "$HERE/datasets.py" --max-tokens "$DATASET" 2>/dev/null || echo 0)
CFG_TOKENS=$(awk '/^max_tokens:/ {print $2; exit}' "$HERE/config.yaml")
GEN_TOKENS="${MAX_TOKENS:-0}"
[ "$GEN_TOKENS" -le 0 ] && GEN_TOKENS=$([ "${DS_TOKENS:-0}" -gt 0 ] && echo "$DS_TOKENS" || echo "$CFG_TOKENS")
# leave room for the prompt on top of the generation budget
MODEL_LEN="${MAX_MODEL_LEN:-$((GEN_TOKENS + 8192))}"
[ "$MODEL_LEN" -lt 32768 ] && MODEL_LEN=32768
stamp "generation budget $GEN_TOKENS tok -> --max-model-len $MODEL_LEN"

stamp "launching vllm (weights load takes 3-6 min on a cold page cache)"
"$VLLM_BIN" serve "$HF_REPO" \
  --served-model-name "$MODEL" --port "$PORT" \
  --max-model-len "$MODEL_LEN" --max-num-seqs 32 --gpu-memory-utilization 0.92 \
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

# SKIP_STAGE0=1 reuses the frozen batch. This is the whole reason Stage 0 is a
# separate step: when M1-M4 fails, the four hours of GPU generation that
# preceded it are still valid, and regenerating them would only add noise (the
# sampler is not seeded). Refuses rather than silently regenerating if absent.
if [ "${SKIP_STAGE0:-0}" = "1" ]; then
  if [ ! -f "$LOG_DIR/cases.json" ] && [ -n "$RUN_TAG" ] && [ -f "$BASE_DIR/cases.json" ]; then
    cp "$BASE_DIR/cases.json" "$LOG_DIR/cases.json"   # self-contained tagged run
  fi
  if [ ! -f "$LOG_DIR/cases.json" ]; then
    stamp "SKIP_STAGE0=1 but $LOG_DIR/cases.json does not exist"; exit 1
  fi
  stamp "STAGE 0 skipped — reusing $LOG_DIR/cases.json"
  rc=0
else
  stamp "STAGE 0 build_cases (slowest step; a full census of a 650+ item slice runs 4-6 h — see README)"
  # shellcheck disable=SC2206  # BUILD_ARGS is deliberately word-split
  BEXTRA=($BUILD_ARGS)
  "$EVAL_PY" -u "$HERE/build_cases.py" \
    --model "$MODEL" --dataset "$DATASET" --n "$NCASES" --base-url "$BASE_URL" "${BEXTRA[@]}"
  rc=$?
  if [ $rc -ne 0 ]; then
    # exit 1 here is usually the deliberate out-of-band refusal, not a crash
    stamp "build_cases exited $rc — if it refused on band position, pick a dataset"
    stamp "that sits mid-band for THIS model size (see README section 1)."
    exit $rc
  fi
fi

# Flags shared by every run_pipeline invocation below.
COMMON=(--model "$MODEL" --dataset "$DATASET" --base-url "$BASE_URL")
if [ -n "$RUN_TAG" ]; then COMMON+=(--out-tag "$RUN_TAG"); fi
if [ "$MAX_CASES" -gt 0 ] 2>/dev/null; then COMMON+=(--max-cases "$MAX_CASES"); fi
if [ "$EXPLORE" = "0" ]; then COMMON+=(--no-explore); fi
# shellcheck disable=SC2206  # PIPELINE_ARGS is deliberately word-split
EXTRA=($PIPELINE_ARGS)
if [ "$ANALYSIS_ONLY" = "1" ]; then
  stamp "STAGE 1 run_pipeline --analysis-only (M1->[explore]->M2->M3)"
  "$EVAL_PY" -u "$HERE/run_pipeline.py" "${COMMON[@]}" --analysis-only "${EXTRA[@]}"
elif [ "$CONFIRM_ONLY" = "1" ]; then
  stamp "STAGE 2' run_pipeline --confirm-only (M5->M4->fix on the last run's M2/M3; logs_confirm/)"
  "$EVAL_PY" -u "$HERE/run_pipeline.py" "${COMMON[@]}" --confirm-only "${EXTRA[@]}"
else
  stamp "STAGE 2 run_pipeline (M1->[explore]->M2->M3->M5->M4)"
  "$EVAL_PY" -u "$HERE/run_pipeline.py" "${COMMON[@]}" "${EXTRA[@]}"
fi
rc=$?

# ---- Stage W (optional): internals for a small subset -----------------------
# Skipped unless WHITEBOX_PYTHON points at an interpreter whose transformers
# knows the architecture (5.15.0 for Qwen3.5; the evalvitals venv's 4.57.6 does
# not). It runs AFTER vllm is stopped on purpose: the server holds 92% of the
# card, and loading the same weights again in transformers needs that back.
if [ -n "${WHITEBOX_PYTHON:-}" ] && [ $rc -eq 0 ]; then
  stamp "stopping vllm before STAGE W (transformers needs the VRAM back)"
  cleanup; trap - EXIT INT TERM; VLLM_PID=""
  for _ in $(seq 1 30); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU")
    [ "${used:-9999}" -lt 2000 ] && break
    sleep 2
  done
  stamp "STAGE W run_whitebox (attention over a label-balanced subset)"
  "$WHITEBOX_PYTHON" -u "$HERE/run_whitebox.py" \
    --model "$MODEL" --dataset "$DATASET" --n "${WHITEBOX_N:-24}"
  wrc=$?
  [ $wrc -ne 0 ] && stamp "stage W exited $wrc (the main chain already succeeded)"
elif [ -z "${WHITEBOX_PYTHON:-}" ]; then
  stamp "STAGE W skipped: WHITEBOX_PYTHON unset — no attention/hidden-state"
  stamp "  analyzers ran. See README 'Stage W' if you want them."
fi

stamp "done (rc=$rc). Results in $LOG_DIR"
echo
echo "  dashboard:"
echo "    cd $PKG_ROOT"
echo "    $EVAL_PY -m evalvitals.cli dashboard examples/dataset_selection/llm_benchmark/outputs/$MODEL/$DATASET"
exit $rc
