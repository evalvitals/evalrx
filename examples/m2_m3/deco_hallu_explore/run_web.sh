#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

# Upload-and-explore web workbench — the ONE page for this example's results:
# anyone can drop a .zip of results (JSON/JSONL/CSV — e.g. a zipped copy of
# data_attn_full/) and each upload becomes one `evalvitals explore` run (M2+M3)
# rendering in place; results already produced by the sibling scripts
# (run_attn.sh -> outputs_attn_full, run_attn_pipeline.sh ->
# outputs_pipeline/1_explore, run.sh -> outputs) are attached read-only in the
# same sidebar. Every result uses the SAME fixed five-tab layout — stages a run
# never reached (held-out verdicts / fix) show as greyed "not available"
# panels instead of disappearing. Runs execute as detached subprocesses, so
# closing the browser never kills an analysis.
#
# Env overrides: PORT / WORKSPACE / CODER_PROVIDER / CODER_MODEL / TIMEOUT_SEC /
# ATTACH_DIRS (space-separated result dirs to list; only existing ones attach).

PORT="${PORT:-8500}"
WORKSPACE="${WORKSPACE:-web_runs}"
CODER_PROVIDER="${CODER_PROVIDER:-claude_code}"
CODER_MODEL="${CODER_MODEL:-}"
ATTACH_DIRS="${ATTACH_DIRS:-outputs_attn_full outputs_pipeline/1_explore outputs}"

cmd=(
  evalvitals web "$WORKSPACE"
  --port "$PORT"
  --backend "$CODER_PROVIDER"
  --timeout-sec "${TIMEOUT_SEC:-1200}"
)
if [[ -n "$CODER_MODEL" ]]; then
  cmd+=(--model "$CODER_MODEL")
fi
for d in $ATTACH_DIRS; do
  if [[ -d "$d" ]]; then
    cmd+=(--attach "$d")
  fi
done

exec "${cmd[@]}"
