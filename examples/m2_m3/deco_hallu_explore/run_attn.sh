#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

CODER_PROVIDER="${CODER_PROVIDER:-claude_code}"
CODER_MODEL="${CODER_MODEL:-}"
OUT_DIR="${OUT_DIR:-outputs_attn_full}"

# The attention-enriched variant: data_attn_full/ ships with the repo (606/606
# per-case attention-geometry coverage across the 2B/4B/8B checkpoints), so
# this run needs NO GPU — regenerate the data with extract_attention_all.py
# if you want fresh scalars. The question steers M2 toward the analyses the
# continuous signals make possible: FAIL/PASS distribution views and
# cross-checkpoint comparisons.
cmd=(
  evalvitals
  explore
  data_attn_full
  --backend "$CODER_PROVIDER"
  --outcome-col label
  --out "$OUT_DIR"
  -q "What predicts hallucination failures (label=fail) in this VLM object-presence probe? The per-case attention-geometry scalars (attention_entropy, focus_share, center_offset, edge_mass, top1_share, max/mean_relative_weight) cover ALL 606 cases across three checkpoints (2B/4B/8B). (1) Compare their DISTRIBUTIONS between FAIL and PASS (violin/ECDF), overall and within adversarial probes only. (2) Does the attention geometry of failures change with model size — do the same signals separate FAIL/PASS at 2B, 4B and 8B, and do the distributions themselves shift across checkpoints? (3) Relate attention geometry to probe_type and object. Attention signals are collinear — say which ones carry independent signal."
  --max-attempts 3
  --timeout-sec "${TIMEOUT_SEC:-900}"
)

if [[ -n "$CODER_MODEL" ]]; then
  cmd+=(--model "$CODER_MODEL")
fi

exec "${cmd[@]}"
