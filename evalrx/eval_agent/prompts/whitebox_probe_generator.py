"""Prompt templates for generated code-writing stages."""

from __future__ import annotations

_MANIFEST = "m1_whitebox_manifest.json"
_DUMP_DIR = "m1_whitebox"
_RESULT_MARKER = "PROBE_RESULT_JSON="

_GENERATE_PROMPT = """\
You are writing a self-contained Python script that probes a vision-language
model's INTERNAL attention for a specific failure mechanism.

GOAL (what mechanism to probe for):
{need}

DATA in the current working directory:
- "{manifest}": {{"cases": [{{"id": str, "label": "pass"|"fail"|null,
   "seq_len": int, "n_layers": int, "tokens": [str, ...]}}, ...]}}
- "{dump_dir}/<case_id>.npz" per case with arrays:
   - attn_last:        float16 (n_layers, seq) — head-averaged attention from
                       the LAST query position to every key position, per layer
   - image_token_mask: bool (seq,) — True at image-patch token positions
  The "tokens" list in the manifest gives the token string at each position —
  use it to locate structural/special tokens (e.g. positions whose token
  contains "im_start", "image_pad", "vision").

REQUIREMENTS:
- numpy only (`import numpy as np`); no network, no other file writes.
- Compute ONE per-case numeric signal for the mechanism above (e.g. attention
  mass on structural tokens, image-vs-text attention ratio at a layer,
  attention entropy inside the image region).
- The LAST line of stdout MUST be exactly:
  {marker}{{"findings": {{"<aggregate_metric>": <number>}}, "per_case": [{{"sample_id": "<case id>", "<signal_name>": <number>}}, ...]}}
- Every per_case entry MUST use "sample_id" equal to the case "id".
- Print NOTHING after that line. Keep the script under ~60 lines.

Return ONLY the Python code{fences_hint}."""

