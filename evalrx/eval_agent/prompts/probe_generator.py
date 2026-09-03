"""Prompt templates for generated code-writing stages."""

from __future__ import annotations

_INPUT_FILENAME = "m1_probe_input.json"
_RESULT_MARKER = "PROBE_RESULT_JSON="
_MAX_OUTPUT_CHARS = 4000

_GENERATE_PROMPT = """\
You are writing a self-contained Python script that probes a model's text outputs
for a specific failure pattern.

GOAL (what to probe for):
{need}

DATA: a JSON file named "{input_filename}" sits in the current working directory:
{{
  "cases": [
    {{"id": "<case id>", "prompt": "<input prompt>", "expected": <expected or null>,
     "label": "pass"|"fail"|"unknown"|null, "output": "<the model's text output>"}}
  ]
}}

REQUIREMENTS:
- Read "{input_filename}" from the current directory; do NOT hardcode the data.
- Compute, per case, a numeric or boolean signal for the failure pattern above
  (e.g. 1 if the output refuses / drifts language / breaks format, else 0).
- Standard library + `import numpy` only. No network, no file writes, no model calls.
- The LAST line of stdout MUST be exactly:
  {marker}{{"findings": {{"<aggregate_metric>": <number>}}, "per_case": [{{"sample_id": "<case id>", "<signal_name>": <number_or_bool>}}, ...]}}
- Every per_case entry MUST carry "sample_id" equal to the case "id".
- Print NOTHING after that line. Keep the script under ~60 lines.

Return ONLY the Python code{fences_hint}."""

