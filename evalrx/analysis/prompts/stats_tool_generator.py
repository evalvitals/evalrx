"""Prompt templates for generated code-writing stages."""

from __future__ import annotations

_INPUT_FILENAME = "m2_stats_input.json"
_RESULT_MARKER = "STATS_RESULT_JSON="

_GENERATE_PROMPT = """\
You are writing a self-contained Python statistics script for a model-failure analysis.

GOAL (what statistic to compute):
{need}

DATA: a JSON file named "{input_filename}" sits in the current working directory with:
{{
  "labels":   {{case_id: is_fail_bool}},          # PASS/FAIL labels
  "per_case": {{"analyzer.metric": {{case_id: value}}}},  # per-case signals
  "scalars":  {{"analyzer.metric": value}},         # aggregate metrics
  "groups":   {{strategy: {{case_id: success}}}} or null  # strategy comparison
}}

Data shape available for THIS run:
{data_shape}

REQUIREMENTS:
- Read "{input_filename}" from the current directory. Do NOT hardcode the data.
- You MAY `import numpy`. No network, no file writes, no other I/O.
- You decide WHAT statistic to compute, but you do NOT adjudicate significance.
  Do NOT emit a "reject", "e_value", or "p_value" verdict — the HOST recomputes
  the decision from your SUFFICIENT STATISTICS with its validated,
  multiplicity-aware core; a self-declared verdict is ignored.
- The LAST line of stdout MUST be exactly one line of the form:
  {marker}{{"summary": "<one sentence>", "effect": <number or null>, "ci": [lo, hi] or null, "underpowered": <true/false>, "details": {{}}, "sufficient": <a SUFFICIENT-STATISTICS object or null>}}
- "sufficient" must be ONE of these host-adjudicable shapes:
    {{"kind": "paired_binary", "b": <int: #cases that flipped the GOOD way>, "c": <int: #cases that flipped the BAD way>}}
    {{"kind": "two_group", "a": [0/1, ...], "b": [0/1, ...]}}   # two independent success/indicator vectors (e.g. is_fail among signal-absent vs signal-present)
  If your statistic cannot be expressed as one of these, set "sufficient": null —
  your tool is then DESCRIPTIVE (it reports effect/CI but can never claim a
  rejection). Choose the shape that captures your test; the host owns the verdict.
- Print NOTHING after that line. Keep the script under ~60 lines.

Return ONLY the Python code{fences_hint}."""

