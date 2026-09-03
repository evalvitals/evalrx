"""Prompt templates for the ``run_codebase`` entry: a CLI coding agent runs a
user's existing evaluation/inference codebase and harvests its per-case
results into the records-file output contract that
:func:`evalrx.analysis.run_codebase.run_codebase` reads.
"""

from __future__ import annotations

RUN_PROMPT_TEMPLATE = """\
You are working inside a copy of a user's evaluation/inference codebase, at
the current working directory. Your job:

1. Understand the codebase: find its evaluation or inference entry point
   (a script, notebook-derived script, or main/run/eval module) and whatever
   dependencies or config it needs.
2. Run it so it actually executes the evaluation over its dataset. Install
   missing lightweight dependencies if needed. If it requires resources you
   do not have (GPU, network, missing API keys), do the best you can and
   report what happened in your final message.
3. Produce a file named "{records_name}" in this directory: a JSON array or
   JSON-Lines file with ONE ROW PER EVALUATION CASE. Each row MUST include:
   - a "label" field: "PASS"/"FAIL" (or another clear correctness verdict
     for the task) for that case
   - the case's input, prediction/output, and target/expected fields, using
     whatever field names make sense for this task
   If the codebase already writes per-case outputs (predictions, logs, a
   results table), read them and CONVERT them into "{records_name}" in this
   format rather than re-running everything from scratch when that is
   faster and equally faithful.

Do not fabricate results. If the run genuinely fails, still write
"{records_name}" with whatever real per-case rows you were able to produce
(even from a partial run), and explain the failure in your final message.

Task context supplied by the user:
{question}
"""

REPAIR_PROMPT_TEMPLATE = """\
Your previous attempt did not leave a usable "{records_name}" in this
directory ({reason}). Look at what actually happened (error output, partial
results, logs) and try again: get the evaluation to run and write
"{records_name}" as a JSON array or JSON-Lines file, one row per case, each
with a "label" field plus the case's input/prediction/target fields. If a
full run is not possible, write whatever real partial results you have
rather than nothing.
"""
