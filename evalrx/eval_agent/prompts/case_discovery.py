"""Prompt templates for case discovery and labeling."""

_JUDGE_PROMPT = """\
You are scoring one model answer for an evaluation case.

Experiment protocol:
{protocol}

Success criteria:
{success_criteria}

Prompt:
{prompt}

Expected answer or rubric:
{expected}

Observed model answer:
{observed}

Return a JSON object:
{{"label": "PASS|FAIL|UNKNOWN", "reason": "one concise sentence"}}

Use PASS only when the observed answer satisfies the expected answer/rubric under
the success criteria."""
