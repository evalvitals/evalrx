"""Prompt templates for M5 hypothesis testing."""

_CONSISTENCY_PROMPT = """\
Experiment protocol:
{protocol_text}

Hypothesis under review:
Statement: {statement}
Predicted failure mode: {failure_mode}

Does this hypothesis address a failure mode that is relevant to the experiment protocol above?
Answer YES or NO on the first line, then give a one-sentence reason.

Answer YES if the hypothesis explains a failure that the protocol would care about.
Answer NO if the hypothesis is about something the protocol does not mention or test."""
