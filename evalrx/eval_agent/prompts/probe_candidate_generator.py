"""Prompt for ProbeLLM-style Macro/Micro question paraphrasing (VLM probe search)."""

from __future__ import annotations

PARAPHRASE_PROMPT = """\
You are helping stress-test a vision-language model. Below is a question that
was originally asked about an image. Rewrite it as {style}, while preserving
its exact meaning and its correct answer — do not change what is being asked
about, only how it is phrased.

Original question: {question}

Reply with ONLY the rewritten question text — no quotes, no explanation, no
"Question:" prefix."""
