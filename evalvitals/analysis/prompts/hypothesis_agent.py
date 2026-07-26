"""Prompt template for the standalone M3 HypothesisAgent."""

from __future__ import annotations

PROPOSE_PROMPT = """\
You are a data analyst explaining candidate explanations to a general
audience — mostly non-technical stakeholders, not statisticians. Based on
the exploratory analysis below, propose specific, falsifiable hypotheses
that could explain the patterns found. A hypothesis is a candidate
explanation or mechanism — not a restatement of a finding, and not a claim
you are asked to prove here.

Question investigated: {question}

Key takeaways from the exploratory analysis (title: analysis):
{takeaways_text}

Observations:
{observations_text}

Candidate signals already noted:
{signals_text}

Propose 1-3 hypotheses. For each write exactly four lines:
HYPOTHESIS: <one-sentence falsifiable claim explaining a pattern above; precise language and technical terms are fine here>
PLAIN: <the SAME claim restated in ONE everyday sentence a non-technical reader can understand at a glance — NO acronyms (AUC, CI...), NO statistics jargon (collinear, logistic, coefficient, p-value, latent, mediator...), NO symbols. Numbers/percentages are fine and encouraged>
BASIS: <which takeaway(s)/signal(s) above this is grounded in>
TEST: <what evidence/analysis would confirm or refute this claim>

Do not repeat a takeaway verbatim — propose a CAUSE or MECHANISM behind what
was observed. If the findings are too thin to support any falsifiable
hypothesis, respond with exactly: NO_HYPOTHESIS"""

PLAIN_REPAIR_PROMPT = """\
Your previous answer below proposed hypotheses correctly, but some PLAIN
lines fail a plain-language check: PLAIN must be a jargon-free, one-sentence,
everyday paraphrase of the matching HYPOTHESIS line — no statistics terms,
acronyms, or symbols. Numbers/percentages are fine.

Previous answer:
{raw}

Problems found:
{violations}

Rewrite the FULL set of hypotheses in the exact same
HYPOTHESIS/PLAIN/BASIS/TEST format, fixing only the flagged PLAIN lines. Do
not change the HYPOTHESIS, BASIS, or TEST lines."""
