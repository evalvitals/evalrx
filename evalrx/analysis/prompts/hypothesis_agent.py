"""Prompt template for the standalone M3 HypothesisAgent."""

from __future__ import annotations

# Column-name markers that identify agent-trajectory records (produced by
# evalrx.analysis.trajectory_records / the agent M1 probes).  When any of
# these appear in the report, AGENT_TRAJECTORY_HINT is appended to the propose
# prompt.
AGENT_COLUMN_MARKERS: tuple[str, ...] = (
    "n_tool_calls", "n_calls_", "max_consecutive_repeat", "repeated_call_frac",
    "tool_error_rate", "tool_seq_diversity", "n_images_returned",
    "failure_mode", "shap_outcome_", "shap_answer_",
    "success_rate", "pass_at_k", "pass_all_k", "trajectory",
)

AGENT_TRAJECTORY_HINT = """\
These records describe AGENT TRAJECTORIES (a model calling tools in a loop).
Extra guidance for this data:
- Prefer hypotheses that name an INTERVENABLE cause — something a prompt or
  scaffold change could alter: the system-prompt wording, a tool's description
  or argument conventions, which tools are available, or the loop/stop policy.
  Fixes for agents are black-box only, so "this part of the agent's setup
  causes failures" is repairable, while "failures correlate with X" alone is
  not a mechanism.
- Column-family semantics, where present: shap_outcome_<tool> is the causal
  contribution of that tool to passing (measured by re-running with tool
  subsets); success_rate / flaky / pass_at_k come from repeated runs and
  measure stability, not capability; failure_mode is a judge-assigned label —
  itself a hypothesis to verify, never ground truth; total_*_tokens and
  *_latency_ms are cost.
- A test may propose re-running the agent with a changed prompt, tool
  description, or tool subset on held-out cases — interventions are directly
  testable here, not just observational splits."""

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

Propose 1-3 hypotheses. For each give:
- hypothesis: one falsifiable sentence explaining a pattern above; precise
  language and technical terms are fine here.
- plain_statement: the SAME claim in ONE everyday sentence a non-technical
  reader understands at a glance: no acronyms (AUC, CI...), no statistics
  jargon (collinear, logistic, coefficient, p-value, latent, mediator...), no
  symbols. Numbers and percentages are fine and encouraged.
- basis: which takeaway(s) or signal(s) above it is grounded in.
- test: what evidence or analysis would confirm or refute it.

Do not repeat a takeaway verbatim; propose a CAUSE or MECHANISM behind what
was observed. If the findings are too thin to support any falsifiable
hypothesis, give an empty list."""
