"""Prompt templates for M3 diagnosis."""

_DIAGNOSE_PROMPT = """\
You are an expert ML diagnostician. Based on the analysis report below, propose
specific, falsifiable hypotheses about the root cause of the model's failures.
{prior_section}
Model: {model_name}
Overall severity (threshold rules): {severity}

Analysis conclusion (the analyst's interpretation):
{conclusion}
{evidence_section}{stats_section}{explore_section}{failure_modes_section}
Raw findings (JSON):
{findings_json}

{available_signals_section}Propose 1-3 hypotheses. For each write exactly four lines:
HYPOTHESIS: <one-sentence falsifiable claim about the failure mode>
FAILURE_MODE: <short snake_case tag naming the MECHANISM, not the symptom.
  Vision/agent: attention_sink / hallucination / loop / ignored_obs / language_prior_bias
  Text reasoning: computation_slip / chain_break / knowledge_gap / selection_failure /
    overthinking / brittleness / memorization / self_correction_failure
  Harness (suspect these before any mechanism): answer_extraction / truncation /
    degenerate_repetition
  Use a tag from these lists when one fits; invent one only when none does.>
TEST: <which evidence verifies this claim — name a signal/analyzer from the
available evidence list when one fits (e.g. "relative_attention.max_relative_weight"
or "prompt_contrast describe_first contrast"); otherwise describe the analyzer
or intervention that should be run next cycle>
EXPECTED_ASSOCIATION: <higher_on_failures if larger/present values of the named
signal support the hypothesis, or lower_on_failures if smaller/absent values
support it. Pre-register this direction from the claim; do not infer it from an
observed effect. For an intervention expected to repair failures, use
higher_on_failures for a fixed_by_* signal.>

Base your hypotheses on the analysis conclusion and evidence above — an analyzer
can surface a real failure mode even when no numeric threshold fired, so do NOT
rely on the threshold severity alone.
Do NOT repeat hypotheses already listed in the prior cycles above.
If the conclusion and evidence genuinely show no problem, respond with: NO_ISSUE"""

_VALIDATE_PROMPT = """\
You are an adversarial ML reviewer. Your job is to find reasons to REJECT each
hypothesis below. Only approve a hypothesis if you cannot find a significant flaw.

Check each for:
1. Unsupported claim — does the cited evidence actually imply this failure mode?
2. Circular reasoning — does the hypothesis merely restate the symptom?
3. Overgeneralisation — does it make a claim far broader than the evidence supports?
4. Confounded alternative — is there a simpler explanation the hypothesis ignores?

Findings summary (the evidence the hypotheses were drawn from):
{findings_json}

Hypotheses to review:
{hypotheses_text}

For each hypothesis output exactly two lines:
KEEP: <hypothesis statement>  or  REJECT: <hypothesis statement>
REASON: <specific flaw, or "evidence directly supports this claim" if keeping>"""
