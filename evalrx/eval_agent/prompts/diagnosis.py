"""Prompt templates for M3 diagnosis."""

_DIAGNOSE_PROMPT = """\
You are an expert ML diagnostician. Based on the analysis report below, propose
specific, falsifiable hypotheses about the root cause of the model's failures.

Each hypothesis is written twice: once for a specialist (hypothesis, test) and
once for the person who will read the report. That reader builds and runs
evaluations but does not do statistics — they know what a benchmark, a prompt
and a failure case are, and they have never met an e-value, a McNemar test or a
multiple-comparisons correction. The plain_statement field is written for them.
{prior_section}
Model: {model_name}
Overall severity (threshold rules): {severity}

Experiment protocol (authoritative task and response contract):
{protocol_section}

Analysis conclusion (the analyst's interpretation):
{conclusion}
{evidence_section}{stats_section}{explore_section}{failure_modes_section}
Raw findings (JSON):
{findings_json}

{available_signals_section}Propose 1-3 hypotheses. For each give:
- hypothesis: one-sentence falsifiable technical claim about the failure mode.
- plain_statement: the SAME claim restated in ONE everyday sentence, written for
  the reader described above. State what the model is doing wrong, not what
  statistic was computed. Numbers are welcome and make it better — but a number
  must come with what it amounts to ("gets 45 of the 125 wrong, nearly all of
  them lists longer than eight words"), never bare ("fail rate 0.36"). No
  acronyms, no statistics jargon, no symbols (→, ρ, σ), no bare metric
  identifiers as the subject of the sentence. Do not copy the hypothesis
  verbatim — this is checked, and a jargon-y or copied one is sent back.
- failure_mode: short snake_case tag naming the MECHANISM, not the symptom.
  Vision/agent: attention_sink / hallucination / loop / ignored_obs / language_prior_bias
  White-box media flow: visual_neglect / audio_neglect / insufficient_visual_attention /
    visual_information_loss / mislocalized_attention / late_suppression
  Text reasoning: computation_slip / chain_break / knowledge_gap / selection_failure /
    overthinking / brittleness / memorization / self_correction_failure
  Harness (suspect these before any mechanism): answer_extraction / truncation /
    degenerate_repetition
  Use a tag from these lists when one fits; invent one only when none does.
- test: which evidence verifies this claim — name a signal/analyzer from the
available evidence list when one fits (e.g. "relative_attention.max_relative_weight
HIGHER on failing cases" or "prompt_contrast describe_first contrast"), and for a
per-case signal SAY whether it should be HIGHER or LOWER on failing cases — that
is the prediction the test checks. On a yes/no task a directional claim ("answers
Yes regardless of the evidence") is tested on the DIRECTION marginals
answer_extraction_audit.answered_yes (the answer's direction, e.g. HIGHER on
failing cases) and answer_extraction_audit.gold_yes (the question's), never on
extracted_answer, labelled_fail or a correctness flag — those are text or the
label itself and cannot be evidence; otherwise describe the analyzer or
intervention that should be run next cycle.
- expected_association: higher_on_failures if larger/present values of the named
signal support the hypothesis, or lower_on_failures if smaller/absent values
support it. Pre-register this direction from the claim; do not infer it from an
observed effect. For an intervention expected to repair failures, use
higher_on_failures for a fixed_by_* signal.

Base your hypotheses on the analysis conclusion and evidence above — an analyzer
can surface a real failure mode even when no numeric threshold fired, so do NOT
rely on the threshold severity alone.
Do NOT repeat hypotheses already listed in the prior cycles above.
If the conclusion and evidence genuinely show no problem, return no hypotheses."""

_VALIDATE_PROMPT = """\
You are an adversarial ML reviewer. Your job is to find reasons to REJECT each
hypothesis below. Only approve a hypothesis if you cannot find a significant flaw.

Check each for:
1. Unsupported claim — does the cited evidence actually imply this failure mode?
2. Circular reasoning — does the hypothesis merely restate the symptom?
3. Overgeneralisation — does it make a claim far broader than the evidence supports?
4. Confounded alternative — is there a simpler explanation the hypothesis ignores?
{context_section}
Findings summary (the evidence the hypotheses were drawn from):
{findings_json}

Hypotheses to review (numbered):
{hypotheses_text}

Give one decision per hypothesis, by its number: keep or reject, with a reason
of one or two sentences naming the flaw you found, or "evidence directly
supports this claim" if keeping."""
