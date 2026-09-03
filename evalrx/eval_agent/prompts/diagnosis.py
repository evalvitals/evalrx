"""Prompt templates for M3 diagnosis."""

_DIAGNOSE_PROMPT = """\
You are an expert ML diagnostician. Based on the analysis report below, propose
specific, falsifiable hypotheses about the root cause of the model's failures.

Each hypothesis is written twice: once for a specialist (HYPOTHESIS, TEST) and
once for the person who will read the report. That reader builds and runs
evaluations but does not do statistics — they know what a benchmark, a prompt
and a failure case are, and they have never met an e-value, a McNemar test or a
multiple-comparisons correction. The PLAIN_STATEMENT line is written for them.
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

{available_signals_section}Propose 1-3 hypotheses. For each write:
HYPOTHESIS: <one-sentence falsifiable technical claim about the failure mode>
PLAIN_STATEMENT: <the SAME claim as HYPOTHESIS, restated in ONE everyday
  sentence, written for the reader described above. State what the model is
  doing wrong, not what statistic was computed. Numbers are welcome and make
  it better — but a number must come with what it amounts to ("gets 45 of the
  125 wrong, nearly all of them lists longer than eight words"), never bare
  ("fail rate 0.36"). No acronyms, no statistics jargon, no symbols (→, ρ, σ),
  no bare metric identifiers as the subject of the sentence. Do not copy the
  HYPOTHESIS line verbatim — this is checked, and a jargon-y or copy-pasted
  line is sent back for a rewrite.>
FAILURE_MODE: <short snake_case tag naming the MECHANISM, not the symptom.
  Vision/agent: attention_sink / hallucination / loop / ignored_obs / language_prior_bias
  Text reasoning: computation_slip / chain_break / knowledge_gap / selection_failure /
    overthinking / brittleness / memorization / self_correction_failure
  Harness (suspect these before any mechanism): answer_extraction / truncation /
    degenerate_repetition
  Use a tag from these lists when one fits; invent one only when none does.>
TEST: <which evidence verifies this claim — name a signal/analyzer from the
available evidence list when one fits (e.g. "relative_attention.max_relative_weight
HIGHER on failing cases" or "prompt_contrast describe_first contrast"), and for a
per-case signal SAY whether it should be HIGHER or LOWER on failing cases — that
is the prediction the test checks. On a yes/no task a directional claim ("answers
Yes regardless of the evidence") is tested on the DIRECTION marginals
answer_extraction_audit.answered_yes (the answer's direction, e.g. HIGHER on
failing cases) and answer_extraction_audit.gold_yes (the question's), never on
extracted_answer, labelled_fail or a correctness flag — those are text or the
label itself and cannot be evidence; otherwise describe the analyzer or
intervention that should be run next cycle>
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
{context_section}
Findings summary (the evidence the hypotheses were drawn from):
{findings_json}

Hypotheses to review:
{hypotheses_text}

For each hypothesis output two lines:
KEEP: <hypothesis statement>  or  REJECT: <hypothesis statement>
REASON: <one or two sentences naming the flaw you found, or why it survives>
REASON: <specific flaw, or "evidence directly supports this claim" if keeping>"""


_PLAIN_REPAIR_PROMPT = """\
Your previous answer below proposed hypotheses correctly, but some
PLAIN_STATEMENT lines fail a plain-language check.

PLAIN_STATEMENT must be a one-sentence everyday paraphrase of the matching
HYPOTHESIS line, written for an engineer who runs evaluations but does no
statistics: no statistics terms, no acronyms, no symbols, and not a verbatim
copy of the HYPOTHESIS line. Numbers are welcome, but each one must come with
what it amounts to rather than standing bare.

Previous answer:
{raw}

Problems found:
{violations}

Rewrite the FULL set of hypotheses in the exact same
HYPOTHESIS/PLAIN_STATEMENT/FAILURE_MODE/TEST/EXPECTED_ASSOCIATION format,
fixing only the flagged PLAIN_STATEMENT lines. Do not change any other line."""
