"""Prompt templates for M2 statistical analysis."""

_ANALYSIS_PROMPT = """\
You are an expert in ML failure analysis for vision-language models and agentic systems.

Experiment protocol:
{protocol_text}

Task domain: {task_domain}

Analyzer summary:
{narrative}

Raw per-analyzer findings (JSON):
{findings_json}

WHO READS THIS: an engineer who builds and runs evaluations but does not do
statistics. They know what a benchmark, a prompt and a failure case are. They
have never met an e-value, a McNemar test, a bootstrap interval or a
Benjamini-Hochberg correction, and they will not go and look them up.

HOW TO WRITE FOR THEM:
- Answer first. The opening sentence says what is going wrong, in ordinary
  words. Numbers, mechanism and caveats all come after it.
- A bare number tells this reader nothing. Give the count AND what it amounts
  to: "the model got 45 of the 125 wrong, and almost all of those were lists
  longer than eight words" — not "fail rate 0.36, effect +0.28".
- Prefer counts to rates, and give both when you have them ("24 of 30 cases,
  80%"). For a comparison, name both groups and how many cases each holds.
- If a statistical term really is the clearest word, define it in the same
  sentence the first time it appears: "an e-value of 45 — the evidence runs
  about 45 to 1 against this being chance". Never leave one standing bare.
- Never make a bare metric identifier the subject of a sentence. Say what it
  measures, then put the identifier in parentheses if the reader needs the
  handle: "how often the model answered Yes (answer_extraction_audit.answered_yes)".
- No symbols standing in for words (→, ρ, σ), and no "p<0.05" shorthand.
- These are patterns on the analysis split, not proven causes. Write "shows up
  more often on" or "is linked with", never "causes" or "explains", unless an
  intervention actually tested it.

Based on the protocol and the findings above, write:

CONCLUSION: <one paragraph. FIRST sentence: what is going wrong, in plain words,
  no numbers. Then the evidence for it, with counts. Then what this evidence
  does NOT settle. If the model looks healthy given the protocol, say that
  plainly in the first sentence.>
EVIDENCE_CHAIN:
- <step 1: which measurement first caught your attention — say what it measures
  in words, not just its identifier — and why it stood out>
- <step 2: how it connects to the protocol's stated failure patterns>
- <step 3: any corroborating or contradicting signals from other analyzers>
QUALITATIVE:
- <observation 1: a pattern not captured by numbers alone>
- <observation 2: anything unexpected or surprising>

Keep each bullet to one sentence."""

_TOOL_SELECT_PROMPT = """\
You are selecting statistical tools to test why a model fails, given the data on hand.

Experiment protocol:
{protocol_text}

Available statistical tools:
{tool_catalog}

Data shape available for testing:
{data_shape}

Pick the tools whose data requirements are satisfied and that best test the \
protocol's question. Return ONLY a JSON object, no other text:
{{"tools": ["name1", "name2", ...], "rationale": "one sentence"}}"""
