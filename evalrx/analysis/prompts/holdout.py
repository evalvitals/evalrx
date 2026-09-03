"""Prompt template for the held-out hypothesis judge (`analysis.holdout`)."""

JUDGE_PROMPT = """You are adjudicating a proposed hypothesis about failure patterns
in a dataset, using HELD-OUT rows the proposer never saw.

HYPOTHESIS: {statement}
BASIS (from the exploration phase): {basis}
PROPOSED TEST DESIGN: {test_design}

HELD-OUT EVIDENCE (validate split, n={n_rows}; recipe thresholds frozen from the
exploration phase; REJECT H0 means the signal separated the outcome groups on
held-out data):
{evidence}

Grade the hypothesis STRICTLY against this held-out evidence:
- "supported": the held-out statistics directly back the hypothesis's claim.
- "partial": the correlational part holds up but the hypothesis claims more
  (e.g. a mechanism or causal direction) than these statistics can establish.
- "refuted": the held-out statistics contradict the claim.
- "not_testable": these observational statistics cannot decide the claim at
  all (e.g. causal direction); it needs an intervention/surgery experiment.

Reply with ONLY a JSON object:
{{"verdict": "supported|partial|refuted|not_testable",
  "reasoning": "<2-3 sentences citing the held-out numbers>",
  "needs_surgery": true/false}}
"""
