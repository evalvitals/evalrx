"""Prompt template for the held-out hypothesis judge (`analysis.holdout`)."""

JUDGE_PROMPT = """You are adjudicating a proposed hypothesis about failure patterns
in a dataset, using HELD-OUT rows the proposer never saw.

HYPOTHESIS: {statement}
BASIS (from the exploration phase): {basis}
PROPOSED TEST DESIGN: {test_design}

HELD-OUT EVIDENCE (validate split, n={n_rows}; recipe thresholds frozen from the
exploration phase; REJECT H0 means the signal separated the outcome groups on
held-out data after correction for every claim tested; each line also states
the claim that was tested and whether it holds, is contradicted or stays
inconclusive — a claim of the form "|effect| < bound" is a control that was
predicted to come out small):
{evidence}

Grade the hypothesis STRICTLY against this held-out evidence:
- "supported": the held-out statistics directly back the hypothesis's claim.
- "partial": the correlational part holds up but the hypothesis claims more
  (e.g. a mechanism or causal direction) than these statistics can establish.
- "refuted": the held-out statistics contradict the claim.
- "not_testable": these observational statistics cannot decide the claim at
  all (e.g. causal direction); it needs an intervention/surgery experiment.

Give the verdict, 2-3 sentences of reasoning citing the held-out numbers, and
whether the claim needs an intervention (surgery) experiment. Where the host
already checked a numeric prediction, take its result as given and judge only
what the numbers leave open. If an intervention is needed, also give
"experiment" with four short fields: data_needed, operation, expected_result,
and falsified_by (the result that would show the hypothesis is wrong).
"""
