"""Post-loop tiered repair (M5 fix path).

:class:`~.agent.FixAgent` orchestrates; each other module owns one piece:

  types.py           FixCandidate / FixValidation / FixOutcome / FixContext
  tiers.py           FixTier ladder (L0 runtime → L4 parameter space) + routing
  proposal.py        FixProposer: candidates per tier, built-in repairs as data
  codegen.py         PipelineCoder: write / run / repair agent-written pipelines
  validation.py      CandidateValidator, Baseline: run candidates by kind,
                     paired test against the baseline
  verdict.py         verdicts, e-BH survivors, no-fix recommendations
  recorder.py        FixRecorder: the only writer to the run logger
  descriptions.py    plain-language candidate descriptions
  tools.py           L2 tool catalog + PipelineSpec executor
  pipeline.py        L2 coded-pipeline sandbox with the model bridge
  internals.py       L3b intervention primitives, L4 LoRA repair
  repair_catalog.py  discovery of pre-audited model repair capabilities

Import ``FixAgent`` from :mod:`evalrx.eval_agent`.
"""
