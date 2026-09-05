# M4 → M5: Verification & Intervention

Once M2/M3 (or a full diagnosis loop) has a candidate hypothesis, two
loop-internal agents decide whether it's real and whether it can be fixed:

- **M4 — `HypothesisTester`**: verifies a hypothesis two ways — a
  statistical test (do flagged cases fail more often than non-flagged
  cases, McNemar + e-value / e-BH corrected) and a protocol-consistency
  check (does it match what the user said they were investigating). A
  hypothesis is only `SUPPORTED` when both hold; this is the gate the loop
  checks before stopping.
- **M5 — intervention**: two paths, chosen by what you're trying to learn:
  - **`SurgeryAgent`** verifies *why* something fails (correlation check or
    param sweep, no repair attempt).
  - **`FixAgent`** proposes and validates candidate *fixes* for a verified
    hypothesis, each checked against the unmodified baseline with paired
    McNemar + e-value — never a bare p-value.

This page covers the loop-internal M4/M5 workflow. For the standalone,
no-code exploratory stage, see [Exploratory Analysis (M2/M3)](m2_analysis.md).
For stage contracts and full data flow, see
[Architecture](architecture.md#eval_agent-automated-diagnosis-pipeline).

## Quickstart — verify, then fix

```python
from evalrx.eval_agent import VLDiagnoseLoop, RunLogger
from evalrx.eval_agent.stages.probe_agent import ProbeAgent
from evalrx.analysis.stats_agent import StatsAnalysisAgent
from evalrx.eval_agent.stages.diagnosis import DiagnosisAgent

loop = VLDiagnoseLoop(
    model=model,
    probe_agent=ProbeAgent(max_analyzers=3),
    stats_agent=StatsAnalysisAgent(judge=judge),      # feeds M4
    diagnosis_agent=DiagnosisAgent(judge=judge),
    max_cycles=3,
    run_logger=RunLogger(),
)
report = loop.run(failure_cases)   # M1→M2→M3→M4; stops on a supported, consistent hypothesis

print(report.resolved)             # True once M4 confirms a hypothesis
print(report.final_hypotheses)     # status: SUPPORTED / REFUTED / INCONCLUSIVE

# M5, post-loop: propose a targeted fix for the best verified hypothesis
outcome = loop.run_fix(report, failure_cases)
print(outcome.fixed)               # True if a candidate validated
print(outcome.best)                # winning FixValidation, or None
print(outcome.recommendation)      # e.g. {"recommend_tier": "L3a", "reason": ...} when nothing validated
```

## Fix tiers (`FixTier`)

`FixAgent` proposes candidates inside an allowed intervention space — an
**input**, default `L2`. There is no automatic escalation unless you ask for
it:

| Tier | Space | What changes |
|---|---|---|
| L1 | prompt | Judge-proposed prompt rewrites / instruction strategies. |
| L2 | scaffold | Agent-designed pipeline around the *unchanged* model (multi-call, tools, aggregation) — sandboxed, bridged model access; labels never reach the code. |
| L3a | internals (read) | Reads attention/logits to guide scaffold actions. |
| L3b | internals (write) | Modifies the forward pass (attention reweighting, sink suppression, activation steering). |
| L4 | parameter space | Fine-tune recipe, always recorded for a human decision. v1 additionally *executes* one shape — LoRA on the language model — when you pass `FixAgent(finetune_pool=...)` (a diagnosis-only `CaseBatch`, disjoint from what's being validated); other recipe shapes stay recorded-only. |

```python
outcome = loop.run_fix(report, failure_cases, auto_escalate=True)  # steps L2 → L3a → L3b
```

`auto_escalate=True` steps the ceiling tier automatically, stopping as soon
as a candidate validates and feeding each round the full history of prior
failures so the judge proposes genuinely different strategies rather than
repeating one that already failed.

A *fixed* verdict means paired McNemar rejects with a positive net effect —
the candidate repairs significantly more cases than it breaks — and, when
multiple candidates were tried, the candidate's e-value survives e-BH
correction across the tested family (`outcome.ebh_survivors`).

## Custom verification (`SurgeryAgent`)

Swap in domain-specific logic instead of the default label-correlation check:

```python
from evalrx.eval_agent import SurgeryAgent, InterventionResult, HypothesisStatus

def my_verify(hypothesis, model, results, data):
    fixed = run_my_intervention(model, data)
    return InterventionResult(
        hypothesis=hypothesis,
        status=HypothesisStatus.SUPPORTED if fixed else HypothesisStatus.INCONCLUSIVE,
        fixed=fixed,
        evidence={"custom": True},
    )

loop = AutoDiagnoseLoop(model=model, diagnosis_agent=DiagnosisAgent(judge=judge),
                         surgery_agent=SurgeryAgent(verify_fn=my_verify))
```

## Output layout

With a `RunContext` attached (see
[RunContext](architecture.md#runcontext-single-owner-of-a-runs-output-directory)),
each `FixAgent` candidate and each `ExperimentWriter` trial gets its own
folder under the run directory:

```text
run_dir/
  fixes/NN_label/         # one folder per FixAgent candidate: code, sandbox, validation record
  experiments/NN_label/   # one folder per ExperimentWriter trial (M5 code-writing path)
```

## Notes

- `loop.run_fix` and `loop.run_m5` both re-split held-out data automatically:
  hypotheses are mined on the `explore` partition, so verification and fixes
  are validated on `confirm` — cases the loop never used to pick them.
- `ExperimentWriter` (used when `FixAgent` needs to author a multi-file L2
  coded pipeline) supports the same CLI agent backends as M2/M3 explore:
  `claude_code`, `codex`, `opencode`, `gemini_cli`, `kimi_cli`.
