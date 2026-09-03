# EvalRX Documentation

EvalRX is a package for LLM and VLM evaluation designed around the same
engineering posture that made sklearn useful: small composable contracts,
discoverable estimators, uniform result objects, and predictable behavior across
many model/runtime combinations.

Where sklearn standardizes `fit`, `predict`, and `score` around tabular learning,
EvalRX standardizes `generate`, `forward(capture=...)`, `Analyzer.run`, and
`Result` around model behavior, internals, failures, and agent trajectories.

## Core Idea

EvalRX separates three things that are often mixed together:

| Concern | EvalRX object | Example |
|---|---|---|
| Model identity | `ModelSpec` (curated) or inferred via `wrap()` | `qwen2.5-7b-instruct`, or any loaded HF model |
| Runtime | `Backend` | `hf_local`, `api`, `vllm_offline` |
| Analysis | `Analyzer` | `AttentionAnalyzer`, `TokenEntropyAnalyzer` |

This separation lets an analyzer ask for capabilities instead of asking for a
specific model class. For example, an attention analyzer requires
`Capability.ATTENTION`; any model runtime that provides attention traces can run
it.

## Mental Model

There are two ways to get a `Model`:

```text
# Public on-ramp (captum-style): user brings their own loaded model
wrap(hf_model, tokenizer)  ->  Model

# Curated path: load from the spec registry by key
ModelSpec + Backend  ->  compose(...)  ->  Model
```

Both paths produce the same `Model` object — the same analyzers work on both.

```text
Model + data + Analyzer -> Result
Result + Experiment    -> comparable evidence
FailureCase + Trajectory -> reusable cases for humans and agents
```

The intended workflow is:

1. Get a model: `evalrx.wrap(your_model, tokenizer)` or `evalrx.load("key")`.
2. Discover compatible analyzers from the registry.
3. Run analyzers that match the model's capabilities.
4. Store results as structured findings and artifacts.
5. Use those results to refine cases, hypotheses, and experiments.

Or hand the loop to `AutoDiagnoseLoop` and let it drive steps 2–5 automatically:

```python
from evalrx.eval_agent import AutoDiagnoseLoop, DiagnosisAgent

loop   = AutoDiagnoseLoop(model=my_model, diagnosis_agent=DiagnosisAgent(judge=judge))
report = loop.run(failure_cases)
# → report.resolved, report.final_hypotheses, report.final_results
```

## Documentation Map

- [Quickstart](quickstart.md): runnable examples and common entry points.
- [Exploratory Analysis (M2/M3)](m2_analysis.md): standalone `evalrx
  explore` — descriptive analysis + hypothesis proposal, no code required.
- [Intervention & Verification (M4/M5)](intervention.md): `HypothesisTester`
  verification and `FixAgent`/`SurgeryAgent` tiered repair, loop-internal.
- [Stage Input/Output Reference](stage_io.md): one page listing the exact
  input and output type of every M1–M5 stage.
- [Analyzer Zoo](analyzers.md): reference tables of implemented analyzers and registered models.
- [Architecture](architecture.md): package structure and design contracts.
- [Extending EvalRX](extending.md): how to add analyzers, specs, and backends.
- [Roadmap](roadmap.md): current implementation status and planned surfaces.

## Current Status

EvalRX is currently an alpha package. The core contracts, spec/backend
composition, capability matching, public `wrap()` on-ramp, 26 registered
analyzers, statistics layer, and the full automated diagnosis pipeline are
implemented and covered by 599 unit tests (no GPU required).  VLM forward capture
(image-token mask + spatial layout) is implemented for all models in the spec
registry. Several analyzers are Stage-2 stubs that intentionally raise
`NotImplementedError`; see the [Roadmap](roadmap.md) for details.

Two diagnosis loops are available:

**`AutoDiagnoseLoop`** (M1→M4) ships with production-grade operational
infrastructure: atomic checkpoints with `resume()`, heartbeat liveness,
git-native run versioning (`ExperimentGitManager`), cross-run lesson accumulation
(`EvolutionStore` with 30-day half-life decay), a durable `JsonlStore`, multi-phase
`ExperimentWriter` (blueprint → sequential → hard-validate → exec-fix → tree-search →
review), CLI agent backends (codex, claude_code, opencode, agy …), and a VLM
image-attention analysis rule that closes the M1→M4 loop for vision models.

**`VLDiagnoseLoop`** (M1→M2→M3→M5, M4 post-loop) adds protocol-guided diagnosis:
the user supplies an `ExperimentProtocol` (a natural-language description of what
to investigate), which drives analyzer prioritization in M1 and protocol-consistency
checking in M5.  `StatsAnalysisAgent` (M2) generates an LLM-written evidence chain
alongside the threshold-based findings.  `HypothesisTester` (M5) applies a
statistical fail-rate test and verifies protocol consistency; the loop stops as soon
as a supported, consistent hypothesis is found.

The M1–M5 stage implementations live in `evalrx/eval_agent/stages/`; shared
infrastructure (loop orchestration, logging, hypothesis types, CLI agent) lives at
the `eval_agent/` top level.  The public API at `evalrx.eval_agent` is unchanged.
