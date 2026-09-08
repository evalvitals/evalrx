# Architecture

EvalRX is structured as a small framework substrate plus extension points.
The goal is to make LLM/VLM evaluation feel like using sklearn estimators:
objects are composable, parameters are explicit, capabilities are discoverable,
and outputs follow a common shape.

## Package Layout

```text
evalrx/
+-- core/              # stable contracts and shared substrate
+-- specs.py           # model identity registry
+-- models/            # model composition, runtime backends, compatibility shims
+-- analyzers/         # analyzers grouped by capability (attention, lens, uncertainty, …)
+-- datasets/          # loaders that produce FailureCase / CaseBatch
+-- stats/             # statistical tests: McNemar, e-value, bootstrap CI, Friedman
`-- eval_agent/        # automated diagnosis loop + selective-inference orchestration
```

## Core Contracts

### Two paths to a `Model`

```text
# Public on-ramp — user brings their own already-loaded HF causal LM
evalrx.wrap(model, tokenizer)  ->  HFLocalModel

# Curated path — load a registered checkpoint by key
evalrx.load("qwen2.5-7b-instruct")  ->  HFLocalModel
```

Both paths return the same `HFLocalModel`: capabilities are inferred from the
live model in the `wrap()` case, and read off the spec in the `load()` case.
`wrap()` also applies attention fix-ups automatically (eager mode is required to
capture attention weights; sdpa/flash return `None`).

### `ModelSpec`

`ModelSpec` describes what a model is, not how it is run. It stores identity and
architecture facts such as model family, Hugging Face repo, decoder-layer paths,
vision-token handling, MoE flags, reasoning flags, and attention semantics.

Specs live in `evalrx.specs` and are intentionally torch-free.  When
`wrap()` is used, a minimal spec is inferred at runtime from `model.config` via
`evalrx.models.inference.infer_spec` — no registry entry is required.

**Modality is a set, not a class fork.** A spec declares modalities by the
components it carries — `vision` adds `"image"`, `audio` adds `"audio"`,
`video` adds `"video"` — so an *omni* model (Qwen3-Omni reference) is just a
spec with more than one. Analyzers match on `model.modalities`, and `Inputs`
carries `image` / `audio` / `video` slots beside the prompt:

```python
omni = compose("qwen3-omni-30b-a3b-instruct", "api", rt)
omni.modalities    # frozenset({'text', 'image', 'audio', 'video'})
spec.is_omni       # True;  the audio-only Captioner -> {'text', 'audio'}
```

The thinker (text-emitting multimodal LM) is what failure analysis hooks; the
talker (speech out) is out of scope.

### `Backend`

`Backend` describes how a model is run. Backends declare the capabilities they
can provide and build concrete `Model` objects from a `ModelSpec`.

Current backend categories:

| Backend | Purpose |
|---|---|
| `hf_local` | Local Hugging Face execution with internals capture. |
| `api` | Black-box generation through an injected API function. |
| `vllm_offline` | Planned high-throughput offline inference backend. |

Capabilities belong to the backend because the same model identity can expose
different information under different runtimes — `compose()` negotiates them
up front, before any weights load:

```python
from evalrx import compose, RuntimeConfig
from evalrx.core import Capability

# 1) API / black-box (also covers a `vllm serve` endpoint). Reuse your own engine:
from evalrx.models.backends import call_vision_api_generate_fn
rt = RuntimeConfig(generate_fn=call_vision_api_generate_fn(my_call_vision_api))
api_model = compose("qwen3-vl-8b-instruct", "api", rt)         # caps: GENERATE, TOOL_CALLS

# 2) Local white-box, full internals (forces eager when attention is requested):
wb = compose("qwen3-vl-8b-instruct", "hf_local", want={Capability.ATTENTION})

# 3) Wrong ask fails immediately, before any weights load:
compose("qwen3-vl-8b-instruct", "api", want={Capability.ATTENTION})   # -> CapabilityError
```

Module paths in a spec are *hints*: the white-box backend **discovers** the
real decoder-layer `ModuleList` at load time (`models/_discover.py`) instead
of trusting a hardcoded path — robust across transformers releases and the
doubled-`.model.` / no-`.model` / fused-experts traps.

### `Model`

`Model` is the runtime object analyzers consume. It exposes:

```python
model.generate(inputs, **kwargs) -> str
model.forward(inputs, capture={...}, spec=None) -> Trace
```

`forward` returns a `Trace`, which is the common carrier for captured internals
such as tokens, token ids, attentions, hidden states, logits, and backend-specific
extras.

### `Analyzer`

`Analyzer` is the EvalRX analogue of an sklearn estimator. It has explicit
constructor parameters, declares required capabilities, and returns a `Result`.

```python
analyzer = SomeAnalyzer(**params)
result = analyzer.run(model, data)
```

Analyzers should not depend on concrete model classes. They should depend on
the `Model` protocol, requested captures, and `Trace` fields.

### `Capability`

`Capability` is the matching vocabulary between analyzers and runtimes.

An analyzer declares:

```python
requires = frozenset({Capability.ATTENTION})
```

A backend/model declares:

```python
capabilities = frozenset({Capability.GENERATE, Capability.ATTENTION})
```

The registry can then list compatible analyzers for a model, and `compose(...,
want=...)` can fail early before loading weights.

### `FailureCase`

`FailureCase` is the common data unit. It is meant to hold inputs, labels,
provenance, metadata, and agent trajectories. Datasets should produce
`FailureCase` or `CaseBatch`; analyzers should accept those types in addition to
plain strings where appropriate.

### `Result`

`Result` is the common output object. It separates:

- a short human-readable summary,
- structured `findings` for agents and downstream code,
- optional heavy artifacts such as plots, tensors, or tables.

## Why This Shape Works

The design keeps common failure modes contained:

- Adding a new model family should usually mean adding a `ModelSpec`, not
  rewriting analyzers.
- Adding a new runtime should usually mean implementing a `Backend`, not
  changing model identity.
- Adding a new analysis should usually mean implementing an `Analyzer` that
  requests capabilities, not adding methods to every model.
- Agent tooling can discover what is possible from registries instead of reading
  source code or hard-coding model names.

## eval_agent — automated diagnosis pipeline

`eval_agent/` implements a multi-stage automated diagnosis cycle on top of the
core contracts described above. Three loops are available — the same M1-M4
stages, three different orchestration strategies:

```text
AutoDiagnoseLoop  (legacy.py — four-stage sweep, kept for existing callers)
  M1 · ProbeAgent         detect model kind → run ranked analyzers
  M2 · AnalysisModule     threshold rules + derived metrics → AnalysisReport
  M3 · DiagnosisAgent     judge.generate(report) → Hypothesis list
  M5 · SurgeryAgent       correlate / param-sweep / ExperimentWriter → SUPPORTED/REFUTED
       ↑_________________________________________________________________| (refocus or stop)

VLDiagnoseLoop  (loop.py — current, protocol-guided, stops on verified hypothesis)
  ExperimentProtocol  ← user's NL description of what to investigate
       │
  M1 · ProbeAgent         same as above; protocol.probe_hints() boosts relevant analyzers
  M2 · StatsAnalysisAgent protocol-aware; LLM judge writes conclusion + evidence chain
  M3 · DiagnosisAgent     same as above
  M4 · HypothesisTester   statistical test + protocol consistency check
       │
  loop exits when M4 finds a SUPPORTED + protocol-consistent hypothesis
       │
  M5 · SurgeryAgent       called once post-loop on the best verified hypothesis

AgenticDiagnoseLoop  (agentic/loop.py — judge-decided, same M1-M4 stages)
  A CLI judge picks the next tool (run_probe / run_stats / explore_data /
  cluster_failures / search_probes / propose_hypotheses / test_hypothesis /
  run_surgery / run_fix / stop) each turn from an EvidenceBoard, instead of a
  fixed cycle. The host — not the judge — enforces tool call caps,
  preconditions, and the stop-gate (declaring success requires an
  actually-tested, supported, protocol-consistent hypothesis). See
  [quickstart](quickstart.md#agenticdiagnoseloop--judge-decided-m1-m4-alternative-to-the-fixed-cycle).
```

The agent touches models only through the `Model` protocol and stores all
evidence in a `Store`.

`.run()` on all three loops returns the same class, `AutoDiagnoseReport`
(`loop_reports.py`; `VLDiagnoseReport` is a back-compat alias for the
identical class). Each loop populates only the fields relevant to what it
ran — see [Stage Input/Output Reference](stage_io.md#loop-level-output) for
the full field table.

### Package layout

```text
eval_agent/
├── loop.py               VLDiagnoseLoop (current)
├── legacy.py             AutoDiagnoseLoop, SelfEvolveLoop (kept for existing callers)
├── loop_reports.py       AutoDiagnoseReport (VLDiagnoseReport is an alias — all
│                         three loops return this one unified report class)
├── checkpoint.py         write_checkpoint / write_heartbeat / read_checkpoint
├── run_metadata.py       shared run-provenance + logging-wiring helpers
├── agentic/              AgenticDiagnoseLoop — judge-decided M1-M4 (see below)
├── run_context.py        RunContext, Trial — single owner of a run's output directory
├── run_logger.py         RunLogger — per-cycle JSONL log + artifact sink
├── log_schema.py         published run_log.jsonl JSON Schema
├── hypothesis.py         Hypothesis, HypothesisStatus — shared across M3/M4/M5
├── cli_agent.py          compatibility facade over agent_runtime's CLI providers/judges
├── store.py              Store / InMemoryStore / JsonlStore
├── evolution.py          EvolutionStore — cross-run lesson accumulation
├── orchestrator.py       EvalOrchestrator — thin A/B facade
├── ab_runner.py          ABRunner — A/B execution
├── preregister.py        DataSplit, PreregisteredHypothesis
├── git_manager.py        ExperimentGitManager
├── report.py             DiagnosticReport
└── stages/               ← M1, M3, M4, M5 stage implementations (M2 lives in evalrx.analysis)
    ├── probe.py          M1  StrategyProbe
    ├── probe_agent.py    M1  ProbeAgent
    ├── protocol.py       M1  ExperimentProtocol, ProbingSchema
    ├── diagnosis.py      M3  DiagnosisAgent, DiagnosisResult
    ├── surgery.py        M5  SurgeryAgent, InterventionResult
    ├── experiment_writer.py  M5  ExperimentWriter
    ├── fix_agent.py       M5 (post-loop)  FixAgent, FixCandidate, FixOutcome
    ├── fix_tiers.py       FixTier ladder (L1 prompt → L4 parameter space)
    ├── hypothesis_tester.py  M4  HypothesisTester, HypothesisTestResult
    ├── case_discovery.py  CaseDiscoveryAgent — runs candidates through a model,
    │                      labels PASS/FAIL/UNKNOWN (judge or heuristic scorer)
    ├── probe_candidate_generator.py  VLMProbeCandidateGenerator — ProbeLLM
    │                      Macro/Micro question paraphrasing over a fixed
    │                      VLM seed pool (see probe search, below)
    └── probe_search_agent.py  ProbeSearchAgent — wires ProbeSearch (below) to
                           a real model + judge via CaseDiscoveryAgent +
                           VLMProbeCandidateGenerator

evalrx.agent_runtime/  shared CLI-agent runtime (sandbox, codegen, providers,
                            judges, skills) — used by both eval_agent and analysis;
                            imports neither. See "Package Layout" above.

evalrx.analysis/       M2 lives here now (StatsAnalysisAgent, AnalysisModule,
                            stats tool catalog) — usable standalone, not just from
                            the loop. Also: ExploratoryAnalysisAgent, HypothesisAgent,
                            cluster_failures (failure-mode clustering, with
                            opt-in ProbeLLM-style failure-aware embedding +
                            boundary-aware induction), probe_search.py
                            (ProbeSearch — standalone hierarchical MCTS over
                            injected generator/verifier callables; the eval_agent-
                            layer glue that supplies real callables lives in
                            stages/probe_search_agent.py above, since it needs a
                            live target model + judge).
```

### Stage contracts

| Stage | Module | Class | Key method |
|---|---|---|---|
| M1 | `stages/probe.py` | `StrategyProbe` | `routed_slots(model, data) → set[str]` (what ranking composes over); `select(model, hints, data) → list[str]`; `detect_kind(model) → ModelKind` (display label) |
| M1 | `stages/probe_agent.py` | `ProbeAgent` | `probe(model, data, hint_failure_modes) → dict[str, Result]` |
| M1 | `stages/protocol.py` | `ExperimentProtocol` | `probe_hints() → list[str]` — maps NL description to failure-mode tags |
| M2 | `evalrx.analysis.analysis_module` | `AnalysisModule` | `analyze(results, model_name) → AnalysisReport` |
| M2 | `evalrx.analysis.stats_agent` | `StatsAnalysisAgent` | `analyze(results, model_name, protocol) → StatsAnalysisReport` |
| M3 | `stages/diagnosis.py` | `DiagnosisAgent` | `diagnose(report, prior_cycles) → DiagnosisResult` |
| M5 | `stages/surgery.py` | `SurgeryAgent` | `operate(hypothesis, model, results, data) → InterventionResult` |
| M4 | `stages/hypothesis_tester.py` | `HypothesisTester` | `test(hypotheses, report, data, protocol) → list[HypothesisTestResult]`; `stopping_criteria_met(results) → bool` |

**Pre-M1 (optional): `ProbeSearchAgent.run(model, seed_pool) → ProbeSearchResult`**
— a hierarchical Macro/Micro MCTS (`evalrx.analysis.probe_search.ProbeSearch`)
that synthesizes and evaluates new test cases (VLM QA in v1) rather than
analyzing an already-collected dataset; `result.failure_cases` is a `CaseBatch`
that can seed or extend the data M1 then probes. See
[m2_analysis.md#probe-search--hierarchical-mcts-failure-discovery-vlm](m2_analysis.md#probe-search--hierarchical-mcts-failure-discovery-vlm).

**M1 analyzer selection** happens in two tiers:

- **Tier (a)** `StrategyProbe` ranks analyzers by diagnostic value for the
  detected model kind, guided by the protocol description via an LLM judge:

  | Kind detected | Priority analyzers |
  |---|---|
  | VLM (image/video) | `pope`, `chair`, `attention`, `attention_rollout`, `attention_sink`, `prompt_contrast`, `mm_shap`, `logprob_entropy` |
  | Agent (`TOOL_CALLS`) | `loop_detect`, `ignored_obs`, `first_error_judge`, `counterfactual` |
  | LLM (text-only) | `attention`, `logit_lens`, `token_entropy`, `logprob_entropy`, `attention_sink`, `prompt_contrast`, `cka` |

- **Tier (b)** `ProbeGenerator` / `WhiteboxProbeGenerator` — when no standard
  analyzer covers the failure mode, an LLM or CLI agent writes a bespoke probe
  and runs it in a sandbox.

**M2 `StatsAnalysisAgent`** runs a catalog of statistical tools
(`signal_label_assoc`, `mcnemar_evalue`, `bootstrap_diff`, `friedman`,
`rank_corr`, `single_rate_evalue`) over the analyzer findings, applies e-BH
FDR correction, and produces a `StatsAnalysisReport` with a structured
evidence chain for M3. It can also be used standalone, outside the loop:

```python
from evalrx.analysis import StatsAnalysisAgent

rows = [
    {"case_id": "c0", "label": "fail", "low_img_attn": 1},
    {"case_id": "c1", "label": "pass", "low_img_attn": 0},
]

report = StatsAnalysisAgent().analyze_records(
    rows, id_col="case_id", label_col="label", signal_cols=["low_img_attn"],
)
print(report.conclusion)
print([r.summary for r in report.stats_results])
```

This is a different, confirmatory system from the standalone
`ExploratoryAnalysisAgent`/`HypothesisAgent` pair covered in
[Exploratory Analysis (M2/M3)](m2_analysis.md) — that one is purely
descriptive with no verdict; `StatsAnalysisAgent` is the loop's FDR-controlled
confirmatory stage.

**M4 `HypothesisTester`** asks two questions per hypothesis:

1. *Statistical support* — does the signal group fail at a significantly
   higher rate than the control group? Consumes M2's FDR-corrected stats when
   present; falls back to a clustered-bootstrap `stats.compare` call.
2. *Protocol consistency* — does the hypothesis match what the user
   described? Keyword-based by default; an optional `judge=` runs an LLM
   critic.

All stages are injectable:

```python
# AutoDiagnoseLoop
loop = AutoDiagnoseLoop(
    model=model,
    probe_agent=ProbeAgent(...),
    analysis_module=AnalysisModule(...),
    diagnosis_agent=DiagnosisAgent(judge=judge),
    surgery_agent=SurgeryAgent(judge=judge),
    store=JsonlStore(run_dir / "store"),
    run_logger=RunLogger(run_dir),
    run_dir=run_dir,
)

# VLDiagnoseLoop — ctx is a RunContext; see "RunContext" below for what it owns
ctx = RunContext(run_dir)
protocol = ExperimentProtocol(
    description="VLM suspected to ignore visual tokens ...",
    failure_patterns="spatial confusion, hallucinated objects",
)
loop = VLDiagnoseLoop(
    model=model,
    protocol=protocol,
    stats_agent=StatsAnalysisAgent(judge=model, figure_dir=str(ctx.figures_dir)),
    diagnosis_agent=DiagnosisAgent(judge=model),
    hypothesis_tester=HypothesisTester(judge=model, min_effect=0.05),
    surgery_agent=SurgeryAgent(judge=model, run_context=ctx),
    max_cycles=5,
    run_logger=ctx.logger,
)
report = loop.run(cases)
ctx.write_diagnose_report(report, cases)
fix = loop.run_m5(report, cases)   # M5 called post-loop on best hypothesis
```

### M5 SurgeryAgent — four strategies

M5 selects the first matching strategy:

1. **`verify_fn`** — caller-supplied callable; full custom override.
2. **`analyzer_params`** — re-run named analyzers with modified parameters; surface before/after findings.
3. **`ExperimentWriter`** (when `judge` is provided) — multi-phase LLM/CLI agent writes and executes a targeted Python diagnostic project:
   - Phase 1: Blueprint (YAML spec: file list, pseudocode, dependency order)
   - Phase 2: Sequential file generation with AST-based CodeMem context
   - Phase 3: Hard validation (AST parse; critical issues trigger repair)
   - Phase 4: Exec-fix loop (parse traceback → targeted single-file repair)
   - Phase 5: Tree search (optional; explore multiple candidates, score by metrics)
   - Phase 6: Review dialog (optional; coder-reviewer LLM exchange)
4. **Label correlation** — passive; correlate per-case signals with PASS/FAIL labels.

CLI coding agents (`codex`, `claude_code`, `opencode`, …) can substitute for the
LLM writer in Phase 1+2 via `ExperimentWriterConfig(cli_agent=CliAgentConfig(provider="codex"))`.
The generated code project is executed by `ExperimentSandbox.run_project()`.  Case images
are saved as JPEG files in the sandbox workdir so the agent can load them.

### Sandbox

`ExperimentSandbox` runs Python code safely in a subprocess.

```python
sandbox = ExperimentSandbox(workdir=Path("tmp/"))

# Single-file execution
result = sandbox.run("print('verdict: 1.0')")

# Multi-file project execution (M5 ExperimentWriter path)
result = sandbox.run_project(
    project_dir,
    entry_point="main.py",
    timeout_sec=60,
)
```

Key properties:
- **Path traversal protection**: `entry_point` is validated syntactically before copy and
  after copy (symlink-resolved) to prevent directory escape.
- **Harness injection**: `experiment_harness.py` is copied into every project directory before
  execution.  It provides time-budget management, metric reporting with NaN guards, and
  `results.json` persistence.  Projects cannot overwrite it.
- **Numbered project dirs**: concurrent calls produce `_project_1/`, `_project_2/`, etc. (thread-safe).
- **Cleanup policy**: project/script directories are deleted only on success (`rc==0` and
  not timed out), preserving failure artefacts for debugging.

`SandboxProtocol` is a structural type allowing transparent substitution of subprocess,
Docker, SSH, or other backends.  `create_sandbox(SandboxFactoryConfig, workdir)` selects
the backend from a `mode` string (`"subprocess"` default, `"docker"` with graceful fallback).

### Run-directory infrastructure

Pass `run_dir` to `AutoDiagnoseLoop` to enable the full operational stack:

```python
loop = AutoDiagnoseLoop(
    model=model,
    diagnosis_agent=DiagnosisAgent(judge=judge),
    run_dir=Path("runs/my_experiment"),   # enables all infrastructure below
)
report = loop.run(cases)

# Resume a previously interrupted run
report = AutoDiagnoseLoop.resume(Path("runs/my_experiment"), model=model, data=cases)
```

When `run_dir` is set, `AutoDiagnoseLoop` creates:

```text
run_dir/
├── checkpoint.json          ← atomic (temp+rename); last_completed_cycle + run_id
├── heartbeat.json           ← pid + last_cycle + timestamp (liveness signal)
├── artifacts/<run_id>/      ← per-run staging area
└── evolution/
    └── lessons.jsonl        ← cross-run lesson accumulation (append-only)
```

**Checkpointing and resume**: after every completed cycle, `checkpoint.json` is written
atomically.  `AutoDiagnoseLoop.resume(run_dir, model, data)` reads the checkpoint and
restarts from `last_completed_cycle + 1`, skipping already-completed work.

**Git integration**: when `run_dir` is inside a git repository, `ExperimentGitManager`
auto-detects it and:
- Creates branch `eval/{run_id}` at the start of the run.
- Commits all staged files with hypothesis statuses on a resolved run.
- Calls `git reset --hard HEAD` on an unresolved run (non-destructive: only uncommitted
  changes are discarded).

### EvolutionStore — cross-run lesson accumulation

`EvolutionStore` accumulates lessons from every diagnosis run in an append-only JSONL
file.  Lessons are weighted by a 30-day half-life exponential decay so recent findings
rank higher.

```python
store = EvolutionStore(Path("runs/my_experiment/evolution"))

# Lessons are appended automatically when run_dir is set.
# Read them back for prompt injection:
overlay = store.build_overlay("surgery", max_lessons=5)
# → "## Lessons from Prior Diagnosis Runs\n1. [WARN] ..."
```

`extract_lessons(report)` auto-derives lessons from an `AutoDiagnoseReport`:
- INCONCLUSIVE hypotheses → `surgery / warning`
- Loop exhausted without resolution → `diagnosis / warning`
- HIGH/CRITICAL analysis severity with no resolution → `analysis / info`

### Persistent store — `JsonlStore`

`JsonlStore` is a durable implementation of the `Store` interface backed by three JSONL files
(`hypotheses.jsonl`, `results.jsonl`, `cases.jsonl`).  Hypotheses survive process restarts
and are fully reconstructed via `hypothesis_to_dict` / `hypothesis_from_dict`.

```python
store = JsonlStore(Path("runs/store"))
loop = AutoDiagnoseLoop(model=model, store=store, ...)
```

### Analysis rules — VLM image-attention

`AnalysisModule` includes a VLM-specific derived metric for the `attention` analyzer.
Before applying threshold rules, it sums the attention weights of image-related tokens
(`<|image_pad|>`, `<|vision_start|>`, `<|vision_end|>`, …) from `top_attended_tokens`
and exposes `image_token_attention_ratio`.  A ratio below 0.05 fires a `medium`-severity
finding:

```
[MEDIUM] attention.image_token_attention_ratio=0.012 < 0.05:
VLM nearly ignores image tokens — attention dominated by text/structural tokens
```

This finding propagates to M3 DiagnosisAgent and M5 SurgeryAgent, closing the loop from
attention measurement to codex-generated diagnostic code.

### Result image overlays

A bare heatmap (`spatial_map`, `fail_mean_map`, …) carries no spatial reference
to the photo it was computed from, so neither a human nor a multimodal judge
can tell whether a highlighted patch corresponds to anything sensible.
`RelativeAttentionResult` (and any `Result` subclass that wants the same
treatment) exposes:

```python
result.overlay(key="spatial_map", alpha=0.6, cmap="jet")  # -> PIL.Image | None
result.save_overlay(key, path)                             # -> bool
result.image_overlays(fig_dir, stem_prefix)                 # -> list[Path]
```

`overlay()` alpha-blends the map onto its representative case image
(CAM-style, intensity-weighted so the background stays visible); it resolves
lazy `Inputs.image` paths/URLs the same way the model's forward pass did
(`transformers.image_utils.load_image`), so the overlay matches what was
actually fed to the model. `image_overlays()` is a duck-typed hook: `RunLogger`
calls it on any `Result` that defines it and saves the PNGs into `figures/`
alongside the bare heatmaps, so overlays flow into the same artifact list a
multimodal judge already receives — no per-analyzer wiring required.

### RunContext — single owner of a run's output directory

`RunContext` replaces the old per-example pattern of hand-written report
files, `RunLogger` buried under a `logs/` subdir, hand-built figure-dir paths,
and M5 sandboxes living in ephemeral temp dirs deleted on success.  One
`RunContext` owns the whole run root and hands every producer its
subdirectory.

`RunContext`'s **default** is now `logger_version="v2"`
(`evalrx/eval_agent/RUN_LOGGER_V2.md`): one `run.json` plus one
`M1/log.json`..`M5/log.json` per stage, no `run_log.jsonl`, no persisted
`prompts/`/`experiments/`/`tools/`/`workspace/`/`fixes/` (generated text/code
is captured inline into the relevant stage's JSON instead), no
`manifest.json`/`README.txt`. The rest of this section — the layout below,
`run_log.jsonl`'s event schema, and the trial-folder mechanics — describes the
**V1** layout (`logger_version="v1"`), which still exists, stays readable for
old runs, and is what to pass explicitly when that's what's wanted:

```text
<root>/
├── manifest.json     run config + index of every produced file
├── run_log.jsonl     structured event stream (RunLogger)
├── README.txt        auto-generated file guide (from manifest)
├── report/           human deliverables (summary.md, hypotheses.json, m4_results.json, …)
├── figures/          M1 heatmaps (+ overlay PNGs) and M2 effect plots
├── artifacts/        M1 heavy numeric data (.npy / .json)
├── prompts/          judge prompt / response
├── experiments/      one self-contained folder per M5 ExperimentWriter trial
├── tools/            synthesised probe / stats tool code (M1/M2, run-global)
├── workspace/        sandbox working dirs outside any trial
└── fixes/            one self-contained folder per FixAgent repair attempt
```

Each line in `run_log.jsonl` carries `event`, `cycle`, `ts` (ISO-8601), a
`schema_version` (int, bumped only when an existing event's fields are
renamed/removed/change meaning — additive fields don't bump it, so a
downstream parser can detect breaking changes without guessing from
`evalrx_version`), and stage-specific fields (findings, narrative, raw LLM
output, intervention status …). The first `run_start` event records run
provenance — `model`, `judge`, `git_commit` (falls back to the
`EVALRX_GIT_COMMIT` env var when the `git` CLI is unavailable, e.g. inside
the example Docker images), `data_fingerprint` (an order-independent hash of
the case batch, so two runs can be confirmed to use the same data) and
`label_distribution` (the base PASS/FAIL/UNKNOWN counts the diagnosis is
conditioned on). The `analysis` event's stats fields
(`stats_tool_results`, `stats_results`, `stats_plan`, `corrected_rejections`)
are externalized to `artifacts/` the same way `probe`'s `artifact_paths` are
once their JSON size exceeds 4 KB — the JSONL line then carries
`{"path", "n_items", "bytes"}` instead of the raw value. Standard shell tools
work directly on it:

```bash
tail -f run_dir/run_log.jsonl                           # live stream
jq 'select(.event=="diagnosis")' run_log.jsonl          # all judge outputs
jq 'select(.event=="probe") | .findings' run_log.jsonl  # M1 findings
jq 'select(.event=="surgery") | .evidence' run_log.jsonl
```

The event format is a **published JSON Schema** (Draft 2020-12), shipped as
package data at `evalrx/eval_agent/run_log.schema.json` and built from
`evalrx/eval_agent/log_schema.py` — so downstream parsers (in any language)
can validate `run_log.jsonl` instead of guessing field shapes. It's permissive
by design: it pins the common envelope (`event`, `schema_version`, `ts`,
`trace_id`), the per-event required fields and core types, but allows additive
fields (matching the `schema_version` rule above).

```python
from evalrx.eval_agent import iter_log_errors, validate_event

for line_no, msg in iter_log_errors("run_dir/run_log.jsonl"):  # empty == conforms
    print(line_no, msg)
```

Set `EVALRX_VALIDATE_LOG=1` to have `RunLogger` self-check every event it
writes against the schema and warn (never raise) on a violation — a CI/dev aid
to catch a producer drifting from the contract. Both paths need the optional
`jsonschema` dependency (`pip install evalrx[dev]`).

```python
from evalrx.eval_agent import RunContext, VLDiagnoseLoop

with RunContext("examples/foo/outputs", verbose=True, logger_version="v1") as ctx:
    stats_agent = StatsAnalysisAgent(judge=judge, figure_dir=str(ctx.figures_dir))
    loop = VLDiagnoseLoop(..., run_logger=ctx.logger)
    report = loop.run(cases)
    ctx.write_diagnose_report(report, cases, discovery=discovery_rows)
# manifest.json + README.txt written, logger closed on exit. (V2's finalize()
# just closes run.json/M*/log.json, already flushed incrementally throughout.)
```

`write_diagnose_report(report, cases, discovery=...)` writes the standard
`report/` deliverables from an `AutoDiagnoseReport` (all three loops return
this one class — see "Stage contracts" above), replacing the
`_write_report_artifacts` boilerplate previously copy-pasted into every
example.

**Per-trial folders** (`ctx.new_trial("fixes" | "experiments", label)`):
each fix candidate or M5 experiment gets its own numbered folder —
`fixes/03_widen_crop/` — holding the generated code, the sandbox it ran in,
judge prompt/output, `record.md`, and `result.json`, instead of scattering
those across `tools/` / `workspace/` / `fixes/` and re-correlating them by
filename slug.  A `Trial`'s folder (and its `workspace/`) is created lazily
on first write, so a candidate discarded before producing anything (e.g. a
deduped proposal) leaves no empty folder — a gap in the numbering honestly
means "proposed, then discarded," not a missing record.  `ctx.new_workdir(label)`
is the non-trial equivalent for sandboxes that don't belong to a numbered
attempt (e.g. M1/M2 tool codegen). Under V2 there are no persisted trial
folders at all — a trial's generated code/output is inlined into its stage's
JSON, and the sandbox it ran in lives under an ephemeral `ctx.runtime_root`
tempdir that's deleted at `finalize()`, not `fixes/`/`experiments/`.

**Not the same as `run_dir`** in the "Run-directory infrastructure" section
above: `AutoDiagnoseLoop(run_dir=...)` owns *resume* mechanics (checkpoint,
heartbeat, evolution, git) and is orthogonal — a run can use either, both, or
neither. `RunContext` owns *output* (report/figures/artifacts/fixes/manifest).
`AutoDiagnoseLoop`'s own `run_dir` infra was deliberately left untouched when
`RunContext` was introduced.

### FixAgent — tiered post-loop repair (M5)

`FixAgent` (`stages/fix_agent.py`) is a second M5 path, invoked via
`loop.run_fix(report, data)` after `loop.run()` — distinct from `SurgeryAgent`
above, which verifies *why* something fails; `FixAgent` proposes and validates
candidate *fixes*. The allowed intervention space is an **input** (`FixTier`,
default `L2_SCAFFOLD`); there is no automatic escalation:

```text
L1   input space        prompt rewrites, instruction strategies
L2   scaffold space     agent-designed pipelines around the unchanged model
                         (multi-call, external tools, aggregation) — sandboxed,
                         bridged model access; labels never reach the code
L3a  internals (read)   read attention/logits to guide scaffold actions
L3b  internals (write)  modify the forward pass (attention reweighting,
                         sink suppression, activation steering)
L4   parameter space    fine-tune recipe — v1 executes LoRA on the language
                         model, trained on a caller-supplied diagnosis-only
                         pool; other recipe shapes (sft/full, vision_encoder/
                         projector) are recorded, not yet executed
```

Every candidate is validated against the unmodified baseline with paired
McNemar + e-value (never a bare p-value); a *fixed* verdict means the
candidate repairs significantly more cases than it breaks.  `max_repair_rounds`
(default 1) lets the judge/coder retry with *different* strategies within the
same tier after a round where nothing validates — never re-proposing an
identical candidate (`FixAgent._signature`), never raising the tier itself.
`loop.run_fix(..., auto_escalate=True)` steps the ceiling tier L2 → L3a → L3b
automatically, feeding each round the full history of prior failures.  When no
candidate validates, the outcome carries a `recommendation` (e.g. "raise to
L3a") for the caller to act on. Each candidate's code/sandbox/record lives
under its own `fixes/NN_label/` folder when a `RunContext` is attached (see
above).

## Public Surface Guidance

The intended stable public entry points are:

```python
# Model construction — two paths, same result object
evalrx.wrap(model, tokenizer, *, want=(), **runtime)  # bring your own model
evalrx.load(key, *, backend, want, checkpoint, **runtime)  # curated checkpoints

# Config-driven run
evalrx.run(config, data)
evalrx.load_config(path)

# Registry / discovery
evalrx.list_specs()
evalrx.get_spec(key)
evalrx.registry

# Core types
evalrx.Capability
evalrx.FailureCase
evalrx.Result

# Automated diagnosis — AutoDiagnoseLoop (legacy M1→M5 sweep)
from evalrx.eval_agent import AutoDiagnoseLoop, DiagnosisAgent, RunLogger, StrategyProbe, SurgeryAgent

# Protocol-guided diagnosis — VLDiagnoseLoop (M1→M2→M3→M4, M5 post-loop)
from evalrx.eval_agent import (
    VLDiagnoseLoop, ExperimentProtocol,
    StatsAnalysisAgent, HypothesisTester,
)

# Run output ownership + post-loop tiered repair
from evalrx.eval_agent import RunContext, FixAgent, FixTier
```

Lower-level implementation details (`compose`, `HFLocalModel`, `infer_spec`,
`Backend`, `ModelSpec`) should remain under their package namespaces unless they
are meant to become long-term extension APIs.
