# Quickstart

This page shows the common ways to run EvalVitals.

## Bring Your Own Model

If you already have a loaded Hugging Face causal LM, wrap it — no registry key
needed. The wrapped model is the same object `evalvitals.load(...)` returns, so
every capability-compatible analyzer works on it.

```python
import evalvitals
from transformers import AutoModelForCausalLM, AutoTokenizer
from evalvitals.analyzers.lens.logit_lens import LogitLensAnalyzer

model = AutoModelForCausalLM.from_pretrained("my-org/my-llama")
tokenizer = AutoTokenizer.from_pretrained("my-org/my-llama")

wrapped = evalvitals.wrap(model, tokenizer)
result = LogitLensAnalyzer().run(wrapped, "The capital of France is")
print(result.summary())
```

Capabilities (attention, hidden states, logits, …) are inferred from the live
model. Attention capture needs eager attention; `wrap` switches the model to it
when it can, otherwise load with `attn_implementation="eager"`. White-box capture
currently supports text decoder-only models (VLM capture is Stage 2).

## One-Liner Model Load

```python
import evalvitals
from evalvitals.analyzers.attention.summary import AttentionAnalyzer

model = evalvitals.load("qwen2.5-7b-instruct")
result = AttentionAnalyzer(layer=-1, top_k=5).run(
    model,
    "The Eiffel Tower is in",
)

print(result.summary())
print(result.findings)
```

By default, `evalvitals.load` uses the `hf_local` backend unless the spec is
API-only.

## Config-Driven Run

```yaml
model: qwen2.5-7b-instruct
analysis: attention
analysis_kwargs:
  layer: -1
  top_k: 5
```

```python
from evalvitals import load_config, run

config = load_config("configs/qwen_attention.yaml")
result = run(config, "The Eiffel Tower is in")
```

## Explicit Backend Selection

Use `compose` when you want to control the runtime and negotiate capabilities
before model weights load.

```python
from evalvitals import Capability
from evalvitals.models import compose

model = compose(
    "qwen2.5-7b-instruct",
    "hf_local",
    want={Capability.ATTENTION},
)
```

If the selected backend cannot provide the requested capability, EvalVitals
raises a `CapabilityError` before constructing the model.

## Discovery

```python
import evalvitals

print(evalvitals.list_specs())
print(evalvitals.registry.analyzers.list())
print(evalvitals.registry.analyzers.names_compatible_with(model))
```

This is the same discovery surface intended for an automated evaluation agent.

## Agent — Tool-Calling Loop, Any Backend

`Agent(wraps=handle)` is **backend-agnostic**: it needs only `GENERATE` +
`TOOL_CALLS` (checked up front), never internals — so the *same* loop drives
an API model and a local model. The single thing that varies is the
`ToolCallCodec` (auto-selected): OpenAI-native structured calls for the API,
Hermes-style `<tool_call>{…}</tool_call>` text parsing for local templates.
Tool execution goes through a pluggable `ToolExecutor` (swap in your
`APIToolHandler`).

```python
from evalvitals import Agent, Tool, compose, RuntimeConfig
from evalvitals.models.backends import call_vision_api_chat_fn

search = Tool(name="search", description="web search",
              parameters={"type": "object", "properties": {"q": {"type": "string"}}},
              fn=my_search)

# API agent (reuse your engine): native tool-calls, OpenAI codec
api = compose("qwen3-8b", "api", RuntimeConfig(chat_fn=call_vision_api_chat_fn(call_vision_api)))
traj = Agent(api, tools=[search]).run("who won the 2022 world cup?")   # -> Trajectory

# Local agent: SAME Agent; tools rendered via the model's chat template, Qwen codec
local = compose("qwen3-8b", "hf_local")        # TOOL_CALLS granted only if spec.tool_calling
traj = Agent(local, tools=[search]).run("...")  # -> Trajectory (steps: USER→ACTOR→TOOL→…)
```

`TOOL_CALLS` is a **conditional** capability for local models: the backend
provides the channel, but it's granted only when the model's chat template
renders tools (`spec.tool_calling`, verified against the template at load).
So `compose(non_tool_model, "hf_local", want={TOOL_CALLS})` fails up front.

**Multimodal agents.** The same loop runs vision cases end-to-end: give the
case an image and it enters the first user message; a tool that returns a
`ToolResult` with `images` gets them re-injected as a new message, so the
model can *look at* what its tool produced. `evalvitals.models.tools` ships
subject-side visual tools (deterministic, model-agnostic schemas):

```python
from evalvitals import Capability, compose, RuntimeConfig
from evalvitals.core.case import FailureCase, Inputs
from evalvitals.models.agent import Agent
from evalvitals.models.tools import zoom_in_tool

vlm = compose("qwen3-vl-2b-instruct", "hf_local",
              runtime=RuntimeConfig(device="cuda:0"),
              want={Capability.GENERATE, Capability.TOOL_CALLS})
agent = Agent(vlm, tools=[zoom_in_tool(image, save_dir="outputs/images")],
              system="Zoom into the relevant region before answering.")
traj = agent.run(FailureCase(inputs=Inputs(prompt="What does the sign say?", image=image)))
json.dump(traj.to_dict(), open("trajectory.json", "w"))   # serializable end-to-end
```

Runnable version: `examples/agent_demos/visual_zoom_agent/run.py`.

**Scale path.** For batch probing, serve the open checkpoint with vLLM and use
the built-in OpenAI-compatible client — no dependency injection needed:

```bash
vllm serve Qwen/Qwen3-VL-2B-Instruct --port 8901 \
    --enable-auto-tool-choice --tool-call-parser hermes
```

```python
from evalvitals.models.agent import run_batch
from evalvitals.models.backends import openai_runtime

vlm = compose("qwen3-vl-2b-instruct", "api",
              runtime=openai_runtime(base_url="http://localhost:8901/v1"))
trajs = run_batch(vlm, cases, tools_factory=lambda c: [zoom_in_tool(c.inputs.image)],
                  system=SYSTEM, concurrency=8)   # threads; error-stub per failed case
```

Every turn records latency and token usage on the step's ``span``; the
trajectory flattener (`evalvitals.analysis.trajectory_records`) turns finished
runs into `records.json` rows that `evalvitals explore` consumes directly.
Closed models: the same `openai_runtime` against an OpenAI endpoint, or
`GeminiModel` (native function calling) for Gemini.

## Standalone M2 Explore

If you already have result logs and want M2 to analyze them without writing
analysis code, run a single-shot exploration:

```bash
evalvitals explore /path/to/results \
  --backend antigravity \
  -q "Which failure patterns distinguish wrong answers from correct ones?" \
  --out evalvitals_explore_output \
  --dashboard          # optional
```

The run writes the generated code, stdout/stderr, a structured exploratory
report (`exploratory_report.json`), and rendered charts under
`evalvitals_explore_output/figures/` + `tables/`.

Add `--holdout-frac 0.4 --holdout-confirm` for a held-out design: the split
is carved off BEFORE exploration (outcome-stratified, deterministic), the
explorer is told to freeze threshold-explicit recipes, and after M3 the
held-out rows re-test every recipe verbatim (e-BH, `split_label="held_out"`)
while an LLM judge grades each hypothesis — `confirm_report.json` lands next
to the report and fills the dashboard's *Held-out Verdicts* tab.

**Agent-trajectory records.** Rows produced by
`evalvitals.analysis.trajectory_records` flow through explore unchanged; use
`trajectory_records.AGENT_QUESTION_TEMPLATE` as the question (it declares the
column-family semantics — causal `shap_outcome_*`, stability `success_rate`,
judge-assigned `failure_mode`, cost columns — and steers hypotheses toward
the black-box fix surface). One held-out caveat is unique to agents: columns
from the *interventional* probes (`reliability_probe`, `tool_shap`) must be
**re-run on the held-out cases** (k repetitions / subset ablations there) —
reusing exploration-set values would test nothing. Budget for it: held-out
confirmation of these columns costs the same per case as exploration did.

Open the saved output as a dashboard:

```bash
pip install -e ".[dashboard]"
evalvitals dashboard evalvitals_explore_output
```

Or serve the browser-first workbench — upload a .zip of tabular data and/or
media to start a persistent analysis thread, with existing result directories
attached read-only in the same sidebar. The page streams durable stage events,
shows M2 before M3 has finished, and supports follow-up questions over the same
normalized bundle. Every result renders with one fixed five-tab layout (problem
setting, exploratory analysis, hypotheses, held-out verdicts, fix); stages a
run never reached grey out as "not available":

```bash
evalvitals web my_runs --port 8500 --attach evalvitals_explore_output
```

See [Exploratory Analysis (M2/M3)](m2_analysis.md) for the full standalone
explore + hypothesis-generation workflow.

### Case study on a paper-method bench run

Point the same command at an `examples/` bench run — a directory holding
`outputs/<name>.json` beside the `data/` manifest the runner read — and the
dashboard opens a **case book** instead of the explore layout:

```bash
evalvitals dashboard examples/agent_loop/qwen2_audio_tcd_mmau
```

Three tabs: *Run Overview* (splits, hypothesis, where the baseline fails),
*Case Study* and *Repair Methods*. The Case Study tab plays each case's actual
stimulus — an `<audio>` player for MMAU clips, the image for the VLM slices —
next to the question and its options, so a human can answer the item before
anything is revealed (blind mode is on by default, and the outcome filter is
disabled while it is, so the filter cannot leak the answer). Unblinding shows
the correct answer, the model's baseline answer and, per repair candidate, what
that candidate did to *this* case: repaired / broke / no change. Answers you
lock in are scored against the model on the same cases and stay in the browser
session — nothing is written back to the report.

The Repair Methods tab is the other half of a case study: for every candidate
in the sweep it shows the mechanism (TCD's blurred-waveform contrast, an L1
prompt template, an L2 self-refine scaffold …), its exact configuration or
prompt text, its paired McNemar / e-value verdict, and a button per flipped
case that jumps straight to that case in the Case Study tab.

## Run a Codebase, Then Explore

If you don't have result logs yet — only an existing evaluation/inference
codebase — `run_codebase` bridges the gap: a CLI coding agent runs the
codebase for you inside an isolated copy (your original directory is never
modified), harvests the per-case results it produces, and hands them
straight to the same M2/M3 explore pipeline above.

```bash
evalvitals run-codebase ./my_eval_repo \
  --backend claude_code \
  -q "Where does the model fail and why?" \
  --out evalvitals_run_codebase_output \
  --dashboard          # optional
```

Or from Python:

```python
import evalvitals

result = evalvitals.run_codebase("./my_eval_repo", out="run_output")
print(result.ran_ok, len(result.records))
print(result.explore.report.observations)
```

**Output contract:** the run agent is instructed to write `records.json` (a
JSON array or JSON-Lines file, one row per evaluation case) in the codebase's
working copy, each row carrying a `label` (e.g. `PASS`/`FAIL`) plus the
case's input/prediction/target fields. If the codebase already writes
per-case outputs, the agent converts them into this format rather than
re-running everything. One repair turn is attempted if nothing usable was
produced (`max_attempts`, default 2) before giving up with `ran_ok=False`
and an `error` explaining why.

Whatever environment the codebase needs — API keys, GPU, installed
dependencies — is the user's responsibility; the agent runs with your local
environment and does not provision infrastructure. Pass `analyze=False` /
`--no-explore` to only run + harvest without the M2/M3 step.

## Convenience Shim

Models expose `call_<analysis>` methods dynamically through the analyzer
registry:

```python
result = model.call_attention(
    "The Eiffel Tower is in",
    layer=-1,
    top_k=5,
)
```

This is convenience syntax for:

```python
from evalvitals.analyzers.attention.summary import AttentionAnalyzer

result = AttentionAnalyzer(layer=-1, top_k=5).run(model, data)
```

Prefer direct analyzer construction in reusable code because it makes parameters
and dependencies explicit.

---

## Hallucination Analysis (POPE + CHAIR)

Both analyzers are black-box (`GENERATE`-only) and work with any vision-capable
model, including API endpoints.

**POPE** — yes/no object-presence probes; reports accuracy, precision, recall, F1.
Each case needs `metadata["pope_label"]` = `"yes"` or `"no"`.

> Paper: Li et al., EMNLP 2023 — <https://arxiv.org/abs/2305.10355>  
> Code: <https://github.com/AoiDragon/POPE>

```python
from evalvitals.analyzers.hallucination.pope import POPEAnalyzer
from evalvitals.core.case import CaseBatch, FailureCase, Inputs

cases = CaseBatch([
    FailureCase(inputs=Inputs(prompt="Is there a cat? Answer yes or no.", image=img),
                metadata={"pope_label": "yes"}),
    FailureCase(inputs=Inputs(prompt="Is there a plane? Answer yes or no.", image=img),
                metadata={"pope_label": "no"}),
])
result = POPEAnalyzer().run(model, cases)
# result.findings → {"accuracy": 0.83, "f1": 0.86, "yes_rate": 0.50, ...}
print(result.summary())
```

**CHAIR** — caption hallucination rate vs a fixed object vocabulary.
Each case needs `metadata["gt_objects"]` = list of gold object strings.

> Paper: Rohrbach et al., EMNLP 2018 — <https://arxiv.org/abs/1809.02156>

```python
from evalvitals.analyzers.hallucination.chair import CHAIRAnalyzer

COCO_VOCAB = ["cat", "dog", "car", "chair", ...]  # 80 COCO categories
cases = CaseBatch([
    FailureCase(inputs=Inputs(prompt="Describe the image.", image=img),
                metadata={"gt_objects": ["cat", "chair"]}),
])
result = CHAIRAnalyzer(object_vocab=COCO_VOCAB).run(model, cases)
# result.findings → {"chair_i": 0.25, "chair_s": 0.50, "n": 2}
# chair_i: mean per-caption hallucination rate; chair_s: fraction of captions with ≥1 hallucination
```

Full runnable example: `examples/analyzer_demos/hallucination/` (launch with `docker compose up`).

---

## Shapley Attribution — MM-SHAP + VL-SHAP

Both analyzers require `LOGPROBS` capability and use Monte-Carlo Shapley sampling.

**MM-SHAP** — decomposes model reliance between text tokens and the image.
`mm_score` near 1.0 ⇒ image-driven; near 0.0 ⇒ text-driven. Measures *reliance*,
not correctness.

> Paper: Parcalabescu & Frank, ACL 2022 — <https://arxiv.org/abs/2212.08158>  
> Code: <https://github.com/coastalcph/mm-shap>

```python
from evalvitals.analyzers.perturbation.mm_shap import MMShapAnalyzer
from evalvitals.core.case import CaseBatch, FailureCase, Inputs

case = FailureCase(inputs=Inputs(prompt="What color is the car?", image=img))
result = MMShapAnalyzer(n_samples=64, top_k=5).run(model, CaseBatch([case]))
# result.findings → {
#   "mm_score": 0.62,              # 0=text-reliant, 1=image-reliant
#   "text_contribution": 0.38,
#   "image_contribution": 0.62,
#   "top_text_tokens": [{"token": "color", "shapley": 0.12}, ...]
# }
```

**VL-SHAP** — spatial Shapley attribution over a grid of image regions.
Ranks which regions most influenced the model's output logprob.

> Based on: Lundberg & Lee, NeurIPS 2017 — <https://arxiv.org/abs/1705.07874>  
> Applied via MM-SHAP framework (Parcalabescu & Frank, ACL 2022)

```python
from evalvitals.analyzers.perturbation.vl_shap import VLShapAnalyzer

result = VLShapAnalyzer(n_regions=16, n_samples=64, top_k=3).run(model, CaseBatch([case]))
# result.findings → {
#   "grid_side": 4,           # 4×4 grid
#   "top_regions": [{"region": 5, "shapley": 0.31}, ...],
#   "total_abs_attribution": 1.84
# }
```

Full runnable example: `examples/analyzer_demos/mm_shap/` (launch with `docker compose up`).

---

## Logprob Entropy (black-box uncertainty)

`LogprobEntropyAnalyzer` computes sequence perplexity and per-token predictive
entropy from output-token logprobs — works on any API model that returns
`top_logprobs` (e.g. OpenAI).  No white-box access needed.

> Predictive entropy: Gal & Ghahramani, ICML 2016 — <https://arxiv.org/abs/1506.02142>  
> LLM self-knowledge: Kadavath et al. 2022 — <https://arxiv.org/abs/2207.05221>

```python
from evalvitals.analyzers.uncertainty.logprob_entropy import LogprobEntropyAnalyzer
from evalvitals.core.case import CaseBatch, FailureCase, Inputs

case = FailureCase(inputs=Inputs(prompt="The capital of France is"))
result = LogprobEntropyAnalyzer().run(model, CaseBatch([case]))
# result.findings → {
#   "n_tokens": 3,
#   "perplexity": 1.12,        # low ⇒ model is confident
#   "mean_logprob": -0.11,
#   "mean_top_entropy": 0.08,  # high ⇒ broad uncertainty at that step
#   "min_token_logprob": -0.31
# }
```

Wire an OpenAI endpoint as the `logprobs_fn`:

```python
from evalvitals.models.backends.api import parse_openai_logprobs
from evalvitals.models.backends.base import RuntimeConfig

def logprobs_fn(prompt, *, model="gpt-4o-mini", max_new_tokens=40, top_k=5, **_):
    resp = client.chat.completions.create(
        model=model, messages=[{"role": "user", "content": prompt}],
        max_tokens=max_new_tokens, logprobs=True, top_logprobs=top_k,
    )
    return parse_openai_logprobs(resp.choices[0].logprobs.content)

rt = RuntimeConfig(generate_fn=..., logprobs_fn=logprobs_fn)
```

Full runnable example: `examples/analyzer_demos/logprob_entropy/` (launch with `docker compose up`).

---

## Statistical Comparison — `stats.compare`

`compare` is the single entry point for pairwise A/B comparison. It **never**
returns a bare p-value: it always gives effect size + clustered bootstrap CI +
anytime-valid e-value + corrected reject decision.

> McNemar: McNemar (1947) <https://doi.org/10.1007/BF02295996>  
> E-values: Grünwald et al. (2022) <https://arxiv.org/abs/1906.07801>  
> e-BH FDR: Wang & Ramdas (2022) <https://arxiv.org/abs/2009.02824>

```python
from evalvitals.stats import compare

# success_a / success_b: list of bool (one per example, same order)
r = compare(success_a, success_b, paired=True, alpha=0.05,
            min_effect=0.03, cluster_by=task_ids)

print(r.summary())
# [mcnemar + e-value] effect=+0.12 (B>A) CI=[+0.04, +0.20] e=18.4 → REJECT H0

print(r.effect)      # float: mean(B) - mean(A)
print(r.ci)          # (lo, hi) 95% clustered bootstrap CI
print(r.e_value)     # float: anytime-valid evidence (reject when e ≥ 1/alpha)
print(r.reject)      # bool: corrected decision
print(r.underpowered)  # bool: True when CI width > 2 × min_effect
```

For **3+ strategies** use `compare_multiple` (Friedman omnibus + Nemenyi post-hoc):

> Friedman + Nemenyi: Demšar (2006) <https://jmlr.org/papers/v7/demsarar06a.html>

```python
from evalvitals.stats import compare_multiple

mr = compare_multiple({"A": success_a, "B": success_b, "C": success_c}, alpha=0.05)
print(mr.reject_global)      # bool: at least one strategy differs
print(mr.avg_ranks)          # {"A": 2.1, "B": 1.7, "C": 2.2}
print(mr.significant_pairs)  # [("A", "B"), ...] — pairs that pass Nemenyi CD
```

---

## VLDiagnoseLoop — Automated Failure Attribution (Current)

Two diagnosis loops are available. `VLDiagnoseLoop` is the current
architecture for VL and LLM tasks (`AutoDiagnoseLoop`, below, is kept for
backward compatibility):

```text
M1  ProbeAgent         protocol-guided analyzer selection + execute
M2  StatsAnalysisAgent stats tools + e-BH FDR correction + LLM evidence chain
M3  DiagnosisAgent     "AI scientist" hypothesis generation
M5  HypothesisTester   stats test + protocol consistency check
     ↑___________________________________|
     stop when M5 finds a verified, protocol-consistent hypothesis
```

M4 (`SurgeryAgent`) runs **after** the loop via `loop.run_m4()` to propose
(Plan A) or execute (Plan B) a targeted fix for the best verified hypothesis.
See [Architecture](architecture.md#eval_agent-automated-diagnosis-pipeline)
for the stage contracts, M1's two-tier analyzer selection, and what M2/M5 ask.

```python
from evalvitals import compose
from evalvitals.core.capability import Capability
from evalvitals.eval_agent import VLDiagnoseLoop, AgyModel, RunLogger
from evalvitals.eval_agent.stages.protocol import ExperimentProtocol
from evalvitals.eval_agent.stages.probe_agent import ProbeAgent
from evalvitals.analysis.stats_agent import StatsAnalysisAgent
from evalvitals.eval_agent.stages.diagnosis import DiagnosisAgent

protocol = ExperimentProtocol(
    description="The VLM gives wrong left/right positions in spatial questions.",
    task_domain="spatial reasoning",
    success_criteria="Positions must match what is visible in the image.",
)

model = compose("qwen2.5-vl-7b-instruct", "hf_local",
                want={Capability.GENERATE, Capability.ATTENTION})
judge = AgyModel()   # or any Model with Capability.GENERATE

loop = VLDiagnoseLoop(
    model=model,
    probe_agent=ProbeAgent(max_analyzers=3),
    stats_agent=StatsAnalysisAgent(judge=judge),
    diagnosis_agent=DiagnosisAgent(judge=judge),
    max_cycles=3,
    protocol=protocol,
    run_logger=RunLogger(),
)
report = loop.run(failure_cases)

print(report.resolved)           # True when M5 finds a supported, consistent hypothesis
print(report.final_hypotheses)   # list[Hypothesis] — status SUPPORTED/REFUTED/INCONCLUSIVE

fix = loop.run_m4(report, failure_cases)   # post-loop fix proposal
```

`RunLogger()` above writes just the JSONL event log. For the full output
directory — `report/`, `figures/`, `artifacts/`, per-trial `fixes/`/`experiments/`
folders, `manifest.json` — construct a `RunContext` and pass `run_logger=ctx.logger`
instead; see [RunContext](architecture.md#runcontext-single-owner-of-a-runs-output-directory).

**`ExperimentProtocol`** is the human prior that anchors the loop. M1 uses it
to select analyzers relevant to the task; M5 uses it to reject hypotheses that
drift from what the user was investigating:

```python
from evalvitals.eval_agent.stages.protocol import ExperimentProtocol

protocol = ExperimentProtocol(
    description="free text — what the experiment tests and what failure looks like",
    task_domain="spatial reasoning",      # short label
    success_criteria="what counts as a pass",
    failure_patterns="observations already noticed (optional)",
    target_modalities=frozenset({"text", "image"}),
)
```

## AgenticDiagnoseLoop — Judge-Decided M1-M5 (Alternative to the Fixed Cycle)

`VLDiagnoseLoop` above always runs M1→M2→M3→M5 in the same order every cycle.
`AgenticDiagnoseLoop` wraps the identical stages, confirm-split, and post-loop
`run_m4`/`run_fix` — but a CLI judge decides which tool to call next each
turn (probe, run stats, explore the raw data, cluster failure modes, search
for new failure cases via a Macro/Micro MCTS probe search, propose
hypotheses, test one, fix, or stop), instead of a fixed sequence. The host — not the judge —
enforces call limits, tool preconditions, and the stopping discipline: the
judge cannot declare success (`stop(resolved=true)`) until a hypothesis has
actually been tested and is statistically supported and protocol-consistent;
an early attempt is rejected and fed back to the judge.

```python
from evalvitals.eval_agent import AgenticDiagnoseLoop, ClaudeModel

loop = AgenticDiagnoseLoop(
    model=model,
    protocol=protocol,
    judge=ClaudeModel(),        # the decision judge — any CLI-backed Model
    max_actions=12,             # hard cap on judge decision turns
    run_logger=RunLogger(),
)
report = loop.run(failure_cases)   # same VLDiagnoseReport shape as VLDiagnoseLoop
```

`report.stopped_by` is one of `agent_stop` / `max_actions` / `budget` /
`time_budget` / `invalid_actions` (three consecutive unparseable judge
responses, even after a repair prompt). Two new `run_log.jsonl` event types —
`agent_decision` (the chosen tool + rationale) and `agent_tool` (the dispatch
layer's accept/reject outcome) — sit alongside the reused
`probe`/`analysis`/`diagnosis`/`surgery` events from the wrapped stages; see
`evalvitals.eval_agent.log_schema` (schema version 3).

## Input Modes — Submitting a Diagnosis Run

There are two ways to trigger the diagnosis agent.

### Mode 1 — Container Submission (Reproduce a Known Failure)

Write a `run.py` that defines your failure cases as `FailureCase` objects,
then package it into a Docker container. The agent runs the M1→M5 loop
inside the container and writes findings to `outputs/`.

1. **Define failure cases and run the loop** in `run.py` — see the
   `VLDiagnoseLoop` example above; build a `CaseBatch` of `FailureCase`
   objects instead of `failure_cases` and call `loop.run(cases)`.
2. **Add a Dockerfile + docker-compose.yml** mirroring one of the concrete
   example directories under `examples/analyzer_demos/`, `examples/m2_m3/`,
   or `examples/m1_m3/` / `examples/m1_m4/`.
3. **Submit the container:**

```bash
docker compose up
```

Outputs (logs, analyzer artifacts, hypotheses) are written to `outputs/` in
the container, mounted to your local directory via the compose volume. See
`examples/m1_m3/qwen_loop_agy/` and
`examples/m1_m3/qwen_video_temporal/` for complete working examples.

### Mode 2 — Natural-Language Description (Agent Writes the Container)

Describe the failure in plain English. The scaffold generator produces a
ready-to-run Docker experiment with a `run.py`, `Dockerfile`, and
`docker-compose.yml`. A CLI coding agent (Claude Code, Gemini CLI, …) can
write a fully customised `run.py`; otherwise a template is used as a
starting point.

```python
from evalvitals.eval_agent import scaffold_from_description

out = scaffold_from_description(
    description="My VLM frequently confuses left and right when answering "
                "spatial relationship questions about images.",
    model_key="qwen2.5-vl-7b-instruct",
    output_dir="./my_experiment",
    provider="claude_code",   # optional: generates a bespoke run.py; omit for a template
    cli_model="sonnet",
)
# cd my_experiment && docker compose up
```

Or via CLI:

```bash
python -m evalvitals.eval_agent.nl_runner \
    --description "My VLM confuses left and right in spatial questions" \
    --model qwen2.5-vl-7b-instruct \
    --out ./my_experiment \
    --provider claude_code --cli-model sonnet   # omit --provider for template mode

cd my_experiment && docker compose up
```

The generated scaffold contains `run.py` (diagnosis script with
`ExperimentProtocol` pre-filled from your description), `Dockerfile`,
`docker-compose.yml` (mounts HF cache, GPU, outputs, and agy binary), and
`.gitignore`. In template mode, edit the `CASES` list in `run.py` to add your
own failure examples before running `docker compose up`.

## AutoDiagnoseLoop — Legacy M1→M4 Pipeline

`AutoDiagnoseLoop` closes the analysis→diagnosis→intervention cycle automatically.
It needs a **judge model** (any instruction-following model with `GENERATE`) and
the model under evaluation.

```python
from evalvitals import Capability
from evalvitals.eval_agent import AutoDiagnoseLoop, DiagnosisAgent
from evalvitals.models import compose
from evalvitals.models.backends.base import RuntimeConfig

# 1. The model under evaluation (local VLM with white-box access)
model = compose("qwen3-vl-8b-instruct", "hf_local", want={Capability.ATTENTION})

# 2. A capable judge (API model — needs only GENERATE)
judge = compose("qwen3-8b", "api", RuntimeConfig(generate_fn=my_generate_fn))

# 3. Build the loop
loop = AutoDiagnoseLoop(
    model=model,
    diagnosis_agent=DiagnosisAgent(judge=judge),
    max_cycles=3,     # max M1→M4 iterations
    max_analyzers=4,  # analyzers per cycle (ranked by diagnostic priority)
)

# 4. Run on your failure cases
report = loop.run(failure_cases)

print(report.resolved)                        # True if an intervention fixed it
for h in report.final_hypotheses:
    print(h.statement, "→", h.status)        # SUPPORTED / REFUTED / INCONCLUSIVE
print(report.final_results.keys())            # analyzers run in the last cycle
```

To persist the run (event log, figures, report files, fix/experiment trial
folders), wrap the loop in a `RunContext` — see "Log and persist a diagnosis
run" in [Extending](extending.md).

### Analysis-only mode

Omit `diagnosis_agent` to run M1+M2 only — useful when you want the probe's
ranked analysis without automated hypothesis generation:

```python
loop = AutoDiagnoseLoop(model=model, max_analyzers=5)
report = loop.run(cases)
for name, result in report.final_results.items():
    print(name, result.summary())
```

### Controlling analyzer selection

`StrategyProbe` automatically detects model kind and ranks analyzers:

```python
from evalvitals.eval_agent import StrategyProbe, ModelKind

probe = StrategyProbe()
probe.detect_kind(model)                     # ModelKind.VLM / AGENT / LLM
probe.select(model, max_analyzers=4)         # e.g. ["pope", "chair", "attention", "mm_shap"]
```

### Custom intervention (SurgeryAgent)

Override the default label-correlation verification with your own logic:

```python
from evalvitals.eval_agent import SurgeryAgent, InterventionResult, HypothesisStatus

def my_verify(hypothesis, model, results, data):
    # domain-specific logic — return True when fixed
    fixed = run_my_intervention(model, data)
    return InterventionResult(
        hypothesis=hypothesis,
        status=HypothesisStatus.SUPPORTED if fixed else HypothesisStatus.INCONCLUSIVE,
        fixed=fixed,
        evidence={"custom": True},
    )

loop = AutoDiagnoseLoop(
    model=model,
    diagnosis_agent=DiagnosisAgent(judge=judge),
    surgery_agent=SurgeryAgent(verify_fn=my_verify),
)
```

Or trigger a **param sweep** — re-run analyzers with modified settings to compare
before/after findings:

```python
loop = AutoDiagnoseLoop(
    model=model,
    diagnosis_agent=DiagnosisAgent(judge=judge),
    surgery_agent=SurgeryAgent(analyzer_params={"attention": {"top_k": 20}}),
)
```

---

## Eval Agent — Pre-Registered A/B Loop

`EvalOrchestrator` enforces selective-inference safety: mine on `explore`,
pre-register a falsifiable hypothesis (hash + timestamp) **before** unblinding,
test once on `validate`, lock `confirm` for the final report.

```python
from evalvitals.eval_agent import EvalOrchestrator, PreregisteredHypothesis, DataSplit

hyp = PreregisteredHypothesis(
    predicate="cluttered scenes",
    statement="chain-of-thought prompt improves accuracy on multi-step questions",
    direction="B>A",
    min_effect=0.03,
    alpha=0.05,
    split="validate",
)

orch = EvalOrchestrator(split=DataSplit(explore_frac=0.5, validate_frac=0.3))
report = orch.run(cases, hyp, strategy_a, strategy_b)

# report keys: prereg_hash, decision, effect, ci, e_value, split, hypothesis
print(report["prereg_hash"])   # proves hypothesis was fixed before unblinding
print(report["decision"])      # "REJECT H0" or "inconclusive"
print(report["effect"])        # float
```

**CounterfactualReplay** ranks agent trajectory steps by causal influence (flip-rate):

> Causal framework: Pearl (2000) *Causality*  
> Applied to NLP: Vig et al. 2020 — <https://arxiv.org/abs/2004.12265>

```python
from evalvitals.analyzers.agent.counterfactual import CounterfactualReplay

def rerun_fn(trajectory, step_idx, seed):
    # wrap your live agent + verifier here; return True/False (success)
    ...

analyzer = CounterfactualReplay(rerun_fn=rerun_fn, n_replays=5)
result = analyzer._run(model, CaseBatch([case_with_trajectory]))
# result.findings["per_case"][0]["steps"] →
#   [{"step": 1, "action": "extract_answer", "flip_rate": 0.80}, ...]
# High flip_rate ⇒ that step was causally influential.
```

Full runnable example: `examples/preregistered_ab_demo/eval_agent/` (no API key needed, `docker compose up`).
