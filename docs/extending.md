# Extending EvalRX

EvalRX is designed to grow through small extension points: analyzers, model
specs, backends, datasets, and statistical routines.

## Add an Analyzer

An analyzer should behave like an sklearn estimator:

- constructor parameters are explicit,
- required capabilities are declared on the class,
- `run(model, data)` returns a `Result`,
- model-specific logic is avoided whenever possible.

Skeleton:

```python
from evalrx.core import Analyzer, Capability, Result, register_analyzer


@register_analyzer("my_analysis")
class MyAnalyzer(Analyzer):
    name = "my_analysis"
    requires = frozenset({Capability.LOGITS})

    def __init__(self, top_k: int = 5):
        super().__init__(top_k=top_k)

    def _run(self, model, cases):
        case = cases[0]
        trace = model.forward(case.inputs, capture={Capability.LOGITS})
        findings = {
            "top_k": self.top_k,
            "seq_len": trace.seq_len,
        }
        return Result(
            analyzer=self.name,
            model=repr(model),
            findings=findings,
            artifacts={},
            cases=cases,
        )
```

Guidelines:

- Use `Capability` checks instead of `isinstance(model, SomeModel)`.
- Request only the internals needed for the analysis.
- Keep heavy artifacts out of `findings`; put them in `artifacts`.
- Make `summary()` useful for humans and `findings` useful for agents.
- If an artifact is a 2-D spatial map over a case's image (e.g. an attention
  heatmap), add `overlay()` / `image_overlays(fig_dir, stem_prefix)` methods to
  your `Result` subclass (see `RelativeAttentionResult` in
  `analyzers/attention/relative_attn.py`). `RunLogger` calls `image_overlays()`
  on any `Result` that defines it — no per-analyzer wiring needed to get the
  overlay PNGs into `figures/` alongside the bare heatmap.

## Use a Custom or Fine-Tuned Model

If you have a model that is already loaded in memory — a fine-tuned checkpoint,
a research model, or anything `from_pretrained` returns — use `evalrx.wrap`
instead of registering a spec:

```python
import evalrx
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("my-org/my-llama")
tokenizer = AutoTokenizer.from_pretrained("my-org/my-llama")

wrapped = evalrx.wrap(model, tokenizer)

# Discover which analyzers are compatible
print(evalrx.registry.analyzers.names_compatible_with(wrapped))

# Run any compatible analyzer
from evalrx.analyzers.lens.logit_lens import LogitLensAnalyzer
result = LogitLensAnalyzer().run(wrapped, "The capital of France is")
```

`wrap()` infers capabilities from the live model: attention, hidden states, and
logits are available for any text decoder-only model. Attention capture requires
eager attention — `wrap` enables it automatically when the model supports it; if
not, reload with `attn_implementation="eager"`.

`wrap()` currently supports text decoder-only (causal LM) models. For VLMs, use
the curated spec path (`evalrx.load` or `compose`) — VLM forward capture with
image-token mask and spatial layout is handled automatically for models in the spec
registry. If your model has an unusual architecture not supported by automatic
inference, add a `ModelSpec` (see below) and use `evalrx.load`.

## Add a Model Spec

Add model identity to `evalrx/specs.py`:

```python
# Text-only LLM
_add(ModelSpec(
    key="new-model-key",
    family="new_family",
    model_type="new_model_type",
    hf_repo="org/repo",
    auto_class="AutoModelForCausalLM",
    processor_class="AutoTokenizer",
    min_transformers="4.50.0",
    module_paths=ModulePaths(decoder_layers="model.layers"),
))

# VLM — add a VisionSpec so the backend can locate image tokens and the patch grid.
# image_token_id_attr: name of the config attribute holding the image-pad token id.
# merge_size_attr:     dotted path to the spatial merge size (None if not applicable).
# grid_source:         "grid_thw" (Qwen-VL style) | "grid_hw" | "fixed".
_add(ModelSpec(
    key="my-vlm-7b",
    family="my_vlm",
    model_type="my_vlm",
    hf_repo="org/my-vlm-7b",
    auto_class="AutoModelForImageTextToText",
    processor_class="AutoProcessor",
    min_transformers="4.50.0",
    module_paths=ModulePaths(
        decoder_layers="model.language_model.layers",
        vision_tower="model.visual",
    ),
    vision=VisionSpec(
        image_token_id_attr="image_token_id",
        merge_size_attr="vision_config.spatial_merge_size",
        grid_source="grid_thw",
    ),
))
```

Specs should describe architecture facts and caveats. They should not load model
weights or import heavy runtime dependencies.

## Add a Backend

A backend turns a `ModelSpec` into a concrete `Model` and declares what it can
provide.

Skeleton:

```python
from evalrx.core import Capability
from evalrx.models.backends.base import Backend


class MyBackend(Backend):
    kind = "my_backend"
    capabilities = frozenset({Capability.GENERATE})

    def build(self, spec, runtime):
        return MyModel(spec=spec, runtime=runtime)
```

Then register it in `evalrx/models/backends/__init__.py`:

```python
BACKENDS["my_backend"] = MyBackend
```

Guidelines:

- Keep heavy imports inside `build` or inside the concrete model class.
- Fail early when the backend cannot support a requested spec.
- Let capabilities describe runtime behavior, not model identity.

## Add a Dataset Loader

Dataset loaders should produce `FailureCase` or `CaseBatch`. This keeps raw
benchmarks, hand-authored cases, and agent-generated cases interoperable.

```python
from evalrx.core import FailureCase, CaseBatch


def load_cases(path) -> CaseBatch:
    cases = [
        FailureCase.from_prompt(
            "...",
            label="fail",
            metadata={"source": str(path)},
        )
    ]
    return CaseBatch(cases)
```

## Add a Diagnosis Judge

`DiagnosisAgent` accepts any `Model` with `Capability.GENERATE` as the judge.
The simplest swap is an API model:

```python
from evalrx.eval_agent import DiagnosisAgent, AutoDiagnoseLoop
from evalrx.models import compose
from evalrx.models.backends.base import RuntimeConfig

judge = compose("qwen3-8b", "api", RuntimeConfig(generate_fn=my_generate))
loop  = AutoDiagnoseLoop(model=my_model, diagnosis_agent=DiagnosisAgent(judge=judge))
```

The judge receives a JSON dump of all analyzer findings and must reply with lines
of the form `HYPOTHESIS: ...` / `FAILURE_MODE: ...` (one pair per hypothesis) or
`NO_ISSUE`.  You can use `evalrx.eval_agent.diagnosis._DIAGNOSE_PROMPT` as a
starting point and override it by subclassing `DiagnosisAgent`:

```python
from evalrx.eval_agent.diagnosis import DiagnosisAgent, _parse_hypotheses, DiagnosisResult
import json

class MyDiagnosisAgent(DiagnosisAgent):
    _PROMPT = "Your custom prompt with {model_name} and {findings_json}."

    def diagnose(self, results, model_name):
        summary = {name: r.findings for name, r in results.items()}
        raw = self.judge.generate(
            self._PROMPT.format(
                model_name=model_name,
                findings_json=json.dumps(summary, indent=2, default=str),
            )
        )
        return DiagnosisResult(
            model_name=model_name,
            hypotheses=_parse_hypotheses(str(raw), model_name),
            findings_summary=summary,
            raw_judge_output=str(raw),
        )
```

## Add a Custom SurgeryAgent Intervention

For domain-specific verification, pass `verify_fn` to `SurgeryAgent`:

```python
from evalrx.eval_agent import SurgeryAgent, InterventionResult, HypothesisStatus

def domain_verify(hypothesis, model, results, data):
    # e.g. re-evaluate after patching the prompt template
    improved = run_ablation(model, data, hypothesis)
    return InterventionResult(
        hypothesis=hypothesis,
        status=HypothesisStatus.SUPPORTED if improved else HypothesisStatus.REFUTED,
        fixed=improved,
        evidence={"ablation_result": improved},
    )

from evalrx.eval_agent import AutoDiagnoseLoop, DiagnosisAgent
loop = AutoDiagnoseLoop(
    model=my_model,
    diagnosis_agent=DiagnosisAgent(judge=judge),
    surgery_agent=SurgeryAgent(verify_fn=domain_verify),
)
```

## Extend StrategyProbe for a new model kind

If you add a new capability type (e.g. `Capability.AUDIO`), you can teach
`StrategyProbe` about it by passing a `priority_override`:

```python
from evalrx.eval_agent import StrategyProbe, ModelKind

probe = StrategyProbe(priority_override={
    ModelKind.LLM:   ["attention", "logit_lens", "token_entropy"],
    ModelKind.VLM:   ["pope", "chair", "attention"],
    ModelKind.AGENT: ["loop_detect", "ignored_obs"],
})
loop = AutoDiagnoseLoop(model=my_model, probe=probe, ...)
```

An override is looked up by `ModelKind` and replaces composition entirely — it
is an explicit statement about ordering, so the batch's modality slots do not
reorder it. Without one, ranking composes the per-slot lists in
`_SLOT_PRIORITY`, which is where a new modality's analyzers belong.

An analyzer that dereferences a media slot must say so:

```python
class MyAudioAnalyzer(Analyzer):
    applies_to_modalities = frozenset({"text", "audio"})   # what the MODEL must accept
    requires_modalities   = frozenset({"audio"})           # what the BATCH must fill
```

The first is matched by intersection against the model's declaration, so
`"text"` alone matches every model; the second is the gate that keeps the
analyzer off a batch with no audio in it.

## Log and persist a diagnosis run

The recommended way to persist a run is `RunContext` — it owns the whole
output directory (`report/`, `figures/`, `artifacts/`, `experiments/`,
`fixes/`, `manifest.json`, …) and hands every producer its subdirectory,
including a bound `RunLogger`. See the "RunContext" section in
[Architecture](architecture.md) for the full layout and the per-trial
`fixes/` / `experiments/` folders.

```python
from evalrx.eval_agent import AutoDiagnoseLoop, DiagnosisAgent, RunContext

with RunContext("runs/exp_01") as ctx:
    loop = AutoDiagnoseLoop(
        model=model,
        diagnosis_agent=DiagnosisAgent(),
        run_logger=ctx.logger,
    )
    report = loop.run(cases)
# manifest.json + README.txt written, logger closed on exit.
```

If you only want the JSONL event log and artifact sink — without `RunContext`'s
`report/`/`figures/`/manifest layout — construct `RunLogger` standalone:

```python
from evalrx.eval_agent import AutoDiagnoseLoop, DiagnosisAgent, RunLogger

loop = AutoDiagnoseLoop(
    model=model,
    diagnosis_agent=DiagnosisAgent(),
    run_logger=RunLogger("runs/exp_01"),   # explicit path
    # run_logger=RunLogger()              # auto: runs/<YYYYMMDD_HHMMSS>/
)
report = loop.run(cases)
```

Output layout (standalone `RunLogger`, no `RunContext`):

```text
runs/exp_01/
├── run_log.jsonl                         ← one JSON line per M1/M2/M3/M5 event
└── artifacts/
    ├── c0_attention_attn_weights.npy     ← attention tensor, cycle 0
    ├── c0_cka_layer_similarities.npy     ← CKA similarity matrix, cycle 0
    └── c1_attention_attn_weights.npy     ← cycle 1 after data refocus
```

Each line in `run_log.jsonl` contains `event` (one of `probe`, `explore`,
`analysis`, `diagnosis`, `surgery`, `loop_end`, …), `cycle`, `ts` (ISO-8601), a
`schema_version` (int — bumps only on a breaking field rename/removal, so a
parser doesn't need to guess from `evalrx_version`), and stage-specific
fields:

| `event` | Key fields |
|---|---|
| `probe` | `analyzers`, `findings` (JSON), `artifact_paths` |
| `explore` | (optional, `VLDiagnoseLoop(explorer=...)`) `ok`, `n_observations`/`n_charts`/`n_charts_rendered`/`n_tables`/`n_candidate_signals`, `observations`, `figures` (rendered PNGs M3 was shown), `out_dir`/`report_path` (`exploratory_report.json` + `tables/` + `figures/`). Descriptive only — the explorer's in-sample verdicts appear as counts under `adjudication`, never in the confirmatory M2 family |
| `analysis` | `severity`, `findings` (human-readable), `narrative`, `stats_tool_results`/`stats_results`/`stats_plan`/`corrected_rejections` (externalized to `artifacts/` above 4 KB) |
| `diagnosis` | `hypotheses`, `raw_judge_output` (full LLM response) |
| `surgery` | `hypothesis`, `status`, `fixed`, `evidence`, `n_refocused_cases` |
| `loop_end` | `cycles`, `resolved`, `final_hypotheses` |

The full, authoritative field contract is the published JSON Schema
(`evalrx/eval_agent/run_log.schema.json`, from `log_schema.py`); validate a
log with `from evalrx.eval_agent import iter_log_errors` (needs the optional
`jsonschema` dep). See `docs/architecture.md` for details.

Standard shell tools work directly on the log:

```bash
# Live-stream events as the loop runs
tail -f runs/exp_01/run_log.jsonl

# Extract all Gemini diagnosis outputs across cycles
jq 'select(.event=="diagnosis") | .raw_judge_output' runs/exp_01/run_log.jsonl

# See which analyzers ran and their findings per cycle
jq 'select(.event=="probe") | {cycle, analyzers, findings}' runs/exp_01/run_log.jsonl

# Load an attention tensor for manual inspection
python -c "import numpy as np; a = np.load('runs/exp_01/artifacts/c0_attention_attn_weights.npy'); print(a.shape)"
```

`RunLogger` is also a context manager, which ensures the file is closed even if
the loop raises:

```python
with RunLogger("runs/exp_01") as logger:
    loop = AutoDiagnoseLoop(model=model, run_logger=logger)
    loop.run(cases)
```

To add custom log entries (e.g. pre/post-run metadata), write directly to the
logger:

```python
logger = RunLogger("runs/exp_01")
logger._write({"event": "run_config", "model": repr(model), "n_cases": len(cases)})
loop = AutoDiagnoseLoop(model=model, run_logger=logger)
loop.run(cases)
```

## Add Statistical Evaluation

Statistical routines should consume `Result` objects or collections of results.
They should avoid depending on analyzer-specific internals unless that contract
is explicit.

Good inputs:

- `Result.findings`
- experiment ids,
- case ids,
- model/backend/spec metadata,
- repeated-run measurements.

Avoid making statistical code depend on raw tensors unless the test is
specifically about tensor-level measurements.
