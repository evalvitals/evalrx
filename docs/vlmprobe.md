# VLMProbe: readouts, explanations and interventions

The optional VLMProbe integration adds 24 method factories to EvalRX through
one registered analyzer, `vlmprobe`. It binds the same local model already
owned by the caller, uses its native image/text preprocessing, and returns an
EvalRX summary together with the original VLMProbe tensors and provenance.
The integration supports language, vision and vision-language experiments when
the supplied adapter exposes the computations the selected method needs.

It is an additive integration: existing `logit_lens`, `gradcam`,
`generic_attention` and `causal_trace` analyzers keep their original contracts.
For example, VLMProbe attention Grad-CAM operates on attention probabilities;
it does not substitute for EvalRX's vision-feature Grad-CAM.

## Installation and model support

```bash
pip install "evalrx[vlmprobe]"
```

The optional extra installs VLMProbe 0.9.x and its visualization dependencies.
The core EvalRX install remains Torch-free. It does not change the version
range of EvalRX's local-model backend.

VLMProbe's automatic Hugging Face adapters are audited against **Transformers
5.3.x**. Use a separate environment for that adapter contract when the existing
EvalRX environment uses another release:

```bash
python -m venv .venv-vlmprobe
. .venv-vlmprobe/bin/activate
pip install "evalrx[local,vlmprobe]" "transformers>=5.3,<5.4"
```

This version choice belongs to the probing environment, not to EvalRX's shared
dependency constraints. VLMProbe also has limited 4.57.x support for native
GPT-2, Llama and Qwen2.5-VL. An explicit, verified VLMProbe adapter can support
another architecture/version without changing EvalRX's loader. See the
[upstream model matrix](https://github.com/gexinh/vlm-probing/blob/main/docs/models/README.md)
and [adapter contract](https://github.com/gexinh/vlm-probing/blob/main/docs/ADAPTERS.md).

Automatic contracts cover native GPT-2/Llama, eligible ViT classifiers, and
eight VLM families: Qwen2-VL, Qwen2.5-VL, Qwen3-VL, Qwen3.5, SmolVLM,
InternVL, LLaVA and Pixtral through LLaVA's Mistral backbone. Support remains
method-specific. API endpoints, vLLM servers, JAX arrays, and unknown model
wrappers do not expose the live PyTorch computations needed by this bridge.

## Configure once, run on cases

```python
from evalrx.analyzers.vlmprobe import VLMProbeAnalyzer

analyzer = VLMProbeAnalyzer(
    method="lens.logit",
    method_config={"layers": [0, 4, 8], "tokens": "last_prompt"},
    max_cases=2,
)
result = analyzer.run(local_evalrx_model, cases)
print(result.summary())

# The raw results remain available for numerical analysis and plots.
probe_result = result.artifacts["probe_results"][0]
ax = result.visualize(index=0, plot="lens_heatmap", target_token_id=answer_id)
result.save_probes("outputs/logit-lens")
```

`method` is a required `family.factory` key from the inventory below.
`method_config` configures the bound factory; `run_config` supplies arguments
for the actual experiment. `run_config` can be a mapping or a callback
`callback(case) -> mapping`, so paired inputs and masks can differ by case.
`max_cases=1` limits expensive probing by default; set it deliberately for a
larger evaluation (`0` analyzes all cases). `output_dir` optionally saves
native results during the run. The registry key `vlmprobe` requires explicit
method selection; it is not a default analysis with a guessed target.

For an EvalRX Hugging Face local handle, the bridge uses the handle's native
text/chat or multimodal encoding and its already-loaded model. It does not
convert detached EvalRX `Trace` tensors back into a gradient graph. For
another local wrapper, pass `prober=...` and an explicit input builder:

```python
from vlmprobe import Prober

probe = Prober(torch_model, processor=processor, adapter=verified_adapter)

def build_inputs(model, case):
    # Return native model kwargs or ProbeInputs with an explicit TokenLayout.
    return processor(text=case.inputs.prompt, return_tensors="pt")

analyzer = VLMProbeAnalyzer(
    method="lens.logit", prober=probe, input_builder=build_inputs,
)
```

The custom builder owns chat formatting, device placement and visual token
expansion. Masks and token IDs must align with the model's **expanded**
sequence, not simply the tokenizer's unexpanded prompt. `ProbeInputs` can
carry an explicit `TokenLayout(valid, visual, prompt, token_ids)` when the
adapter cannot infer it.

## Method inventory

The source links identify the original method or a relevant application.
The scope column states what the bridge actually exposes; a related citation
does not imply that every experiment or decoding component is implemented.

```python
from evalrx.analyzers.vlmprobe import list_methods
print(list_methods())  # Discover the 24 keys without loading a model.
```

### Lens

| Factory key | What it measures or does | Requirements and source |
|---|---|---|
| `lens.logit` | Decodes intermediate residuals through native final normalization and vocabulary readout. | Residual/readout access; [nostalgebraist, *Interpreting GPT: the logit lens*](https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens). |
| `lens.tuned` | Applies learned per-layer affine translators before readout. | Fitted model-bound artifacts; [Belrose et al., *Eliciting Latent Predictions from Transformers with the Tuned Lens*](https://arxiv.org/abs/2303.08112). |
| `lens.jacobian` | Transports residuals using a calibrated mean downstream Jacobian, then decodes. | Fitted transport artifacts; [Gurnee et al., *Verbalizable Representations Form a Global Workspace in Language Models*](https://transformer-circuits.pub/2026/workspace/index.html). |
| `lens.attention` | Uses learned vocabulary decoders on projected head contributions. | Projected heads and fitted joint decoders; [Sakarvadia et al., *Attention Lens*](https://arxiv.org/abs/2310.16270). |
| `lens.embed` | Finds cosine neighbors in the language input-embedding table. | Embedding-space captures; [Fan et al., *What Do Visual Tokens Really Encode?*](https://arxiv.org/abs/2603.00510). No automatic pruning/classification benchmark. |
| `lens.patchscope` | Transfers a residual into a configured target prompt and reads its logits. | Source/target positions; explicit mapping across models; [Ghandeharioun et al., *Patchscopes*](https://proceedings.mlr.press/v235/ghandeharioun24a.html). |

### Attention

| Factory key | What it measures or does | Requirements and source |
|---|---|---|
| `attention.profile` | Reports entropy and token-group attention mass; can retain raw attention. | Observable probabilities; supporting primitive, no unique originating paper. |
| `attention.rollout` | Composes residual-aware self-attention transitions. | Consecutive attention layers from layer 0; [Abnar and Zuidema, *Quantifying Attention Flow in Transformers*](https://aclanthology.org/2020.acl-main.385/). Not attention max-flow. |
| `attention.grad_cam` | Produces target-conditioned maps using attention heads as channels. | Differentiable probabilities and target metric; [Selvaraju et al., *Grad-CAM*](https://arxiv.org/abs/1610.02391), with the [author ViT attention adaptation](https://github.com/hila-chefer/Transformer-Explainability). |
| `attention.attribution` | Integrates gradients along each layer's zero-to-observed attention path. | Editable consumed probabilities; [Hao et al., *Self-Attention Attribution*](https://arxiv.org/abs/2004.11207). Signed attribution is retained. |
| `attention.tam` | Combines backward Markov transitions with input-path attention feedback. | Consecutive stack and floating processed-input tensor; [Yuan et al., *Explaining Information Flow Inside Vision Transformers Using Markov Chain*](https://xai4debugging.github.io/files/papers/explaining_information_flow_in.pdf). |
| `attention.beyond_intuition` | Computes headwise or tokenwise perception with integrated reasoning feedback. | Stack/input path; token variant also requires pre-LN inputs and projected V; [Chen et al., *Beyond Intuition*](https://openreview.net/forum?id=rm0zIzlhcX). |
| `attention.relevance` | Propagates positive attention-times-gradient self-attention relevance. | Differentiable consecutive stack; [Chefer et al., *Generic Attention-model Explainability*](https://arxiv.org/abs/2103.15679). Full cross-attention propagation is outside this scope. |
| `attention.dtd_lrp` | Runs genuine author relevance-propagation rules on a copied ViT backend. | Eligible native ViT classifier and class target; [Chefer et al., *Transformer Interpretability Beyond Attention Visualization*](https://arxiv.org/abs/2012.09838). Not supported on native VLM decoder blocks. |
| `attention.head_logits` | Projects head contributions using a declared fixed normalization scale. | Projected heads and linear readout; [*A Mathematical Framework for Transformer Circuits*](https://transformer-circuits.pub/2021/framework/index.html). A primitive distinct from trained Attention Lens. |
| `attention.reweight` | Rescales selected consumed attention probabilities and measures score changes. | Editable probabilities and metric; generic intervention. It does not implement PAI's complete pre-softmax/contrastive algorithm. |
| `attention.temperature` | Changes masked attention-logit temperature and measures score changes. | Editable pre-softmax scores and metric; generic intervention, no unique originating paper. |

Raw attention is available through
`attention.profile` with `include_attention=True`; it is not an extra factory.
Beyond Intuition's `variant="head"` and `variant="token"` share one factory.

### Causal

| Factory key | What it measures or does | Requirements and source |
|---|---|---|
| `causal.patch` | Replaces selected residual activations with aligned reference activations. | Paired inputs and metric; general activation intervention, including the [causal-tracing application in ROME](https://arxiv.org/abs/2202.05262). No weight-editing implementation. |
| `causal.attribute` | Estimates replacement effects from local gradients and activation differences. | Differentiable residuals, reference and metric; [Nanda, *Attribution Patching: Activation Patching at Industrial Scale*](https://www.neelnanda.io/mechanistic-interpretability/attribution-patching). An approximation, not an actual replacement measurement. |
| `causal.path` | Measures selected sender-to-receiver paths through controlled forwards. | Editable head outputs at every decoder layer and receiver controls; [Wang et al., *Interpretability in the Wild*](https://arxiv.org/abs/2211.00593), Appendix B. Does not automatically assign functional head names. |
| `causal.knockout` | Blocks selected key-to-query attention routes before softmax. | Editable masked scores and metric; [Zhang et al., *Cross-modal Information Flow in Multimodal Large Language Models*](https://arxiv.org/abs/2411.18620). Joint layer windows are supported. |
| `causal.steer` | Injects a supplied residual direction and measures or generates its effects. | Residual access and explicit direction/strength; generic intervention. |
| `causal.vsv` | Constructs image-specific directions from image-plus-text minus text-only residuals. | Positive/negative native inputs and explicit strength; [Li et al., *The Hidden Life of Tokens*](https://proceedings.mlr.press/v267/li25ca.html). Raw residual VSV transfer; SLA is not implemented. |
| `causal.eap_ig` | Scores explicit connections; audited GPT-2 graph also supports actual circuit recovery. | Native GPT-2 transformer graph or independently replaceable explicit edge messages; [Syed et al., *Attribution Patching Outperforms Automated Circuit Discovery*](https://arxiv.org/abs/2310.10348) and [Hanna et al., *Have Faith in Faithfulness*](https://arxiv.org/abs/2403.17806). |

Original text/ViT methods applied to decoder-side VLM computations are method
transfers. Models with hybrid linear attention may expose residual lenses while
failing the consecutive self-attention stack needed by rollout/TAM. Consult
`probe.describe()` for exact supported layers and missing requirements.

The bridge retains upstream numerical guards as well as architecture gates.
For example, VLMProbe 0.9.0 rejects temperature sharpening when finite-minimum
masked logits overflow; it does not silently replace the model's attention
mask. Exact supported configurations are covered by the
[integration validation](versions/vlmprobe-integration-v1.md).

## Paired inputs, metrics and operations

An intervention requires a stated behavior to measure. For one-token candidate
answers, configure a token-logit margin:

```python
analyzer = VLMProbeAnalyzer(
    method="causal.patch",
    method_config={"layers": [0, 1], "tokens": "last_prompt"},
    metric={"kind": "token_margin", "positive": yellow_id, "negative": black_id},
    run_config=lambda case: {"source": reference_inputs[case.id]},
)
result = analyzer.run(local_evalrx_model, cases)
```

Candidate token IDs must correspond to the actual tokenizer. A single-token
margin is not a multi-token answer probability. For complete answers, supply
an upstream `SequenceLogProb` metric with explicit teacher-forced answer mask
and expanded token IDs. Custom metrics can be graph-preserving
`callable(logits) -> scalar_or_batch_tensor`, or objects exposing
`score(logits, layout)`. Do not detach or convert logits to NumPy inside a
metric used by attribution.

Metric mappings also support `kind="class_score"` with an explicit `target`,
and `kind="sequence_logprob"` with an explicit `answer_mask`. To select a
classifier's probability instead of its logit, pass a native
`ClassScore(target, kind="probability")` object.

| Method / operation | Per-case `run_config` arguments |
|---|---|
| `causal.patch`, `causal.attribute` | `source`: reference native inputs; `alignment` optional. |
| `causal.path`, `operation="run"` or `"sweep"` | `donor`: reference native inputs; `senders`/`receivers` in `method_config`. Sweep tests senders independently. |
| `lens.patchscope` | `target_inputs`: native inputs for the configured target prompt. |
| `causal.steer` | `direction`: tensor or layer-to-tensor mapping. For generation set `joint=True` in `method_config`, `operation="generate"` and `max_new_tokens` in `run_config`. |
| `causal.vsv` | `positive`, `negative`: contrastive native inputs; explicit `strength` in `method_config`. For `operation="generate"`, also provide generation options. |
| `causal.eap_ig`, `operation="run"` | Reference under `source` or `donor`; the bridge selects the graph's required name. Set `graph` in `method_config`. |
| `causal.eap_ig`, `operation="evaluate"` | Native GPT-2 transformer graph; reference and `budgets` for actual retained-edge recovery. |
| `attention.dtd_lrp` | `target`: class ID or aligned target tensor; native inputs must contain `pixel_values`. |

`operation="run"` is the default. `sweep` is specific to controlled path
patching; `evaluate` is specific to the native GPT-2 circuit implementation;
`generate` is available for joint residual steering and VSV. These operations
retain different output axes, recorded in the native metadata.

Strict patching checks valid/visual/prompt masks, token IDs and adapter grid
metadata. Explicit `alignment="position"` relaxes token equality only;
it does not align image regions or remap token positions. The caller must
verify semantic correspondence. Independent per-layer interventions remain
independent; `joint=True` on steering or knockout edits all selected live
layers together.

## Calibrate outside evaluation

Tuned, Attention and Jacobian Lens require a fitted artifact. `calibration_dir`
loads VLMProbe's per-layer artifact format. It is not a directory of arbitrary
third-party weights. Bindings identify the model checkpoint, tokenizer,
readout and calibration split, and are checked on loading.
`configured_method` is reserved for these explicitly fitted lenses and must
match the declared lens and bound model. Choose either `configured_method`
or `calibration_dir`, not both.

```python
binding = {
    "model_id": "checkpoint-and-revision",
    "tokenizer_id": "tokenizer-and-revision",
    "readout_id": "native-final-norm-and-head",
    "calibration_id": "independent-train-validation-split",
}
analyzer = VLMProbeAnalyzer(
    method="lens.tuned",
    method_config={"layers": [0, 1], "binding": binding},
)
lens = analyzer.bind(local_evalrx_model)
lens.fit_batches(train_inputs, validation_inputs, "artifacts/tuned")

# Evaluation uses the explicit fitted method; it never trains on its cases.
evaluation = VLMProbeAnalyzer(method="lens.tuned", configured_method=lens)
result = evaluation.run(local_evalrx_model, held_out_cases)
```

For later processes, provide the same binding and `calibration_dir`.
Training preserves the base model's parameters. Use independent datasets of a
reasonable size, not the one or two demonstration cases. Author GPT-2 Tuned
Lens artifacts require the upstream validated importer, including checkpoint
identity and translator-index conversion. See the
[training guide](https://github.com/gexinh/vlm-probing/blob/main/docs/TRAINING.md).

## Results, visualization and persistence

`result.findings` holds compact agent-facing summaries. The full
`result.artifacts["probe_results"]` list preserves VLMProbe `ProbeResult`
objects with `.method`, detached CPU `.tensors`, and configuration/coordinate
`.metadata`. Each entry corresponds to one analyzed case.

`result.visualize(index=0, plot=..., **kwargs)` delegates to the matching
VLMProbe plotting function. Supported plots include `lens_heatmap`,
`attention_overlay`, `causal_heatmap`,
`intervention_curves`, `connections`, `head_paths` and `receiver_paths`.
Pass the image, native grid/visual positions, fixed target token or row labels
needed by the chosen plot; the bridge does not invent image geometry or target
labels. See the
[visualization contract](https://github.com/gexinh/vlm-probing/blob/main/docs/VISUALIZATION.md).

Plots that need prepared series or connection tables accept `data=...`.
For example, `plot="intervention_curves"` needs a labeled curve mapping;
`plot="connections"` needs explicit edge/node coordinates. The complete
upstream visualization module also remains accessible for multi-map
comparisons and other helpers.

`result.save_probes(directory)` writes portable native tensor/metadata
artifacts for each analyzed case. `Result.to_json()` retains the compact
findings and metadata without serializing heavy tensors. Complete model
weights, processors and trained artifacts remain separate.

## Experimental interpretation and limits

- A lens reports how a representation decodes; it does not establish causal use.
- Attention maps describe observed weights or estimated score contributions.
  Gradient attributions may legitimately be zero or signed; neither is an
  unsupported-method placeholder.
- Activation/path patching and knockout measure actual controlled changes.
  Attribution patching and EAP scores are estimates and can disagree with
  measured replacement effects.
- EAP-IG's `graph="transformer"` follows input-embedding integration on the
  audited GPT-2 graph. `graph="activation"` is a separately labeled
  caller-defined edge-message interpolation variant, not an automatic VLM
  circuit finder. Retained-edge recovery needs actual complement-edge edits.
- DTD/LRP requires genuine relevance-propagation rules on the eligible ViT
  backend. Generic attention-times-gradient does not become LRP by renaming it.
- VSV supplies residual directions only. The full VISTA SLA decoder and
  author MLP/PCA processing are outside this integration; strength is
  model-dependent and larger values can cause repetition.
- Probes require uncached full-sequence forwards. Gradient methods cannot run
  inside `torch.inference_mode()`. Concurrent or nested probes on the same
  bound model/adapter are unsupported.

The [offline CPU example](../examples/vlmprobe_integration.py) checks the bridge
with a tiny randomly initialized GPT-2 and an in-memory tokenizer. It downloads
no weights or data and demonstrates API behavior, not an interpretation claim
about a pretrained model.
