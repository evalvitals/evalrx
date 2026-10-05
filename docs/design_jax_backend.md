# Design: a JAX white-box backend (`jax_local`), reference model Gemma-4-E2B

Status: 2026-09-26, phase 1 implemented (text read access on Gemma-4-E2B /
E4B through Google DeepMind's `gemma` library); see section 9 for what landed,
how to run it, and what is still open. Phases 2 and 3 are design only.

For Colab TPU setup, runnable examples and measured single-device validation,
see the [Colab TPU tutorial](colab_tpu.md).

EvalRX already runs JAX-served models as black boxes: any OpenAI-compatible
endpoint, or an in-process `generate_fn`, gives the loop `GENERATE` and
`LOGPROBS`, which covers Stage 0, the black-box M1 analyzers, M2 to M4, and the
L0 to L2 repair tiers. What it cannot do is read or write a JAX model's
internals: attention, hidden states, logits from a forward pass, gradients,
forward-pass interventions, and LoRA. This document designs that.

The reference model is **Gemma-4-E2B** (`gemma-4-e2b-it` in `evalrx/specs.py`).
It is the right first target for three reasons:

- it is one checkpoint that serves all three benchmark modalities (text,
  image + text, audio + text), so one adapter exercises every capture path;
- the torch twin already runs in `examples/benchmark/{llm,vlm,alm}/gemma`,
  which gives a parity oracle for free: same manifests, same graders, same
  pinned M1 sets, `--backend hf_local` versus `--backend jax_local`;
- Google publishes Gemma checkpoints for JAX and maintains a Flax reference
  implementation, so the adapter wraps a first-party model instead of a port.

## 1. Goals and non-goals

Goals, in priority order:

1. A `jax_local` backend that returns the same `Trace` the torch backend
   returns, so every existing white-box analyzer runs unchanged on a JAX model.
2. A small adapter protocol so a JAX model of any framework, including a
   user's own training code, plugs in without touching EvalRX internals.
3. Forward-pass interventions (L3b) and the contrastive-decoding repairs (L3a)
   expressed against the `Model` contract, so both backends share them.
4. LoRA repair (L4) on JAX.

Non-goals for this design: replacing the torch backend, supporting every JAX
framework out of the box, or removing torch from the analyzer code. The last
one is possible later and is discussed in section 4.3.

## 2. What exists today, and where PyTorch leaks through

Analyzers never touch a framework directly. They call five things on
`evalrx.core.model.Model`: `generate`, `forward`, `logprobs`, `chat`, and the
two lens accessors `unembed_weight()` / `final_norm()`. `forward` returns a
`Trace` (`evalrx/core/model.py`):

| field | shape (torch path) | notes |
|---|---|---|
| `tokens`, `token_ids` | `(seq,)` | prompt tokens after the chat template |
| `hidden_states` | list of `L+1` tensors `(seq, dim)` | entry 0 is the embedding output, as in HF |
| `attentions` | list of `L` tensors `(heads, seq, seq)` | requires eager attention on torch |
| `logits` | `(seq, vocab)` | |
| `token_type_map` | `TokenTypeMap` | VLM: image positions plus patch grids |
| `extras` | dict | `image_token_mask`, `audio_token_mask`, `attn_semantics`, grids |

`CaptureSpec` bounds layers, heads, and whether tensors move to CPU.
`Capability` negotiation happens in `compose()` before any weights load, and
`ProbeAgent` only offers analyzers whose `requires` the model provides.

This contract is framework-neutral in intent. Three things leak PyTorch
through it, and each one is a work item below:

1. **Trace payloads are `torch.Tensor`**, and six analyzers do torch math on
   them: `lens/logit_lens.py`, `lens/layer_contrast.py`, `geometry/cka.py`,
   `geometry/linear_probe.py` (trains a probe with `torch.optim`),
   `uncertainty/entropy.py`, `attention/summary.py` and `rollout.py`.
2. **L3b and L4 reach around `Model`.** `eval_agent/stages/fix_internals.py`
   reads the private `model._hf` tuple (`_resolve_hf`), registers a torch
   forward hook on the input embedding for `visual_embedding_boost`, and
   hands the same object to `peft` in `run_lora_repair`.
3. **L3a and the calibrated L2 repairs are methods on `HFLocalModel`**
   (`generate_vcd`, `generate_instruction_cd`, `generate_tcd`, `generate_aad`,
   the specialist routes), and `repair_catalog.discover_methods` finds them
   with `callable(getattr(model, executor))`. Layer discovery
   (`models/backends/hf/discover.py`) walks `nn.ModuleList`.

One more fact shapes the priorities. The benchmark's **pinned M1 sets for the
Gemma cells are all black-box plus logprobs** (`_common/tasks/*.py`):
`answer_extraction_audit`, `termination_audit`, `selfcheck_consistency`,
`self_consistency`, `calibration`, `logprob_entropy`, `coverage_verification_gap`,
`perturbation_battery`, `format_sensitivity`. None needs `ATTENTION` or
`HIDDEN_STATES`. So on the default benchmark run, a JAX backend first has to
match the torch path on `GENERATE` and `LOGPROBS`; white-box capture pays off
through `--m1-selection judge` (the catalog then offers attention summary,
sink, rollout, relative attention, logit lens, layer contrast, CKA, linear
probe, entropy, VCD), through `AnalysisModule`'s image-attention rule, and
through the L3a/L3b/L4 repair tiers, which the API backends clamp away.

## 3. Architecture

### 3.1 `jax_local` backend over a thin adapter protocol

JAX has no single model API the way HF torch does. The backend therefore never
knows Flax NNX from Flax Linen from MaxText or Penzai. It talks to a
`JaxModelAdapter`, and each framework, or a user's own model, implements that
protocol. `JaxLocalModel` wraps the adapter exactly as `VLLMOfflineModel`
wraps vLLM (`models/backends/vllm_offline.py` is the template) and registers
as `BACKENDS["jax_local"]` in `models/backends/__init__.py`. Heavy imports
stay inside `load()`.

```python
# evalrx/models/backends/jax/protocol.py (as implemented; the sketch this replaced had
# prefill/step and phase-2/3 methods, which are deferred to those phases)

class Encoding:            # what the adapter's tokenizer / processor produced
    ids: list[int]; tokens: list[str]; text: str | None        # text = the rendered prompt
    media: dict; image_token_mask / audio_token_mask: list[bool] | None
    grids: list[tuple[int, int, int]]; image_token_id: int | None

class SamplingParams:      # hf_local's kwarg contract, backend-neutral
    max_new_tokens: int; temperature: float = 0.0 (<= 0 -> greedy); top_p; top_k; seed; stop

class ForwardOut(NamedTuple):
    logits: Array | None            # (S, V) after the model's own soft-cap
    hidden: list[Array] | None      # L+1 x (S, D): embeddings first, LAST entry post final-norm (HF layout)
    attn: list[Array] | None        # L x (H, S, S) probabilities; reference attention only
    extras: dict

class JaxModelAdapter(Protocol):
    n_layers: int; modalities: frozenset[str]; reference_attention: bool
    def load(self) -> None
    def encode(self, inputs, *, chat_template: bool) -> Encoding
    def render_chat(self, messages, tools=None) -> Encoding
    def decode(self, ids) -> str
    def forward(self, enc, *, capture: frozenset[str], layers: tuple[int, ...] | None = None) -> ForwardOut
    def unembed(self) -> Array                       # (V, D)
    def final_norm_params(self) -> NormParams | None  # scale, eps, plus_one -> rebuilt as a torch RMSNorm
    def generate(self, enc, params: SamplingParams) -> GenerateOut   # ids + text
```

`capture` holds the strings `"logits"`, `"hidden_states"`, `"attentions"`.
When `layers` is given the adapter may leave unrequested entries `None`; the
lists keep full length and the backend subsets them exactly like hf_local's
`_maybe_subset`. Logprobs need no adapter method: the backend decodes greedily,
then runs ONE teacher-forced `forward` over prompt + continuation.

`JaxLocalModel` maps the protocol onto `Model`:

| `Model` method | implementation |
|---|---|
| `capabilities` | `GENERATE`, `LOGITS`, `LOGPROBS`, `HIDDEN_STATES` always; `ATTENTION` only when the adapter runs reference attention (see 3.4); `TOOL_CALLS` when the tokenizer's chat template renders tools, the same rule `vllm_offline` uses |
| `generate` | `adapter.generate` with the spec's `chat_template_kwargs`; `RuntimeConfig.apply_chat_template` honoured as on hf_local |
| `logprobs` | greedy `prefill`/`step` loop, `log_softmax` plus `top_k` per step, mapped to `TokenLogprob` |
| `forward` | `adapter.forward` with `capture` translated to adapter flags and `CaptureSpec.layers/heads` passed through, then the boundary conversion of 3.3 |
| `unembed_weight`, `final_norm` | converted once and cached (3.3) |
| `chat` | render `messages` through the tokenizer's chat template and call `generate`; tool calls parsed by the existing codec |

Alongside `compose(spec, "jax_local", ...)`, add `evalrx.wrap_jax(model,
tokenizer, adapter=...)` mirroring `wrap()`: a model already loaded in
someone's training script gets a `JaxLocalModel` with an inferred spec, no
registry entry needed.

### 3.2 Identity: a `JaxSpec` on `ModelSpec`

`ModelSpec` is HF-centric (`hf_repo`, `auto_class`, `module_paths`). Follow
the `VisionSpec` / `AudioSpec` pattern and add an optional `jax: JaxSpec`
field rather than overloading the HF fields:

```python
@dataclass(frozen=True)
class JaxSpec:                        # evalrx/core/spec.py (implemented)
    framework: str                    # "gemma" | "flax_linen" | "flax_nnx" | "maxtext" | "custom"
    checkpoint: str                   # Orbax dir (gs:// or local), Kaggle handle, or safetensors path
    tokenizer: str                    # SentencePiece model (gs:// or local) or an HF tokenizer id
    adapter: str                      # import string "pkg.module:factory", called as factory(spec, runtime)
    model_class: str = ""             # framework class, e.g. "Gemma4_E2B"
    reference_attention: bool = True  # materialise (H,S,S) probs; False = fused kernels, no ATTENTION
    sharding: dict = field(default_factory=dict)   # mesh axes; empty = single device
```

`compose` fails early when `backend.kind == "jax_local"` and `spec.jax is
None`, the same way it refuses `api_only` specs on local backends. For Gemma 4
the E2B and E4B entries gain a `jax=JaxSpec(framework="gemma", ...)` (the
12B "Unified" variant has no class in the gemma library, so it stays torch-only);
everything else about them (vision/audio specs, `chat_template_kwargs`,
caveats) is shared with the torch path, which is the point.

### 3.3 Trace boundary: convert to torch on the CPU path, inside the backend

Phase 1 keeps the analyzers untouched. `JaxLocalModel.forward` honours
`CaptureSpec.to_cpu` (default `True`) with `jax.device_get` and wraps each
array with `torch.from_numpy`, producing exactly the tensors `HFLocalModel`
produces. The cost is a CPU-only torch wheel in the `jax` extra. The
`to_cpu=False` path is out of scope for phase 1: zero-copy `torch.from_dlpack`
across CUDA needs a CUDA torch build and is a memory-optimisation, not a
correctness need.

Two accessors need care so lens analyzers stay faithful:

- `unembed_weight()` returns the `(V, D)` matrix the model actually applies,
  converted once and cached (Gemma vocab is ~262k, so this is gigabytes in
  float32; keep bf16 and let the analyzer's `.float()` decide).
- `final_norm()` returns a small torch module rebuilt from the JAX norm's
  weight and epsilon. Gemma also soft-caps final logits; the adapter's
  `forward` returns logits after the cap, and the parity test in section 6
  is what catches a mismatch here.

Dropping torch from the analyzers is a separate, optional project: port the
six files in section 2 to the Python array API (`array_api_compat`) and give
`linear_probe` a closed-form or optax fit. It is not needed for any phase
below.

### 3.4 Capture mechanics inside the adapter

These are the JAX-specific problems, and they belong to the adapter, not to
`JaxLocalModel`:

- **No forward hooks.** Intermediates come from `sow` /
  `capture_intermediates` in Flax Linen, `nnx.Intermediate` variables in Flax
  NNX, selector insertion in Penzai, or from a layer `scan`'s stacked outputs.
  Models that `lax.scan` over blocks (MaxText, Levanter) return `(L, S, D)`
  hidden states for free from the scan's `ys`.
- **Attention probabilities are not materialised by fused kernels** (cudnn,
  splash, Pallas flash). The adapter needs a reference-attention switch driven
  by `RuntimeConfig.attn_impl == "eager"`, the analog of hf_local's
  `_ensure_eager_attention`, and must declare `ATTENTION` only when it is on.
  Sliding-window layers still produce a masked `(S, S)` matrix.
- **jit and shapes.** `capture`, `layers` and `heads` are static arguments so
  unrequested captures are dead-code-eliminated inside the jitted forward;
  sub-selecting layers and heads before returning keeps a full-attention
  capture from allocating `L x H x S x S`. Pad sequence length to buckets
  (for example multiples of 128) so a new prompt does not recompile.
- **Reproducible sampling.** `generate` takes an explicit PRNG key derived
  from the case id and the sample index. Stage 0 is then reproducible while
  `self_consistency` and `coverage_verification_gap` still get fresh samples.
- **Logprobs.** The gemma sampler does not return per-step logits, so the
  backend decodes greedily and then runs one teacher-forced forward over
  prompt + continuation: exact for greedy decoding, one extra pass.
- **Memory.** The `to_cpu` transfer happens per layer, not for the whole
  stack at once; on a 48 GB card the E2B forward with reference attention at
  2k context fits, the 12B does not without layer sub-selection.

### 3.5 Multimodal capture: image and audio token maps

The torch path builds `TokenTypeMap` and the `extras` masks from the HF
processor's output (`_encode_vlm`, `_populate_vision_extras`,
`_populate_audio_extras` in `hf_local.py`). `tokentype.py` is already
torch-free and accepts lists or numpy arrays, so the JAX adapter's `Encoding`
feeds `build_token_type_map` directly. The adapter is responsible for:

- `image_token_mask`: positions equal to the image placeholder id, or the
  processor's `mm_token_type_ids` when it emits them (Gemma 4 does on the
  torch side; the JAX processor must expose the equivalent);
- `audio_token_mask`: positions equal to the audio placeholder id, one token
  per encoded frame group;
- image grids: the torch path leaves `grids=[]` for Gemma 4 because the
  variable-resolution tiling is not rebuilt (`grid_source="fixed"` with no
  tile size, see `_GEMMA4_CAVEATS`). The JAX adapter fills them (verified
  2026-09-28): the gemma vision encoder resizes every image to multiples of
  `patch_size * pooling_kernel_size` = 48 px per side, keeping the aspect
  ratio under a patch budget, then average-pools 3 x 3 patches row-major, so
  an image yields exactly `(H/48) * (W/48)` soft tokens in row-major order.
  `Encoding.grids = [(1, H/48, W/48)]` and `extras["image_spatial_shape"]`
  come out exact (a 850 x 600 ChartQA image is 19 x 14 = 266 tokens), which
  unlocks the spatial attention overlays the torch path forgoes.

### 3.6 Interventions through named sites, not hooks (L3b)

Define a framework-neutral site vocabulary in `evalrx.core` and add
`Model.intervene(sites: dict[str, Callable]) -> ContextManager`:

| site | tensor handed to the callable |
|---|---|
| `embed_out` | `(S, D)` token embeddings after the embedding layer, before block 0 |
| `resid_pre[l]`, `resid_post[l]` | residual stream entering / leaving block `l` |
| `attn_scores[l]` | `(H, S, S)` pre-softmax scores |
| `attn_out[l]`, `mlp_out[l]` | block `l` sub-layer outputs before the residual add |
| `final_norm_out` | `(S, D)` input to the unembedding |
| `logits` | `(S, V)` |

`HFLocalModel` implements `intervene` with forward hooks on the modules
`_discover` already locates. The JAX adapter implements it functionally:
wrapping the block call in NNX, `intercept_methods` in Linen, or tree
rewriting in Penzai, then re-jitting. `visual_embedding_boost` becomes
`intervene({"embed_out": scale_rows(image_token_mask, gamma)})`, written once
against the site and the mask in `Trace.extras`, and `fix_internals` stops
reading `_hf`. Future primitives (sink suppression on `attn_scores[l]`,
activation steering on `resid_post[l]`) get both backends at once. The
vocabulary is deliberately close to TransformerLens hook names so it reads
familiarly.

### 3.7 Repairs: L3a executors and L4 LoRA

Move the L3a algorithms out of `HFLocalModel` into `models/paper_methods/` as
functions over the white-box primitives, and give both local backends the same
method names through a `WhiteBoxRepairsMixin`, so `repair_catalog` keeps
discovering by `getattr`. Instruction contrastive decoding already has this
shape in its decoder-only branch: two `forward(capture={LOGITS})` calls and a
fuse step. The audio and vision contrastive methods additionally need input
transforms, which is what `adapter.encode_variants` is for:

| repair | tier | needs from the adapter | Gemma-4 modality |
|---|---|---|---|
| instruction CD (text-prefix variant) | L3a | `forward` LOGITS | image, audio |
| VCD | L3a | `encode_variants("vcd_noise")` on pixel values, `forward` LOGITS | image |
| AAD, TCD | L3a | `encode_variants("tcd_blur")` on the waveform, `forward` LOGITS + HIDDEN_STATES, `audio_token_mask` | audio |
| visual embedding boost | L3b | `intervene` on `embed_out` | image |
| LoRA | L4 | `lora_finetune` (NNX `LoRA` or the framework's own, optax AdamW), trained on `finetune_pool`, returns a new adapter that `JaxLocalModel` swaps in for the paired validation | all |

`paper_method_fidelity` must keep reporting the text-prefix ICD variant as
architecture-adapted, exactly as it does for decoder-only VLMs on torch.

### 3.8 Benchmark, packaging, image

- `examples/benchmark/_common/models.py`: `"jax_local"` is in `BACKENDS`;
  `Family("gemma")` keeps `hf_local` as default, `--backend jax_local` opts in
  and reuses the hf_local spec keys (done).
- `_common/runner.py` `load_model`: a `jax_local` branch building
  `RuntimeConfig(device, dtype, max_new_tokens, apply_chat_template=True,
  engine_kwargs={"checkpoint": --model-path})` (done). `effective_fix_tier`
  clamps jax_local to L2 for now: the backend reads internals, but the L3a
  executors and L3b hooks are still hf_local methods (phases 2 and 3 lift it).
- `pyproject.toml`: a `jax` extra with `jax`, `flax`, `gemma>=4.0.1` (Python
  3.12+ upstream) and `torch` for the trace boundary (done). CUDA wheels are a
  separate install choice (`jax[cuda12]`); the gemma library brings orbax,
  kauldron and the `dialog` formatter; no `transformers` is needed, the
  tokenizer is SentencePiece.
- `examples/benchmark/docker/Dockerfile`: a `gemma_jax` stage from
  `python:3.12-slim` with the CUDA 12 jax wheels (the cluster driver is 12.4,
  which the cu12 wheels support), not `FROM base`, because base is python 3.11
  with the cu129 torch stack (done 2026-10-01; image `evalrx-bench-gemma-jax`,
  11.2 GB). The JAX services live in their own leaves,
  `examples/benchmark/{llm,vlm,alm}/gemma_jax/`, not in the gemma leaves: both
  backends use the same `--model` key, so a shared leaf would write both into
  one `outputs/<model>/<dataset>/`. A local Orbax mirror is mounted at `/ckpt`
  through `EVALRX_JAX_CKPT`.

## 4. Gemma-4-E2B reference adapter

The adapter wraps Google's Flax reference implementation and the published
JAX checkpoints. The repo's spec already fixes what the adapter must
reproduce; the table maps spec facts to adapter responsibilities.

| spec fact (`evalrx/specs.py`, `gemma-4-e2b-it`) | adapter responsibility |
|---|---|
| `chat_template_kwargs={"enable_thinking": False}`; thought-channel leaks are model behaviour | render the same template with the same kwarg; a leaked `thought` preamble is graded by `scoring.py` exactly as on torch |
| `AttnSemantics.STANDARD`, sliding / full interleave, all layers attention | reference attention returns `(H, S, S)` per layer with the window mask applied; `n_layers` counts every block |
| per-layer embeddings (PLE), ~10 GB BF16 | hidden state `l` is the residual stream after block `l`, after the PLE add; PLE weights stay out of `unembed()` |
| `VisionSpec(image_token_id_attr="image_token_id", grid_source="fixed", fixed_tokens_per_tile=None)` | `image_token_mask` marks the soft-token positions (reported as `<|image|>` = 258880, the id hf_local's `image_token_id` plays); `grids` and `image_spatial_shape` are exact on JAX (3.5) |
| `AudioSpec(audio_token_id_attr="audio_token_id", audio_tower="model.audio_tower")` | `audio_token_mask` marks the soft-token positions (`<|audio|>` = 258881); `encode_variants("tcd_blur")` (still open) would blur the waveform before the audio encoder |
| `is_reasoning=True` | nothing extra; thinking stays off |
| 12B is encoder-free "Unified" | same adapter, `modalities` still all three; embed projections replace towers, so `attn`/`hidden` shapes are unchanged |

Per modality, the acceptance path uses the existing Gemma cells:

| modality | cell | Stage 0 parity check | white-box check |
|---|---|---|---|
| text | `llm/gemma` `bbh_causal_judgement` `--limit 8` | identical PASS/FAIL labels to hf_local under greedy decoding; sampled runs compare accuracy within noise | `--m1-selection judge` offers logit lens / attention summary; `Trace` shapes match |
| image | `vlm/gemma` `chartqa` `--limit 8` | same as above with images through the JAX processor | `image_token_mask` count equals the torch path's; relative attention runs |
| audio | `alm/gemma` `mmau` `--limit 8` | same with 16 kHz audio | `audio_token_mask` count matches; TCD's `layer_stability` gets `L+1` hidden states |

Facts about the Gemma JAX library this adapter depends on, verified on
2026-09-26 against `gemma` 4.0.1 / `flax` 0.12.10 / `jax` 0.11.2 (they change
between releases; re-check when bumping):

| item | verified fact | where it matters |
|---|---|---|
| model class and checkpoint | `gm.nn.Gemma4_E2B` / `Gemma4_E4B` (also 31B, 26B-A4B; no 12B Unified). Orbax checkpoints in the PUBLIC bucket `gs://gemma-data/checkpoints/gemma4-{e2b,e4b}-it`, readable anonymously (`gsutil`, or orbax via gcsfs); E2B is 18.25 GB on disk, stored **float32** (19.8 GB text-only in memory, 9.9 GB after the bf16 cast) | `JaxSpec.checkpoint`, `load()`, `RuntimeConfig.dtype` |
| Linen versus NNX | Flax **Linen** (`nn.Module`, `self.param`); blocks named `layer_{i}` (35 on E2B), `final_norm`, `embedder` | the intermediates mechanism |
| intermediates | `capture_intermediates` with a `(module, method)` filter works: `embedder.encode` (embedding output), `layer_i.__call__` -> `(cache, x)`, `final_norm.__call__`; `return_hidden_states` equals the `final_norm` output | `forward`, HF-layout hidden states |
| attention probabilities | materialised softmax (no fused kernel); probs go through `kd.nn.Identity` named `attention_weights` inside `attn`, shape `(B, T, heads, S)`; sliding layers carry the window mask | full `(H, S, S)` per layer; no module wrapping needed |
| jit trap | `Transformer.__call__` is `nn.jit`-wrapped and its cached trace bakes in the first capture filter; the adapter calls the level beneath the wrapper through its own per-configuration `jax.jit` (checked: per-filter intermediates, logits identical) | correctness of `layers` sub-selection |
| padding | right-padding with PAD (0) leaves valid positions unchanged (1e-7 on a random model; PAD masked out of positions and attention) | length buckets (128 ... 4096) bound recompiles |
| tokenizer | SentencePiece `gs://gemma-data/tokenizers/tokenizer_gemma4.model` (4.5 MB, public); vocab 262144; `<bos>`=2, `<eos>`=1, `<pad>`=0, `<|turn>`=105, `<turn|>`=106, `<|think|>`=98, `<|channel>`=100, `<channel|>`=101, `<|image|>`=258880, `<|audio|>`=258881 | `Encoding`, end-of-generation, special-token stripping |
| chat format | `dialog.Format.GEMMA4`: `<|turn>user\n...<turn|>\n<|turn>model\n`; thinking is a `<|think|>` control token, so `enable_thinking=False` renders none. Where the HF `chat_template.jinja` puts it for `enable_thinking=True` could not be checked (gated repo, no HF token on the host): the adapter raises for that setting | prompt parity with hf_local |
| sampler | `gm.text.Gemma4Sampler(model, params, tokenizer, sampling, cache_length, max_out_length, pad_length).sample(text, max_new_tokens, rng, return_state=True)`; `Greedy` / `TopPSampling(p, temperature)` / `TopkSampling(k, temperature)` / `RandomSampling(temperature)`; ends on EOS, `<turn|>`, tool-response | `generate` |
| soft-cap and head-side norm | `final_logit_softcap=30.0` (`tanh(x/30)*30`), `attn_logits_soft_cap=None`; logits `x @ embedder.input_embedding.T` (tied, `(V, D)`); `final_norm` is `x * rsqrt(mean(x^2)+1e-6) * scale` (plain scale, not 1+scale) | `unembed`, `final_norm_params`, lens faithfulness |
| vision (verified 2026-09-28) | `text_only=False` builds the 167 M-param `vision_encoder` (16 layers, d 768, float32). 16-px patches, aspect-ratio-preserving resize to multiples of 48 px, 3 x 3 average pooling row-major, **at most `num_mm_tokens_per_image` = 280 soft tokens per image** on E2B / E4B (`config.vision_encoder`). The library's `Gemma4Sampler` DEFAULT `max_soft_tokens=1120` does not match this encoder: with it the text side reserves ~1090 slots while the encoder emits ~270 pooled tokens, and the remaining slots are filled by gathering token 0 (checked on a random-init encoder: 266 valid pooled tokens for a 266-token reservation at 280, 267 for a 1092-token reservation at 1120). The adapter therefore reads `patch_size`, `num_mm_tokens_per_image`, `pooling_kernel_size` from the model config and passes them to the sampler too. Text-side expansion: `<|image|>` -> `\n\n <|image> P*n <image|> \n\n` (`P` = internal -2, replaced by the merged embedding) | `Encoding`, `grids`, `generate` |
| audio (verified 2026-09-28) | `text_only=False` builds the 305 M-param conformer `audio_encoder` (12 layers, d 1024 -> 1536, float32) on the raw 16 kHz waveform: 128-mel filterbank, 20 ms frames / 10 ms hop (+1-sample unfold quirk), two stride-2 subsamplings, so a clip of `n` samples gives `((n-321)//160 + 1 - 1)//2 + 1` then once more, capped at `audio_seq_length=750` (~30 s; longer clips raise in the adapter as `hf_local._check_audio_duration` does). Text-side expansion: `<|audio|>` -> `<|audio> A*m <audio|>` (no `\n\n`; `A` = internal -4). `audio_soft_token_counts` is a STATIC argument of the forward, so every distinct clip length recompiles | `Encoding`, `audio_token_mask`, `generate` |
| multimodal forward | `Transformer.__call__` with images and `return_last_only=False` applies `remove_mm_logits`, a Gemma-3-era step assuming a fixed count per image, which garbles the sequence axis on Gemma 4's variable counts (the sampler never hits it: prefill uses `return_last_only=True`). The adapter's media forward calls `_encode_and_get_inputs` + `_apply_attention` + `embedder.decode` + soft-cap directly (Flax `apply(method=fn)`), the same code minus that step; `_encode_and_get_inputs.embeddings` (captured, Flax wraps private methods too) is the merged block-0 input, i.e. HF's `hidden_states[0]`. Media towers and their projections stay float32 (`initialize_param_with_dtype` excludes them); the bf16 cast skips the same paths | `forward`, `hidden[0]` |
| thread safety (verified 2026-10-02) | the stack is not thread-safe: kauldron's `ktyping` keeps its type-check scopes on a process-global stack, so two `Gemma4Sampler.sample` calls from different threads fail `assert s == self` in `kauldron/ktyping/scope.py` (reproduced with two concurrent generates; the same calls pass one after the other). `JaxLocalModel` therefore serialises every adapter call on one re-entrant lock; hf_local needs none because torch ops are thread-safe | M1 runs its analyzers from a thread pool: the chartqa shakedown lost `selfcheck_consistency` and `coverage_verification_gap` to this before the lock |
| sampler seed (verified 2026-10-02) | `Gemma4Sampler.sample(rng=...)` takes an int seed or a key; `rng=None` makes the library draw one from Python's `random`. The adapter used to pass a constant 0 when `SamplingParams.seed` was unset, so every sampled call for a prompt returned the same text: M1's `self_consistency` / `selfcheck_consistency` / `coverage_verification_gap` saw 5 identical samples per case and M3 proposed "the harness is not sampling". Now a fresh 31-bit seed per unseeded call, the caller's seed when given, 0 under greedy. Also: the library has no combined top-p + top-k method, so nucleus wins when both are set | every sampled probe and the `self_consistency_N` L2 repair |
| LoRA (phase 3) | `gm.nn.LoRA(rank=..., model=...)` wraps every Dense / Einsum with kauldron `peft` layers; the checkpoint loader knows how to reconcile LoRA trees | L4 |

## 5. Phases

**Phase 1, text read access.** `JaxLocalBackend`/`JaxLocalModel`, the
protocol, `JaxSpec`, the Gemma adapter on text prompts, the torch boundary,
`wrap_jax`. Delivers: Stage 0, `LOGPROBS`, `HIDDEN_STATES`, `LOGITS`,
`ATTENTION` under reference attention; every M1 analyzer that runs on torch
text models; L0 to L2 fixes; text-prefix ICD. Acceptance: the `llm/gemma`
smoke row above, plus the unit tests in section 6.

**Phase 2, sites and interventions.** The site vocabulary, `Model.intervene`
on both backends, `visual_embedding_boost` rewritten on sites,
`fix_internals` freed from `_hf`. Delivers L3b on JAX and keeps the torch
`fix_internals` tests green.

**Phase 3, multimodal and the rest.** Image and audio `Encoding`,
`TokenTypeMap` and masks (landed 2026-09-28, section 9), then
`encode_variants` for VCD and TCD/AAD, the `WhiteBoxRepairsMixin`,
`input_gradient` for the `GRADIENTS` analyzers, LoRA (open).
Acceptance: the `vlm/gemma` and `alm/gemma` smoke rows above and a full
M1-to-fix chain on one cell per modality with `--backend jax_local`.

## 6. Tests

- **Toy model, CPU, no checkpoint.** A two-layer Flax transformer built in the
  test (vocab 64, dim 8, 4 heads) behind a `ToyAdapter`, the JAX twin of
  `FakeCausalLM` in `tests/test_models/test_wrap.py`. Assert `Trace.provided`,
  shapes, `require()`, and that `LogitLensAnalyzer`, `AttentionAnalyzer`,
  `TokenEntropyAnalyzer` run end to end. Assert `compose(..., want={ATTENTION})` raises
  when `reference_attention=False`.
- **Parity, marked `gpu`.** Load Gemma-4-E2B in torch and JAX, run the same
  eight prompts, compare logits and per-layer attention within tolerance,
  and compare greedy outputs token for token. This is where RoPE, soft-cap,
  window masks and PLE drift show up.
- **Stage 0 parity through the harness.** The three smoke rows in section 4;
  `baseline.json` labels are the artefact compared.
- **Intervention identity.** `intervene({})` and
  `intervene({"embed_out": identity})` reproduce the plain forward bit for bit
  on both backends.

## 7. Risks and open questions

- **Attention probabilities may not be exposed by the reference library.**
  Then the adapter maintains a reference attention module, which is a small
  fork to keep in step with releases. Mitigation: declare `ATTENTION` only
  when it works; everything else is unaffected.
- **Two implementations of one model drift.** Parity tests are the guard, and
  they are `gpu`-marked, so they run on demand, not in CI.
- **Compile latency.** Reference attention plus capture flags multiply jit
  variants. Sequence-length buckets and caching per `(capture, layers,
  heads)` tuple keep it bounded; expect the first M1 pass on a cell to be
  slower than torch eager.
- **Whether to build the site vocabulary for torch at the same time.** Doing
  both at once is less total work than a JAX-only patching path beside the
  existing hooks, but it touches `fix_internals`, which has a large test
  suite. Recommendation: yes, in phase 2, behind the current
  `visual_embedding_boost` test expectations.
- **Concurrency.** `--concurrency` is honoured only for the endpoint backend.
  A JAX in-process model wants batching instead; an optional
  `generate_batch` on the adapter, used by `CaseDiscoveryAgent` when present,
  is the natural later addition and is out of scope here.

## 8. Out of scope

Serving a JAX model behind an OpenAI-compatible endpoint (already works via
`--backend endpoint`), a JAX port of the registered calibrated L2 specialists
(they call other models and are backend-independent by nature; they should
move to a shared mixin regardless of JAX), and removing torch from the
analyzer code.

## 9. Status (2026-09-28): phase 1 and the multimodal half of phase 3 landed

What exists in the repo:

- `evalrx/core/spec.py`: `JaxSpec`, `ModelSpec.jax`; `evalrx/specs.py` gives
  `gemma-4-e2b-it` and `gemma-4-e4b-it` their JAX twin.
- `evalrx/models/backends/jax/`: `protocol.py` (the contract above),
  `boundary.py` (jax -> CPU torch, bf16 bit-exact; a torch RMSNorm rebuilt
  from the JAX norm), `adapters/gemma.py` (the reference adapter: text, image
  and audio inputs). The pre-0.1.2 `evalrx/models/jax/` paths are deprecated
  aliases.
- `evalrx/models/_media.py`: image / audio resolution shared by `hf_local`
  and `jax_local` (moved out of `hf_local.py`, whose private names stay bound).
- `evalrx/models/backends/jax/backend.py`: `JaxLocalModel` / `JaxLocalBackend`,
  registered as `BACKENDS["jax_local"]`; `evalrx.wrap_jax(adapter)` mirrors
  `wrap()`.
- Benchmark: `--backend jax_local` on the Gemma sizes (`_common/models.py`,
  `run.py`, `runner.py`) for the llm, vlm and alm cells (the llm cells pass
  `text_only=True` and skip the towers), fix ladder clamped to L2.
- Tests: `tests/test_models/test_jax_local.py` (toy adapters on CPU, no
  downloads: Trace layout, subsetting, logprobs consistency, logit lens, the
  bf16 boundary, media masks -> `TokenTypeMap` / `image_spatial_shape`, the
  audio token formula against the library sampler, the image grid against the
  library's count, placeholder round-trips; two `gpu`-marked real-weights
  tests), plus the spec / registry assertions in `test_compose.py` and
  `test_benchmark_family_specs.py`.

Environment used (the repo `.venv` keeps its torch 2.6 / transformers 4.57
stack; the JAX stack lives beside it):

```bash
uv venv .venv-jax --python 3.12
uv pip install --python .venv-jax/bin/python "jax[cuda12]" "gemma==4.0.1" pytest gcsfs
uv pip install --python .venv-jax/bin/python --index-url https://pypi.org/simple \
    --extra-index-url https://download.pytorch.org/whl/cpu torch     # CPU torch: trace boundary only
uv pip install --python .venv-jax/bin/python -e .
# weights: public bucket, no credentials (18 GB + 4.5 MB)
gsutil -m cp -r gs://gemma-data/checkpoints/gemma4-e2b-it /tealab-data/jiaqiliu/models/gemma4/
gsutil cp gs://gemma-data/tokenizers/tokenizer_gemma4.model /tealab-data/jiaqiliu/models/gemma4/
```

```python
from evalrx.models import RuntimeConfig, compose
m = compose("gemma-4-e2b-it", "jax_local", RuntimeConfig(
    device="cpu", dtype="float32", max_new_tokens=64, apply_chat_template=True,
    engine_kwargs={"checkpoint": "/tealab-data/jiaqiliu/models/gemma4/gemma4-e2b-it"}))
trace = m.forward("What is the capital of France?", capture={Capability.ATTENTION, Capability.HIDDEN_STATES, Capability.LOGITS})
```

Measured on the real E2B checkpoint (CPU, 128 cores, `dtype=bfloat16`, 21-token
chat-templated prompt), through `compose(..., "jax_local")`:

| step | result |
|---|---|
| load (local Orbax mirror, bf16 cast) | 75 s, 9.9 GB params (float32: 90 s, 19.8 GB) |
| `generate` greedy, "capital of France" | `Paris`; 23 s first call (compile), 7 s cached; deterministic |
| `generate` sampled (T 0.7, top-p 0.95, seed 3) | a fluent one-sentence answer, 22 s |
| `logprobs` (greedy 8 tokens, teacher-forced pass) | `Paris` with logprob 0.0 and the same top-3 as the Trace logits |
| `forward` with ATTENTION + HIDDEN_STATES + LOGITS | 7 s; 36 hidden `(21, 1536)`, 35 attention `(8, 21, 21)` (rows sum to 1, strictly causal), logits `(21, 262144)`, all bf16 torch |
| `forward` with `CaptureSpec(layers=[0, 17, 34], heads=[0, 3])` | 6 s; 3 hidden, 3 attention `(2, 21, 21)` |
| `unembed_weight()` / `final_norm()` | `(262144, 1536)` bf16; torch RMSNorm with the JAX scale |
| `LogitLensAnalyzer(top_k=3)` | 13 s; `n_layers=36`, `final_norm_applied=True`, last layer's top token `Paris` |

float32 gave the same 7 s per captured forward and a faster cached greedy
generate (3 s versus 7 s) on this CPU; on a GPU the bf16 default halves the
resident weights, so it stands.

Image and audio inputs, measured on the same CPU host on 2026-09-28 with the
full checkpoint (`text_only=False`: 20.5 GB float32 read, LM cast to bf16,
towers kept float32; 32.6 GB peak RSS; the load took 257 s with two other
loads sharing the NFS link, 111 s alone for `load_params`):

| step | result |
|---|---|
| text `forward` after the change | unchanged: 36 hidden, 35 attention, `hidden[0]` bit-identical to `embedder.encode` (max abs diff 0.0) |
| ChartQA image 850 x 600 + question, `encode` | 0.7 s; 297 ids, 266 image tokens, `grids=[(1, 14, 19)]`; tokens read `<bos> <|turn> user \n \n\n <|image> <|image|> x266 <image|> \n\n ...` |
| image `forward` with ATTENTION + HIDDEN_STATES + LOGITS | 17 s first call (compile), 7 s cached; all finite; `image_token_mask` sum 266, `image_spatial_shape` (14, 19), `TokenTypeMap` with 266 image positions; the last position puts 27 % of its attention mass on the image tokens (mean over heads and layers) |
| image `generate` greedy | 26 s (sampler compile included); "10" for a bar-chart count whose gold is 14 (a wrong answer, not a format failure); "Describe this image in one sentence." gives a fluent description of the chart |
| image `logprobs` | 31 s; teacher-forced pass with the masks carried through |
| MMAU speech clip 28.2 s + 4-way question, `encode` | 0.6 s; 821 ids, 705 audio tokens (= the sampler's own count), waveform `(1, 1, 451520)` |
| audio `forward` (all three captures) | 21 s first call; all finite; `audio_token_mask` sum 705; the last position puts 35 % of its attention mass on the audio tokens |
| audio `generate` greedy | 29 s; "A" = gold; "Transcribe the speech in this audio." returns the call-centre dialogue verbatim ("Thank you for calling Sprint. We care about everybody. ..."); the same question WITHOUT the clip answers "Please provide the audio so I can answer your question." |

Through the benchmark harness (`python -m _common.run --modality llm --model
gemma-4-e2b --backend jax_local --dataset bbh_causal_judgement --limit 8
--baseline-only --device cpu --model-path <mirror>`): weights loaded in 98 s,
Stage 0 ran all 8 rows through the chat template at 44 s per case with a
128-token greedy cap. Every answer was coherent, on-topic causal reasoning
that the cap cut off before the Yes/No line, so that run graded 0/8. Rerun
with a 1024-token cap: all 8 answers end in an `Answer: Yes/No` line, 5/8
correct (accuracy 0.625, inside the usable band), 140 s per case on CPU; the
three misses are genuine wrong answers, not parse failures. The torch arm of
the same 8 rows is what the parity item below still needs.

The vlm and alm cells through the same harness on 2026-09-28 (CPU, `--limit 4
--baseline-only --temperature 0`, the tasks' own 64-token caps, the checkpoint
already in the page cache): `vlm/gemma chartqa` loaded the weights with towers
in 52 s and graded 2/4 at 21.8 s per case, every answer short and
chart-grounded ("1.8" for 1.577, "U.S.", "19.7%", "77 and 77" for 77);
`alm/gemma mmau` loaded in 38 s and graded 2/4 at 33.1 s per case (each clip
length recompiles, open item 7), the two misses being a wrong letter and a
leaked `thought` preamble, the same Gemma behaviour the torch cells show.

Open items, in the order they block the acceptance table of section 4:

1. **Stage-0 parity against hf_local** has not been run: the host's eight
   A100s were fully occupied by other jobs and its torch venv predates
   transformers 5.15 (no `gemma4` model type), so the smoke ran on CPU with
   nothing to compare against. The `llm/gemma` cell needs one GPU and a
   transformers >= 5.15 environment for the torch arm.
2. **Chat-template parity** with the HF `chat_template.jinja` (thinking off) is
   inferred from the `dialog` package, not diffed against the HF render.
3. **Thinking on** (`--enable-thinking`) raises on jax_local.
4. **GPU run** of the adapter: done 2026-10-01 inside the `gemma_jax` image on
   one A100 that another job was sharing (`XLA_PYTHON_CLIENT_PREALLOCATE=false`,
   so memory grows on demand; the card showed about 17 GB more than before
   while the vlm run was loading). Four-row Stage 0 smokes through the leaf compose files:
   chartqa 2/4 at 11.9 s/case (the same four answers as the CPU run), mmau 3/4
   at 29.5 s/case (every clip length recompiles, item 7), gsm8k 4/4 at
   12.8 s/case. One mmau row differs from the CPU run: on CPU it opened a
   `thought` preamble and failed, on GPU it answered the letter; greedy
   decoding in bf16 is not device-identical on a near-tie. A first full-chain
   shakedown (chartqa, 16 rows, 2026-10-02) got through baseline, M1 and the
   explore step; it found the thread-safety trap above (fixed) and then
   stopped at M3 because the judge CLI's OAuth session in the mounted
   `claude-home` had expired. With fresh credentials the same 16-row chain
   ran end to end (M1 3 analyzers, explore, 3 hypotheses, 3 M4 probes
   inconclusive at n=8, one L2 `self_consistency_5` candidate tried and
   rejected) and exposed the constant sampler seed above (fixed). The
   full-size chartqa chain (256 rows, 2026-10-02) then ran end to end in
   2 h 10 min on one shared A100: baseline 131/256 (1.1 s/case), M1 three
   analyzers in 294 s, explore 12 min, three hypotheses, one verified
   (failures concentrate on arithmetic over two chart values), fix stage
   1 h 45 min, winner L1 `bind_target_element_first` 66 -> 93 of 128 CONFIRM
   pairs (+0.211, CI +0.117..+0.305, e = 1379, FIXED). No error in any stage
   log apart from the stats tool's "one group empty" on constant signals.
5. Phase 2 (sites, `Model.intervene`, L3b) and the rest of phase 3
   (`encode_variants` for VCD / TCD / AAD, the `WhiteBoxRepairsMixin`,
   `GRADIENTS`, LoRA) and the mkdocs nav entry for
   this page.
6. **Multimodal parity against hf_local** (image / audio rows: same
   `image_token_mask` count, same labels) waits on the same GPU + transformers
   >= 5.15 environment as item 1; the reservation-vs-encoder check was done
   against the library's own encoder, not against the HF processor.
7. **Recompiles per audio length**: `audio_soft_token_counts` is static in the
   library's forward and prefill, so every distinct clip duration recompiles
   both (about 20 s each on this CPU); a dynamic-count merge would remove it.
8. Tool rendering (`render_chat` with `tools`) still raises on jax_local.
