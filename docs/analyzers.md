# Analyzer Zoo & Model Registry

<!-- The tables between "BEGIN GENERATED" / "END GENERATED" markers are generated from
the registries by tools/gen_analyzer_docs.py. Edit the prose and the hand-written cells
(analyzer name and Paper, model Notes) freely, then run `python tools/gen_analyzer_docs.py`;
every other column is overwritten on each run. -->

Reference tables for every registered analyzer and model spec. Discover the same
information at runtime via `evalrx.list_specs()` and `evalrx.registry.analyzers`
(see [Discovery](quickstart.md#discovery)).

## Analyzers

Analyzers are grouped the way the code is: one section per subpackage of
`evalrx/analyzers/`. Each analyzer is keyed by the capability it needs from the
model, not by a black-box / white-box label.

<!-- BEGIN GENERATED: analyzer-index -->
| Category | Package | Analyzers | Stubs |
|---|---|---|---|
| [Perturbation](#perturbation) | `perturbation/` | 9 | 0 |
| [Uncertainty](#uncertainty) | `uncertainty/` | 6 | 0 |
| [Hallucination](#hallucination) | `hallucination/` | 5 | 2 |
| [Attention](#attention) | `attention/` | 4 | 0 |
| [Attribution](#attribution) | `attribution/` | 2 | 0 |
| [Lens](#lens) | `lens/` | 4 | 0 |
| [Patching](#patching) | `patching/` | 3 | 0 |
| [Geometry](#geometry) | `geometry/` | 2 | 0 |
| [Reasoning](#reasoning) | `reasoning/` | 7 | 0 |
| [Agent](#agent) | `agent/` | 7 | 0 |
| **Total** | | **49** | **2** |
<!-- END GENERATED -->

How to read the columns:

- **Key**: the registered name, as in `evalrx.registry.analyzers.get(key)`.
- **Requires**: capabilities the model must provide (`Analyzer.requires`).
  `trajectories` means the batch must carry agent trajectories
  (`requires_trajectories`). A dash means nothing is required of the model: the
  analyzer reads stored outputs, or re-runs cases through an injected runner or judge.
- **Modality**: `applies_to_modalities`. It is matched by intersection with the
  model's modalities, so `text` covers every text-capable model, VLMs and omni
  models included. **Bold** slots must also be filled in the batch
  (`requires_modalities`); selection drops the analyzer when none of them is.
- **Status**: ✓ = implemented; *stub* = registered, but `_run` raises
  `NotImplementedError` (Stage 2).

### Perturbation

`evalrx/analyzers/perturbation/`: re-query the model on masked, ablated or
rewritten inputs and measure what moves. Black-box (`GENERATE` / `LOGPROBS`), so
these run on API models too; the cost is the number of extra model calls per case.

<!-- BEGIN GENERATED: analyzers/perturbation -->
| Analyzer | Key | Requires | Modality | Paper | Status |
|---|---|---|---|---|---|
| RISE | `rise` | `GENERATE` | text | [Petsiuk et al. 2018](https://arxiv.org/abs/1806.07421) | ✓ |
| VL-SHAP | `vl_shap` | `LOGPROBS` | **image** | [Lundberg & Lee 2017](https://arxiv.org/abs/1705.07874); [Parcalabescu & Frank 2022](https://arxiv.org/abs/2212.08158) | ✓ |
| MM-SHAP | `mm_shap` | `LOGPROBS` | text + **image** + **audio** + **video** | [Parcalabescu & Frank 2022](https://arxiv.org/abs/2212.08158) | ✓ |
| Modality ablation | `modality_ablation` | `GENERATE` | text + **image** + **audio** + **video** | — | ✓ |
| Prompt contrast | `prompt_contrast` | `GENERATE` | text + image | — | ✓ |
| MC format sensitivity | `format_sensitivity` | `GENERATE` | text + image | [Zheng et al. 2024](https://arxiv.org/abs/2309.03882) | ✓ |
| CoT faithfulness | `cot_faithfulness` | `GENERATE` | text + image | [Lanham et al. 2023](https://arxiv.org/abs/2307.13702) | ✓ |
| Context Shapley | `context_shap` | `GENERATE` | text + image | [Goldshmidt & Horovicz 2024](https://arxiv.org/abs/2407.10114); [Cohen-Wang et al. 2024](https://arxiv.org/abs/2409.00729) | ✓ |
| Perturbation battery | `perturbation_battery` | `GENERATE` | text + image | [Mirzadeh et al. 2025](https://arxiv.org/abs/2410.05229) | ✓ |
<!-- END GENERATED -->

### Uncertainty

`evalrx/analyzers/uncertainty/`: cheap signals of how sure the model was, read
from logits or logprobs, from repeated samples, or from what the model says about
itself, and checked against PASS/FAIL labels (calibration, coverage gap).

<!-- BEGIN GENERATED: analyzers/uncertainty -->
| Analyzer | Key | Requires | Modality | Paper | Status |
|---|---|---|---|---|---|
| Token entropy | `token_entropy` | `LOGITS` | text | [Malinin & Gales 2021](https://arxiv.org/abs/2002.07650) | ✓ |
| Logprob entropy | `logprob_entropy` | `LOGPROBS` | text + image | [Kadavath et al. 2022](https://arxiv.org/abs/2207.05221) | ✓ |
| Self-consistency | `self_consistency` | `GENERATE` | text + image | [Wang et al. 2023](https://arxiv.org/abs/2203.11171) | ✓ |
| Verbalized confidence | `verbalized_confidence` | `GENERATE` | text + image | [Lin et al. 2022](https://arxiv.org/abs/2205.14334) | ✓ |
| Calibration | `calibration` | `GENERATE` | text + image | [Guo et al. 2017](https://arxiv.org/abs/1706.04599) | ✓ |
| Coverage–verification gap | `coverage_verification_gap` | `GENERATE` | text + image | [Brown et al. 2024](https://arxiv.org/abs/2407.21787) | ✓ |
<!-- END GENERATED -->

### Hallucination

`evalrx/analyzers/hallucination/`: the dominant MLLM failure mode. Black-box
object probes (POPE, CHAIR) and sampled-passage consistency for text (SelfCheck),
plus decoding-side white-box diagnostics (OPERA, VCD).

<!-- BEGIN GENERATED: analyzers/hallucination -->
| Analyzer | Key | Requires | Modality | Paper | Status |
|---|---|---|---|---|---|
| POPE | `pope` | `GENERATE` | **image** | [Li et al. 2023](https://arxiv.org/abs/2305.10355) | ✓ |
| CHAIR | `chair` | `GENERATE` | **image** | [Rohrbach et al. 2018](https://arxiv.org/abs/1809.02156) | ✓ |
| SelfCheck consistency | `selfcheck_consistency` | `GENERATE` | text + image | [Manakul et al. 2023](https://arxiv.org/abs/2303.08896) | ✓ |
| OPERA | `opera` | `ATTENTION` | **image** | [Huang et al. 2024](https://arxiv.org/abs/2311.17911) | stub |
| VCD | `vcd` | `LOGITS` | **image** | [Leng et al. 2024](https://arxiv.org/abs/2311.16922) | stub |
<!-- END GENERATED -->

### Attention

`evalrx/analyzers/attention/`: read the attention maps of a white-box model
(`ATTENTION`, HF eager attention). The subpackage imports torch at load time, so
on the light (API-only) install it is skipped and these analyzers are not offered.

<!-- BEGIN GENERATED: analyzers/attention -->
| Analyzer | Key | Requires | Modality | Paper | Status |
|---|---|---|---|---|---|
| Attention summary | `attention` | `ATTENTION` | text | — | ✓ |
| Attention rollout | `attention_rollout` | `ATTENTION` | text + image | [Abnar & Zuidema 2020](https://arxiv.org/abs/2005.00928) | ✓ |
| Attention sink | `attention_sink` | `ATTENTION` | text + image | [Xiao et al. 2024](https://arxiv.org/abs/2309.17453); [Gu et al. 2025](https://arxiv.org/abs/2410.10781) | ✓ |
| Relative attention | `relative_attention` | `ATTENTION` | **image** + **video** | ["MLLMs Know Where to Look", 2025](https://arxiv.org/abs/2502.17422) | ✓ |
<!-- END GENERATED -->

### Attribution

`evalrx/analyzers/attribution/`: gradient-based saliency (`GRADIENTS`, a backward
pass through a white-box model: `hf_local` keeps the autograd graph when asked with
`CaptureSpec(with_grad=True)`). Both explain the next-token logit margin and map the
result back onto the image's patch grid, with overlays on the case photo. Grad-CAM
reads the decoder's image-token states rather than the vision tower.

<!-- BEGIN GENERATED: analyzers/attribution -->
| Analyzer | Key | Requires | Modality | Paper | Status |
|---|---|---|---|---|---|
| Grad-CAM | `gradcam` | `GRADIENTS` + `LOGITS` | **image** + **video** | [Selvaraju et al. 2017](https://arxiv.org/abs/1610.02391) | ✓ |
| Generic attention explainability | `generic_attention` | `ATTENTION` + `GRADIENTS` + `LOGITS` | text + image | [Chefer et al. 2021](https://arxiv.org/abs/2103.15679) | ✓ |
<!-- END GENERATED -->

### Lens

`evalrx/analyzers/lens/`: project intermediate hidden states through the
unembedding (`HIDDEN_STATES`) to see at which layer a prediction forms, and where
premature layers disagree with the final one; or project each attention head's own
write, to see which heads write the answer.

<!-- BEGIN GENERATED: analyzers/lens -->
| Analyzer | Key | Requires | Modality | Paper | Status |
|---|---|---|---|---|---|
| Logit lens | `logit_lens` | `HIDDEN_STATES` | text + image | [nostalgebraist 2020](https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens) | ✓ |
| Tuned lens | `tuned_lens` | `HIDDEN_STATES` | text + image | [Belrose et al. 2023](https://arxiv.org/abs/2303.08112) | ✓ |
| Layer contrast | `layer_contrast` | `HIDDEN_STATES` | text + image | [Chuang et al. 2024](https://arxiv.org/abs/2309.03883); [Wang et al. 2025](https://arxiv.org/abs/2410.11779) | ✓ |
| Head logit attribution | `head_attribution` | `HIDDEN_STATES` + `LOGITS` | text + image | [Elhage et al. 2021](https://transformer-circuits.pub/2021/framework/index.html); [Wang et al. 2023](https://arxiv.org/abs/2211.00593) | ✓ |
<!-- END GENERATED -->

### Patching

`evalrx/analyzers/patching/`: causal interventions. Write a clean run's
activations into a corrupted run to localise where information is used
(`HIDDEN_STATES`, read and write hooks), or cut attention edges and measure what
the answer loses. These act on the live torch module of a local `hf_local` model
through `evalrx/analyzers/_internals/` (an attention function registered with
transformers, and decoder-layer pre-hooks), and verify, once per model, that the
edit-free path reproduces the native logits. Hybrid stacks (Qwen3.5, Qwen3-Next:
softmax + gated-delta-rule linear attention) are edited on every layer: a linear
layer's text positions get its output re-run on the sequence without the media
tokens (`_internals/linear_mix.py`). Latent / unverified attention is refused.
Qwen3.5 needs transformers 5 (the `vllm35` environment), not the `[local]` extra.

<!-- BEGIN GENERATED: analyzers/patching -->
| Analyzer | Key | Requires | Modality | Paper | Status |
|---|---|---|---|---|---|
| Causal trace | `causal_trace` | `HIDDEN_STATES` + `LOGITS` | **image** + **audio** + **video** | [Meng et al. 2022](https://arxiv.org/abs/2202.05262); [Basu et al. 2024](https://arxiv.org/abs/2406.04236) | ✓ |
| Attention knockout | `attention_knockout` | `ATTENTION` + `LOGITS` | **image** + **audio** + **video** | [Geva et al. 2023](https://arxiv.org/abs/2304.14767); [Zhang et al. 2025](https://arxiv.org/abs/2411.18620) | ✓ |
| Attention sensitivity | `attention_sensitivity` | `ATTENTION` + `LOGITS` | **image** + **audio** + **video** | [Liu et al. 2024 (PAI)](https://arxiv.org/abs/2407.21771) | ✓ |
<!-- END GENERATED -->

### Geometry

`evalrx/analyzers/geometry/`: representation geometry over hidden states
(`HIDDEN_STATES`). Cross-modal CKA readings were validated on CLIP-style two-tower
encoders; on a decoder-only MLLM, read them as within-stream layer similarity.

<!-- BEGIN GENERATED: analyzers/geometry -->
| Analyzer | Key | Requires | Modality | Paper | Status |
|---|---|---|---|---|---|
| Linear CKA | `cka` | `HIDDEN_STATES` | text + image | [Kornblith et al. 2019](https://arxiv.org/abs/1905.00414) | ✓ |
| Linear probe | `linear_probe` | `HIDDEN_STATES` | text + image | [Alain & Bengio 2016](https://arxiv.org/abs/1610.01644) | ✓ |
<!-- END GENERATED -->

### Reasoning

`evalrx/analyzers/reasoning/`: text-chain diagnostics, all black-box. Read the
two hygiene probes first: `answer_extraction_audit` (is the FAIL label real, or
did the grader miss the answer?) and `termination_audit` (did the generation stop
cleanly, or hit the token budget or a repetition loop?). Both produce confounds
that mimic every other probe's "bad" column. The rest decompose *how* a chain
failed.

<!-- BEGIN GENERATED: analyzers/reasoning -->
| Analyzer | Key | Requires | Modality | Paper | Status |
|---|---|---|---|---|---|
| Answer-extraction audit | `answer_extraction_audit` | — | text + image | [Lightman et al. 2024](https://arxiv.org/abs/2305.20050) | ✓ |
| Termination audit | `termination_audit` | `GENERATE` | text + image | [Holtzman et al. 2020](https://arxiv.org/abs/1904.09751) | ✓ |
| Arithmetic audit | `arith_audit` | — | text + image | [Cobbe et al. 2021](https://arxiv.org/abs/2110.14168) | ✓ |
| Self-repair | `self_repair` | `GENERATE` | text + image | [Huang et al. 2024](https://arxiv.org/abs/2310.01798) | ✓ |
| Step rollout value | `step_rollout_value` | `GENERATE` | text + image | [Wang et al. 2024 (Math-Shepherd)](https://arxiv.org/abs/2312.08935) | ✓ |
| Knowledge / reasoning split | `knowledge_reasoning_split` | `GENERATE` | text + image | [Press et al. 2023](https://arxiv.org/abs/2210.03350) | ✓ |
| Contamination | `contamination_score` | `GENERATE` | text + image | [Golchin & Surdeanu 2024](https://arxiv.org/abs/2308.08493) | ✓ |
<!-- END GENERATED -->

### Agent

`evalrx/analyzers/agent/`: read multi-step agent runs (`Trajectory`) rather than
single turns. Ordered cheapest first: deterministic heuristics, then LLM judges,
then probes that re-run the agent through an injected runner (`counterfactual`,
`reliability_probe`, `tool_shap`). Trajectories are usually loaded from disk with
`model=None`, which is why most of these require nothing of the model.

<!-- BEGIN GENERATED: analyzers/agent -->
| Analyzer | Key | Requires | Modality | Paper | Status |
|---|---|---|---|---|---|
| Loop detect | `loop_detect` | trajectories | text + image | [MAST, Cemri et al. 2025](https://arxiv.org/abs/2503.13657) | ✓ |
| Ignored observation | `ignored_obs` | trajectories | text + image | [MAST, Cemri et al. 2025](https://arxiv.org/abs/2503.13657) | ✓ |
| First-error judge | `first_error_judge` | trajectories | text + image | [Who&When, Zhang et al. 2025](https://arxiv.org/abs/2505.00212) | ✓ |
| Trajectory rubric | `trajectory_rubric` | trajectories | text + image | [MAST, Cemri et al. 2025](https://arxiv.org/abs/2503.13657) | ✓ |
| Counterfactual replay | `counterfactual` | `TOOL_CALLS` + trajectories | text + image | Pearl 2000 | ✓ |
| Reliability probe | `reliability_probe` | — | text + image | pass@k vs pass^k | ✓ |
| Tool Shapley | `tool_shap` | trajectories | text + image | [AgentSHAP / TokenSHAP](https://arxiv.org/abs/2407.10114) | ✓ |
<!-- END GENERATED -->

## Model Registry

From `list_specs()` (`evalrx/specs.py`). **Modalities** are derived from the
components a spec declares (vision tower → image, audio tower → audio, video) and
are what analyzer selection matches on. **Traits** come from the spec flags:
`MoE` (`is_moe`), `reasoning` (`is_reasoning`), `tool calling` (`tool_calling`: the
chat template renders tools, which grants `TOOL_CALLS` on the local backend) and
`JAX` (also runnable on the `jax_local` backend).

### Text-only LLMs

<!-- BEGIN GENERATED: models/text -->
| Key | Family | Modalities | Traits | Notes |
|---|---|---|---|---|
| `qwen2.5-7b-instruct` | `qwen2` | text | tool calling | reference checkpoint |
| `qwen2.5-14b-instruct` | `qwen2` | text | tool calling | — |
| `qwen2.5-32b-instruct` | `qwen2` | text | tool calling | — |
| `qwen2.5-72b-instruct` | `qwen2` | text | tool calling | — |
| `qwen3-4b` | `qwen3` | text | reasoning, tool calling | small smoke-test checkpoint; thinking off |
| `qwen3-8b` | `qwen3` | text | reasoning, tool calling | — |
| `qwen3-14b` | `qwen3` | text | reasoning, tool calling | — |
| `qwen3-32b` | `qwen3` | text | reasoning, tool calling | — |
| `qwen3-30b-a3b` | `qwen3_moe` | text | MoE, tool calling | — |
| `qwen3-235b-a22b` | `qwen3_moe` | text | MoE, reasoning, tool calling | multi-GPU / FP8 |
| `qwen3.5-2b` | `qwen3_5` | text | reasoning, tool calling | hybrid linear / full attention stack |
| `qwen3.5-4b` | `qwen3_5` | text | reasoning, tool calling | hybrid linear / full attention stack |
| `qwen3.5-9b` | `qwen3_5` | text | reasoning, tool calling | hybrid linear / full attention stack |
| `deepseek-v3` | `deepseek_v3` | text | MoE, tool calling | MLA |
| `llama-3.1-8b-instruct` | `llama` | text | tool calling | also loads in TransformerLens |
| `gemma-3-1b-it` | `gemma3` | text | — | tied embeddings |
| `nemotron-3-nano-4b` | `nemotron_h` | text | reasoning | hybrid Mamba / attention stack |
| `nemotron-3-nano-4b-fp8` | `nemotron_h` | text | reasoning | endpoint only (FP8 export) |
<!-- END GENERATED -->

### Vision-language models

<!-- BEGIN GENERATED: models/vision -->
| Key | Family | Modalities | Traits | Notes |
|---|---|---|---|---|
| `qwen2-vl-7b-instruct` | `qwen2_vl` | text + image | tool calling | — |
| `qwen2.5-vl-3b-instruct` | `qwen2_5_vl` | text + image | tool calling | — |
| `qwen2.5-vl-7b-instruct` | `qwen2_5_vl` | text + image | tool calling | reference for relative attention |
| `qwen2.5-vl-32b-instruct` | `qwen2_5_vl` | text + image | tool calling | — |
| `qwen2.5-vl-72b-instruct` | `qwen2_5_vl` | text + image | tool calling | — |
| `qwen3-vl-2b-instruct` | `qwen3_vl` | text + image | tool calling | — |
| `qwen3-vl-4b-instruct` | `qwen3_vl` | text + image | tool calling | small smoke-test checkpoint |
| `qwen3-vl-8b-instruct` | `qwen3_vl` | text + image | tool calling | — |
| `qwen3-vl-30b-a3b-instruct` | `qwen3_vl_moe` | text + image | MoE, tool calling | multi-GPU / FP8 |
| `qwen3-vl-235b-a22b-instruct` | `qwen3_vl_moe` | text + image | MoE, tool calling | multi-GPU / FP8 |
| `qwen3.5-2b-vl` | `qwen3_5` | text + image | reasoning, tool calling | hybrid linear / full attention stack |
| `qwen3.5-4b-vl` | `qwen3_5` | text + image | reasoning, tool calling | hybrid linear / full attention stack |
| `qwen3.5-9b-vl` | `qwen3_5` | text + image | reasoning, tool calling | hybrid linear / full attention stack |
| `glm-4.1v-9b-thinking` | `glm4v` | text + image | reasoning, tool calling | — |
| `glm-4.5v` | `glm4v_moe` | text + image | MoE, reasoning, tool calling | 106B; multi-GPU / FP8 |
| `kimi-vl-a3b-thinking` | `kimi_vl` | text + image | MoE, reasoning | MLA; remote code |
| `llama-4-scout` | `llama4` | text + image | MoE | — |
| `llava-1.5-7b-hf` | `llava` | text + image | — | reference architecture for ViCrop |
| `instructblip-vicuna-7b` | `instructblip` | text + image | — | reference Q-Former architecture |
<!-- END GENERATED -->

### Audio and omni models

<!-- BEGIN GENERATED: models/omni -->
| Key | Family | Modalities | Traits | Notes |
|---|---|---|---|---|
| `qwen2-audio-7b-instruct` | `qwen2_audio` | text + audio | — | audio in, text out |
| `qwen2.5-omni-7b` | `qwen2_5_omni` | text + image + audio + video | tool calling | — |
| `qwen3-omni-30b-a3b-instruct` | `qwen3_omni_moe` | text + image + audio + video | MoE, tool calling | — |
| `qwen3-omni-30b-a3b-thinking` | `qwen3_omni_moe` | text + image + audio + video | MoE, reasoning, tool calling | — |
| `qwen3-omni-30b-a3b-captioner` | `qwen3_omni_moe` | text + audio | MoE | audio captioning only |
| `gemma-4-e2b-it` | `gemma4` | text + image + audio | reasoning, JAX | thinking off |
| `gemma-4-e4b-it` | `gemma4` | text + image + audio | reasoning, JAX | thinking off |
| `gemma-4-12b-it` | `gemma4` | text + image + audio | reasoning | thinking off |
| `nemotron-3-nano-omni-30b-a3b-reasoning` | `nemotron_h_omni` | text + image + audio + video | MoE, reasoning | BF16 needs two 48 GB GPUs |
| `nemotron-3-nano-omni-30b-a3b-reasoning-fp8` | `nemotron_h_omni` | text + image + audio + video | MoE, reasoning | endpoint only (FP8 export) |
<!-- END GENERATED -->

### API-only models

Closed weights served through an API backend: no internals, so only the
black-box analyzers apply. The Gemini specs use `google-genai` with
`GEMINI_API_KEY`.

<!-- BEGIN GENERATED: models/api -->
| Key | Family | Modalities | Traits | Notes |
|---|---|---|---|---|
| `gemini-2.5-flash-lite` | `gemini` | text + image + audio + video | reasoning | — |
| `gemini-2.5-flash` | `gemini` | text + image + audio + video | reasoning | — |
| `gemini-2.5-pro` | `gemini` | text + image + audio + video | reasoning | — |
| `gemini-3.1-flash-lite` | `gemini` | text + image + audio + video | reasoning | — |
| `gemini-3.1-pro-preview` | `gemini` | text + image + audio + video | reasoning | — |
| `gemini-3.5-flash-lite` | `gemini` | text + image + audio + video | reasoning | — |
| `gemini-3.5-flash` | `gemini` | text + image + audio + video | reasoning | — |
| `gemini-3.6-flash` | `gemini` | text + image + audio + video | reasoning | — |
| `gemini-3.7-flash` | `gemini` | text + image + audio + video | reasoning | — |
| `step-1o-vision` | `step` | text | — | spec declares no vision component, so it matches as text-only |
<!-- END GENERATED -->

## Keeping this page current

The tables above are generated from the analyzer registry and `evalrx/specs.py` by
`tools/gen_analyzer_docs.py`. After adding an analyzer or a model spec, run:

```bash
python tools/gen_analyzer_docs.py           # add new rows, refresh the derived columns
python tools/gen_analyzer_docs.py --check   # exit 1 if the page is stale
```

A new key gets a placeholder row in its section; fill in its name and Paper (or
Notes) by hand. A new `analyzers/` subpackage or model group needs its own heading
and marker pair first. The test suite runs the same check
(`tests/test_analyzers/test_analyzer_docs.py`).
