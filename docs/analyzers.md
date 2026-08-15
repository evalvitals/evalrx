# Analyzer Zoo & Model Registry

Reference tables for what's implemented. Discover the same information at
runtime via `evalvitals.list_specs()` and `evalvitals.registry.analyzers`
(see [Discovery](quickstart.md#discovery)).

## Analyzers

| Analyzer | Key | Capability | Modality | Paper | Status |
|---|---|---|---|---|---|
| Attention summary | `attention` | `ATTENTION` | text + image | — | ✓ |
| Attention rollout | `rollout` | `ATTENTION` | text + image | Abnar & Zuidema, 2020 | ✓ |
| Attention sink | `attention_sink` | `ATTENTION` | text + image | [Gu et al. 2023](https://arxiv.org/abs/2309.17453) | ✓ |
| Relative attention | `relative_attention` | `ATTENTION` | image (VLM) | [arXiv:2502.17422](https://arxiv.org/abs/2502.17422) | ✓ |
| RISE | `rise` | `GENERATE` | text | [Petsiuk et al. 2018](https://arxiv.org/abs/1806.07421) | ✓ |
| MM-SHAP | `mm_shap` | `LOGPROBS` | image (VLM) | [arXiv:2212.08158](https://arxiv.org/abs/2212.08158) | ✓ |
| VL-SHAP | `vl_shap` | `LOGPROBS` | image (VLM) | [arXiv:2212.08158](https://arxiv.org/abs/2212.08158) | ✓ |
| Token entropy | `token_entropy` | `LOGITS` | text + image | — | ✓ |
| Logprob entropy | `logprob_entropy` | `LOGPROBS` | text + image | [Kadavath et al. 2022](https://arxiv.org/abs/2207.05221) | ✓ |
| Self-consistency | `self_consistency` | `GENERATE` | text + image | [Wang et al. 2023](https://arxiv.org/abs/2203.11171) | ✓ |
| Verbalized confidence | `verbalized_confidence` | `GENERATE` | text + image | — | ✓ |
| POPE | `pope` | `GENERATE` | image (VLM) | [arXiv:2305.10355](https://arxiv.org/abs/2305.10355) | ✓ |
| CHAIR | `chair` | `GENERATE` | image (VLM) | [arXiv:1809.02156](https://arxiv.org/abs/1809.02156) | ✓ |
| OPERA | `opera` | `ATTENTION` | image (VLM) | [arXiv:2311.17911](https://arxiv.org/abs/2311.17911) | stub |
| VCD | `vcd` | `LOGITS` | image (VLM) | [arXiv:2311.16922](https://arxiv.org/abs/2311.16922) | stub |
| Logit lens | `logit_lens` | `HIDDEN_STATES` | text + image | [nostalgebraist 2020](https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens) | ✓ |
| Tuned lens | `tuned_lens` | `HIDDEN_STATES` | text + image | [Belrose et al. 2023](https://arxiv.org/abs/2303.08112) | stub |
| Grad-CAM | `gradcam` | `GRADIENTS` | image (VLM) | [Selvaraju et al. 2017](https://arxiv.org/abs/1610.02391) | stub |
| Generic attn explain | `generic_attention` | `ATTENTION + GRADIENTS` | text + image | [Chefer et al. 2021](https://arxiv.org/abs/2103.15679) | stub |
| Linear CKA | `cka` | `HIDDEN_STATES` | text + image | [Kornblith et al. 2019](https://arxiv.org/abs/1905.00414) | ✓ |
| Linear probe | `linear_probe` | `HIDDEN_STATES` | text + image | — | stub |
| Causal trace | `causal_trace` | `HIDDEN_STATES` | text + image | [Meng et al. 2022](https://arxiv.org/abs/2202.05262) | stub |
| Loop detect | `loop_detect` | Trajectory | agent | — | ✓ |
| Ignored obs | `ignored_obs` | Trajectory | agent | — | ✓ |
| First-error judge | `first_error_judge` | Trajectory | agent | [Zhang et al. 2024](https://arxiv.org/abs/2406.14855) | ✓ |
| Trajectory rubric | `trajectory_rubric` | Trajectory + judge | agent | [MAST, Cemri et al. 2025](https://arxiv.org/abs/2503.13657) | ✓ |
| Counterfactual | `counterfactual` | Trajectory | agent | Pearl 2000 | ✓ |
| Reliability probe | `reliability_probe` | re-runs (injected) | agent | pass@k vs pass^k | ✓ |
| Tool Shapley | `tool_shap` | re-runs (injected) | agent | [AgentSHAP / TokenSHAP](https://arxiv.org/abs/2407.10114) | ✓ |

## Model Registry

From `list_specs()`:

| Key | Family | Type | Notes |
|---|---|---|---|
| `qwen2.5-7b-instruct` | Qwen2 | LLM | reference checkpoint |
| `qwen3-4b` | Qwen3 | LLM | reasoning, small smoke-test |
| `qwen3-8b` | Qwen3 | LLM | reasoning |
| `qwen3-30b-a3b` | Qwen3-MoE | LLM | MoE |
| `deepseek-v3` | DeepSeek-V3 | LLM | MoE + MLA |
| `llama-3.1-8b-instruct` | Llama | LLM | — |
| `gemma-3-1b-it` | Gemma3 | LLM | text-only |
| `qwen3-vl-4b-instruct` | Qwen3-VL | VLM | small smoke-test |
| `qwen2.5-vl-7b-instruct` | Qwen2.5-VL | VLM | reference for relative-attention |
| `qwen2-vl-7b-instruct` | Qwen2-VL | VLM | — |
| `qwen3-vl-8b-instruct` | Qwen3-VL | VLM | — |
| `glm-4.5v` | GLM-MoE | VLM | MoE + reasoning, 106B |
| `glm-4.1v-9b-thinking` | GLM | VLM | reasoning |
| `kimi-vl-a3b-thinking` | Kimi-VL | VLM | MoE + MLA + reasoning |
| `llama-4-scout` | Llama4 | VLM | MoE |
| `step-1o-vision` | Step | VLM | API-only |
