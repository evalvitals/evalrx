# Roadmap

EvalRX is currently in alpha. The repository contains both implemented
surfaces and planned interfaces.

## Implemented

| Area | Status |
|---|---|
| Core contracts | `Model`, `Analyzer`, `Result`, `FailureCase`, registry, pipeline, experiment. |
| Model specs | Torch-free registry in `evalrx.specs` (16 specs: Qwen3/VL/Omni, DeepSeek, GLM, Kimi, Llama, Gemma, Step). |
| Backend composition | `compose(spec, backend, want=...)` with early capability negotiation. |
| Public `wrap()` on-ramp | Wrap any loaded HF causal LM + tokenizer; capabilities auto-inferred; attention fix-up applied. |
| Capability matching | Analyzer discovery and early backend negotiation. |
| Attention analysis | `AttentionAnalyzer`, `AttentionRolloutAnalyzer`, `AttentionSinkAnalyzer`, `RelativeAttentionAnalyzer` (VLM). |
| VLM forward capture | `HFLocalModel._vlm_forward`: image + text; `image_token_mask` + `image_spatial_shape` in `Trace.extras`. |
| Uncertainty analysis | `TokenEntropyAnalyzer` (LOGITS), `LogprobEntropyAnalyzer` (LOGPROBS), `SelfConsistencyAnalyzer`, `VerbalizedConfidenceAnalyzer`. |
| Hallucination analysis | `POPEAnalyzer`, `CHAIRAnalyzer` (black-box, GENERATE-only). |
| Shapley attribution | `MMShapAnalyzer` (text + image), `VLShapAnalyzer` (image regions). |
| Lens analysis | `LogitLensAnalyzer` (hidden states → vocabulary projections). |
| Geometry | `CKAAnalyzer` (linear CKA over residual stream). |
| Agent analyzers | `LoopDetector`, `IgnoredObservationDetector`, `FirstErrorJudge`, `CounterfactualReplay`. |
| Statistics | `compare()` (McNemar + e-value + clustered bootstrap CI), `compare_multiple()` (Friedman + Nemenyi), `ebh()`, `kendall_tau`, `stratified_subset`. |
| Eval agent — pre-registered A/B | `EvalOrchestrator`, `DataSplit`, `PreregisteredHypothesis`, `PreregistrationLog`. |
| **AutoDiagnoseLoop** | **M1→M5 controller with hint-driven probe focus, prior-cycle context, and `resolved` fast-exit.** |
| **ExperimentWriter** | **6-phase CodeAgent: blueprint → sequential multi-file generation (CodeMem AST summaries) → hard-validate → exec-fix (targeted traceback repair) → tree-search → review.  CLI agent backends: `codex`, `claude_code`, `opencode`, `gemini_cli`, `kimi_cli`.** |
| **ExperimentSandbox** | **`run_project()` for multi-file projects; path traversal protection; harness injection; cleanup-on-success; `SandboxProtocol` + `create_sandbox()` factory.** |
| **ExperimentGitManager** | **Git-native run versioning: `eval/{run_id}` branches; commit on resolve; `git reset --hard` on discard.** |
| **EvolutionStore** | **JSONL lesson accumulation with 30-day half-life time decay; `extract_lessons(report)`; `build_overlay()` for prompt injection.** |
| **JsonlStore** | **Durable JSONL-backed `Store`; hypotheses serialized via `hypothesis_to_dict`/`from_dict`; survives process restart.** |
| **Run-directory infrastructure** | **Atomic checkpoints (`checkpoint.json`), heartbeat (`heartbeat.json`), `AutoDiagnoseLoop.resume()`, auto-created `EvolutionStore`.** |
| **VLM image-attention rule** | **M2 derives `image_token_attention_ratio` from `top_attended_tokens`; fires medium-severity finding when VLM ignores image tokens.** |
| **RunContext** | **Single owner of a run's output directory (`report/`, `figures/`, `artifacts/`, `experiments/`, `fixes/`, `manifest.json`, auto-generated `README.txt`); per-trial folders (`new_trial()`) for fix candidates and M5 experiments.** |
| **FixAgent** | **Tiered post-loop repair (L1 prompt → L2 scaffold → L3a/b internals → L4 parameter space); paired McNemar + e-value validation; `max_repair_rounds` feedback-driven retry with candidate dedup; `loop.run_fix(auto_escalate=True)` steps the tier ladder.** |
| **Result image overlays** | **`RelativeAttentionResult.overlay()`/`.image_overlays()` — CAM-style heatmap-on-source-image blending; duck-typed hook picked up by `RunLogger` into `figures/`.** |
| Contract tests | Parametrized pyod-style suite covering all 26 registered analyzers (554 unit tests). |

## Stage 2 stubs (registered, `_run` raises `NotImplementedError`)

| Analyzer | Key | Blocker |
|---|---|---|
| Tuned lens | `tuned_lens` | requires per-model trained translators |
| Causal trace | `causal_trace` | needs nnsight read+write hooks |
| Linear probe | `linear_probe` | needs labeled hidden-state pairs |
| GradCAM | `gradcam` | needs backward pass through vision tower |
| Generic attention explainability | `generic_attention` | needs Chefer relevance propagation |
| OPERA | `opera` | needs decode-loop control |
| VCD | `vcd` | needs contrastive decoding over distorted image |

## Planned (Stage 2+)

| Area | Planned Scope |
|---|---|
| vLLM backend | Offline batch inference, high-throughput logprob collection. |
| VLM vision-tower internals | Cross-modal hidden states, vision-tower attention maps. |
| LLM-backed hypothesis generation | `HypothesisGenerator.propose()` / `.mutate()` backed by the judge model. |
| Case synthesis | `make_cases(spec)` — agent-driven adversarial case construction. |
| SQLite store | SQLite backend for `Store` with semantic recall (JSONL `JsonlStore` already implemented). |
| RISE | Requires user-supplied `score_fn`; improve discoverability. |
| Omni model forward capture | Audio + video tower internals (needs `transformers≥5.2.0`). |

## Near-Term Priorities

1. Keep the public API narrow and stable.
2. Implement Stage-2 stubs incrementally (causal trace, tuned lens).
3. Add SQLite backend for `Store` with semantic recall.
4. `HypothesisGenerator.propose()` / `.mutate()` backed by the judge model (LLM-driven hypothesis generation).
5. Add `FDR` control across M3 hypotheses in one cycle.
