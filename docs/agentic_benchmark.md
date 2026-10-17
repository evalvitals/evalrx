# Adaptive benchmark diagnosis

Use the same datasets, scorers, and final repair validation with an agent that chooses the next diagnostic action:

```bash
python -m evalrx.benchmark.run --modality llm --model gemini-3.6-flash \
  --dataset bbh_causal_judgement
```

The existing judge/coder options apply. Replace modality and dataset for image/audio tasks. Gemini probability support is probed on the actual model; unsupported logprobs and raw logits are never simulated.

Agentic diagnosis is the default. No mode or per-stage enable flags are needed. `--max-actions` optionally changes the default budget of 12 controller decisions. The hidden legacy `--diagnosis-mode fixed` option remains available only to replay historical fixed-pipeline comparisons; current default runs are a different protocol from those historical results.

## Controller scope

The controller can choose catalog probes, request custom probes, run statistics, explore data, cluster failures, propose hypotheses, and test them on EXPLORE. It can gather more evidence after an empty diagnosis, subject to action and tool-call caps. Custom M1 probes and M2 tool generation default on in agentic mode across modalities; The existing `--no-allow-codegen` disables custom M1/repair generation, and `--no-m2-codegen` disables M2 generation.

Repair and surgery are excluded from the benchmark controller's tool registry. After the investigation, the host runs repair search on EXPLORE and validates a frozen winner once on CONFIRM. The controller cannot inspect confirmation feedback and adapt to it. `--held-out` currently requires fixed mode; agentic mode rejects that combination before loading a model.

Agentic repair does not inject built-in floor candidates. The agent can still propose known repair methods. The existing repair tier/capability constraints and maximum repair rounds remain enforced. This mode is bounded adaptive diagnosis, not an unlimited autonomous training loop.

## Triage policy (opt-in, MVP-0)

`--m1-policy triage_active` replaces the diagnosis with a host-side triage controller (`evalrx/eval_agent/probing/`, design in `evalsmith/probing/probing_design_mix.md`). The default `--m1-policy judge` is unchanged.

```bash
python -m evalrx.benchmark.run --modality vlm --model qwen3.5-0.8b-vl --backend hf_local \
  --dataset pope_adversarial --limit 200 --attn-impl eager \
  --m1-policy triage_active --run-tag triage0
```

After Stage 0 and the same EXPLORE/CONFIRM split as the loop, it draws `--triage-n-fail` FAIL cases (default 8) and gold-matched PASS controls from EXPLORE with a nested sampler, then runs a fixed plan exactly as written — `answer_extraction_audit`, `termination_audit` (no continuation), `modality_ablation(mode="swap")`, `attention_knockout(target="gold")` — with no backfill, case cap or generated probes. No judge is called, M2–M5 do not run, and CONFIRM never reaches the controller. Outputs go to `<run>/probing/`: `selection_trace.jsonl`, `cost_ledger.jsonl` (each Stage 0 invocation separate from the actions' new calls), `observations/<action_id>.json`, `split_manifest.json`, and `triage_report.{json,md}` with H/N/X route labels traced to case, field, action and fidelity verdict. Outcomes are typed (`ok`, `unsupported`, `failed`, `unfaithful`) and never written as zeros. Rerunning the same command reads finished actions back by content identity instead of re-running them; `--triage-budget-sec` skips later actions once the wall-clock budget is spent.

White-box edits refuse when their edit-free self-check fails. On Qwen3.5-0.8B in bf16, the attention tap reproduces `--attn-impl eager` exactly but differs from `sdpa` by about 1–2% of the largest logit, above the 1% tolerance, so `attention_knockout` reports `unfaithful` under `sdpa`; use `--attn-impl eager` for edit-based analyzers.

## Failure and progress semantics

M3 exceptions are recorded as structured error exchanges with exception type, message, attempt, and retry status. The runtime automatically permits one additional attempt; valid empty diagnoses are not retried as transport failures. Exhaustion produces `diagnosis_error`, an error diagnosis status, and benchmark exit code 2 rather than a successful `no_hypotheses` result.

Repair progress prints proposal starts, candidate starts/completions, and case counters for ordinary template/spec evaluation, even while detailed result logging is deferred. These operational events contain no case IDs, labels, outputs, or correctness scores. Coded pipelines emit candidate start/end events; their internal calls are not represented as ordinary per-case counters.

## Scoring correction

The multiple-choice parser now recognizes balanced Markdown around a single option (for example `The correct answer is **A**.`). It preserves rejection of uncommitted thought-channel output and ambiguous option lists. This changes labels for affected historical outputs; keep old runs intact and rescore/rerun under the corrected scorer as a new result version. A parser fix alone is not evidence that the model or repair improved.
