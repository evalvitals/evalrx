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

After Stage 0 and the same EXPLORE/CONFIRM split as the loop, it draws `--triage-n-fail` FAIL cases (default 8) and gold-matched PASS controls from EXPLORE with a nested sampler, then runs a fixed plan exactly as written — `answer_extraction_audit`, `termination_audit` (no continuation), `modality_ablation(mode="swap")`, `attention_knockout(target="gold")`, `self_consistency` (5 samples per case), `prompt_contrast` with two neutral wrappers around the verbatim question, `prompt_contrast` with an instruction not to default to a negative answer (yes/no tasks only), `embed_lens(layer=0.9)` and `patchscope(target="gold", media_layers=3)` — with no backfill, case cap or generated probes. No judge is called, M2–M5 do not run, and CONFIRM never reaches the controller. Outputs go to `<run>/probing/`: `selection_trace.jsonl`, `cost_ledger.jsonl` (each Stage 0 invocation separate from the actions' new calls), `observations/<action_id>.json`, `split_manifest.json`, and `triage_report.{json,md}` with H/N/X route labels traced to case, field, action and fidelity verdict. Outcomes are typed (`ok`, `unsupported`, `failed`, `unfaithful`) and never written as zeros. Rerunning the same command reads finished actions back by content identity instead of re-running them; `--triage-budget-sec` skips later actions once the wall-clock budget is spent.

The swap uses counterfactual donors drawn from the whole EXPLORE partition: a case asking the identical prompt about different media whose gold differs from the model's own answer, so a correct answer has to change (relative to the model's answer, not the case's gold, because on a failed case the two differ). On yes/no and letter tasks only such swaps test N; cases without one are reported as untested, so a larger `--limit` raises coverage (on POPE adversarial the questions repeat per object across images). Open-answer tasks also accept content-only swaps, recorded with that assumption.

U (an answer that moves) needs both of its parts: re-sampling at temperature 1.0 (with the run's `--top-p`/`--top-k` on local backends; temperature only on `--backend gemini`, whose runtime merges per-call settings; the `endpoint` backend drops per-call settings, so this part stays untested there), compared with the Stage-0 answer, and the two wrappers, compared with the question re-asked unchanged. Answers are compared by the task's answer key, not graded; an unreadable answer is set aside, and a wrapper with no readable answer is an invalid control. Any readable answer that differs fires U. S is a FAIL case on which H, N (when the case has media) and both U parts ran and none fired; a missing check leaves the case in X, never in S. The threshold instruction, `embed_lens` and `patchscope` are attached to every case as descriptive S readings — whether the instruction fixes errors and what it breaks, how close the case's own question word gets to the media tokens, whether the gold answer ranked first at some layer before the last, and whether the question word is decoded off the media tokens — and summarised for S failures, other failures and stable PASS controls by gold answer. They are not tests and name no mechanism.

On an API model (`--backend gemini`) the same plan runs with the white-box steps recorded as `unsupported` (nothing spent). Each action's cost carries what the provider counted — requests, retries, prompt/completion/thought tokens — and its USD at the dated listed price in `evalrx/models/backends/api/prices.py` (unknown for an unlisted model); Stage 0's usage goes into `baseline.json`. The report ends with next candidates proposed by rule for every route with FAIL cases, each gated on the actual model, so a run without internal state still names which behavioural experiment it could run next and why the internal ones cannot; none is run.

```bash
GEMINI_API_KEY=... python -m evalrx.benchmark.run --modality vlm --model gemini-3.1-flash-lite \
  --dataset pope_adversarial --limit 1000 --concurrency 8 \
  --m1-policy triage_active --triage-n-fail 32 --run-tag triage-api1
```

White-box edits refuse when their edit-free self-check fails, and the check runs on every analysed case: the first failing case refuses the whole batch. On Qwen3.5-0.8B in bf16, the attention tap reproduces `--attn-impl eager` exactly but differs from `sdpa` by 0.8–2.0% of the largest logit depending on the input, around the 1% tolerance, so `attention_knockout` (and the other tap-based analyzers) report `unfaithful` under `sdpa`; use `--attn-impl eager` for edit-based analyzers. This applies to the default `judge` path as well. Action identities include a hash of the analyzer's code, so a rerun after a code change re-executes the action. White-box margins compare the target answer with the strongest token that is not one of its spellings (`"Yes"` and `" Yes"` are different BPE ids that a model can split one answer across), so margins from runs before that change are not comparable.

## Failure and progress semantics

M3 exceptions are recorded as structured error exchanges with exception type, message, attempt, and retry status. The runtime automatically permits one additional attempt; valid empty diagnoses are not retried as transport failures. Exhaustion produces `diagnosis_error`, an error diagnosis status, and benchmark exit code 2 rather than a successful `no_hypotheses` result.

Repair progress prints proposal starts, candidate starts/completions, and case counters for ordinary template/spec evaluation, even while detailed result logging is deferred. These operational events contain no case IDs, labels, outputs, or correctness scores. Coded pipelines emit candidate start/end events; their internal calls are not represented as ordinary per-case counters.

## Scoring correction

The multiple-choice parser now recognizes balanced Markdown around a single option (for example `The correct answer is **A**.`). It preserves rejection of uncommitted thought-channel output and ambiguous option lists. This changes labels for affected historical outputs; keep old runs intact and rescore/rerun under the corrected scorer as a new result version. A parser fix alone is not evidence that the model or repair improved.
