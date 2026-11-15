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

## Failure and progress semantics

M3 exceptions are recorded as structured error exchanges with exception type, message, attempt, and retry status. The runtime automatically permits one additional attempt; valid empty diagnoses are not retried as transport failures. Exhaustion produces `diagnosis_error`, an error diagnosis status, and benchmark exit code 2 rather than a successful `no_hypotheses` result.

Repair progress prints proposal starts, candidate starts/completions, and case counters for ordinary template/spec evaluation, even while detailed result logging is deferred. These operational events contain no case IDs, labels, outputs, or correctness scores. Coded pipelines emit candidate start/end events; their internal calls are not represented as ordinary per-case counters.

## Scoring correction

The multiple-choice parser now recognizes balanced Markdown around a single option (for example `The correct answer is **A**.`). It preserves rejection of uncommitted thought-channel output and ambiguous option lists. This changes labels for affected historical outputs; keep old runs intact and rescore/rerun under the corrected scorer as a new result version. A parser fix alone is not evidence that the model or repair improved.

## Live validation: 2026-10-12

Gemini 3.6 Flash was run through the full default agentic workflow with Claude Opus 5/high as judge and coder, concurrency 1, and the existing full frozen manifests (BBH causal judgement 187, ChartQA 128, MMAU 128). No mode or stage-skip flags were supplied.

| Dataset | Confirmation correct before → after | Repairs / breaks | e-value (threshold 20) | Result |
| --- | --- | --- | --- | --- |
| BBH causal judgement | 68/94 → 66/94 | 4 / 6 | 0.4433 | Regression; repair rejected |
| ChartQA | 45/64 → 57/64 | 13 / 1 | 78.0190 | Validated repair |
| MMAU | 44/64 → 50/64 | 6 / 0 | 9.1429 | Improvement below the validation threshold |

All three final attempts exited 0. Offline rescoring reproduced these counts, and the exploration/confirmation IDs were disjoint. The original manifest hashes and frozen source snapshots were verified unchanged. Candidate selection tried 13, 10, and 15 candidates respectively, then tested one frozen winner per confirmation set. These runs establish a repair for ChartQA; they do not establish universal repair success or a controlled improvement over the historical fixed workflow.

Live testing caught a duplicate LLM `floor_candidates` argument; it was fixed and BBH was restarted with the original failed attempt preserved. MMAU encountered a real M3 timeout and recovered through bounded retry. Gemini returned `Logprobs is not enabled for this model`; no probability data was fabricated.

The vision/audio source snapshots correspond to `013fb43`; the corrected BBH retry corresponds to `22d4f85`. The later exploration artifact/M3-context fix (`7285788`) is covered by the final **426 passing regression tests**, but was not included in another full live rerun. Local run artifacts, commands, hashes, and offline audits are under `examples/benchmark/smoke_checks/agentic-default-20261012/`; raw run artifacts are not committed.
