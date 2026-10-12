# Adaptive benchmark diagnosis

Use the same datasets, scorers, and final repair validation with an agent that chooses the next diagnostic action:

```bash
python -m evalrx.benchmark.run --modality llm --model gemini-3.6-flash \
  --dataset bbh_causal_judgement --diagnosis-mode agentic --max-actions 12
```

The existing judge/coder options apply. Replace modality and dataset for image/audio tasks. Gemini probability support is probed on the actual model; unsupported logprobs and raw logits are never simulated.

`--diagnosis-mode fixed` remains the default for reproducible fixed-pipeline comparisons. Agentic mode is a separate protocol variant, not an exact model-only comparison with a historical fixed run.

## Controller scope

The controller can choose catalog probes, request custom probes, run statistics, explore data, cluster failures, propose hypotheses, and test them on EXPLORE. It can gather more evidence after an empty diagnosis, subject to action and tool-call caps. Custom M1 probes and M2 tool generation default on in agentic mode across modalities; disable them with `--no-m1-codegen` and `--no-m2-codegen`.

Repair and surgery are excluded from the benchmark controller's tool registry. After the investigation, the host runs repair search on EXPLORE and validates a frozen winner once on CONFIRM. The controller cannot inspect confirmation feedback and adapt to it. `--held-out` currently requires fixed mode; agentic mode rejects that combination before loading a model.

Built-in floor candidates default off in agentic mode (`--fix-floor` opts in). The agent can still propose known repair methods. The existing repair tier/capability constraints and maximum repair rounds remain enforced. This mode is bounded adaptive diagnosis, not an unlimited autonomous training loop.

## Failure and progress semantics

M3 exceptions are recorded as structured error exchanges with exception type, message, attempt, and retry status. `--diagnosis-retries 1` permits one additional attempt; valid empty diagnoses are not retried as transport failures. Exhaustion produces `diagnosis_error`, an error diagnosis status, and benchmark exit code 2 rather than a successful `no_hypotheses` result.

Repair progress prints proposal starts, candidate starts/completions, and case counters for ordinary template/spec evaluation, even while detailed result logging is deferred. These operational events contain no case IDs, labels, outputs, or correctness scores. Coded pipelines emit candidate start/end events; their internal calls are not represented as ordinary per-case counters.

## Scoring correction

The multiple-choice parser now recognizes balanced Markdown around a single option (for example `The correct answer is **A**.`). It preserves rejection of uncommitted thought-channel output and ambiguous option lists. This changes labels for affected historical outputs; keep old runs intact and rescore/rerun under the corrected scorer as a new result version. A parser fix alone is not evidence that the model or repair improved.
