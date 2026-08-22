# ChartQA × Qwen3.5-2B

The [ChartQA × Qwen2.5-VL](../chartqa_qwen2_5_vl/) chain — M1 → explore → M2 →
M3 → held-out M5 → M4 → fix — with **Qwen3.5-2B loaded together with its
vision tower** as the model under test. Same downloader, same scorer
(normalized exact match, ChartQA's relaxed 5 % numeric tolerance), same
explore/confirm split (256 rows → 128/128).

What differs from the Qwen2.5-VL example:

| | Qwen2.5-VL-7B example | this example |
|---|---|---|
| spec key | `qwen2.5-vl-7b-instruct` | `qwen3.5-2b-vl` (`Qwen/Qwen3.5-2B`, `Qwen3_5ForConditionalGeneration`) |
| thinking | n/a | **OFF on every template render** — the spec sends `enable_thinking=False` (the 2B template defaults off when the kwarg is absent, the 9B one defaults on, so it is always explicit) |
| attention | full attention every layer | hybrid: `layer_types` = `[linear ×3, full] × 6` — a forward returns 6 attention tensors for 24 layers (position *i* = model layer 4*i*+3); rollouts are partial-path |
| image | transformers 4.57 / torch cu124 | **transformers 5.15 / torch 2.13 + torchvision 0.28 cu129** (`qwen3_5` is unknown to transformers 4.x; the processor's video branch imports torchvision even for images; `gcc` because torch 2.13 JIT-compiles Triton-backed ops on first use) |

```bash
python download_chartqa.py --limit 256 --seed 5022
python run.py --smoke-test --limit 256          # manifest + scorer checks, no model
CUDA_VISIBLE_DEVICES=0 docker compose up --build -d
docker compose logs -f
```

`run.py` accepts the same flags as the Qwen2.5-VL example (`--limit`,
`--judge-provider claude|agy`, `--judge-model`, `--judge-effort`, `--fix-tier`,
`--run-dir`); `--model` defaults to `qwen3.5-2b-vl`. M1 is pinned to
`answer_extraction_audit`, `selfcheck_consistency`, `coverage_verification_gap`
and, like every analyzer since 2026-08-21, each measures its whole partition
(no per-analyzer `max_cases` cap).

The compose command pins judge and coder to Claude Opus 5
(`--judge-provider claude --judge-model claude-opus-5 --judge-effort high`);
the CLI default provider is `agy`. With no M5-verified hypothesis the run
still enters M4 and the fix stage on the best unverified leads
(`allow_unverified=True` in `vlm_benchmark_common.py`).

Weights: `Qwen/Qwen3.5-2B` (~5 GB) must be in the mounted HF cache
(`HF_HOME`), e.g. `huggingface-cli download Qwen/Qwen3.5-2B`.

## Validation run: 2026-08-21 (48 cases, A6000)

`run.py --limit 48 --judge-provider claude --judge-model sonnet --judge-effort high
--run-dir outputs/smoke48` — the whole chain on the new model, 72 min end to end
(fix stage 54 min, two coded-pipeline rounds):

| stage | result |
|---|---|
| baseline | 72.9 % (35/48), split 24/24 |
| M1 | all three pinned analyzers measured **24/24 on both partitions** (no per-analyzer cap) |
| explore / M2 | 5 observations, 6/6 charts; M2: genuine visual-evidence defect, wrong reading reproduced deterministically |
| M3 / held-out M5 | 3 leads (critic kept 1); all inconclusive on 24 confirm cases |
| M4 | best lead refuted (metric_a 0.46 vs metric_b 3.0) |
| fix | 15 EXPLORE candidates, none with positive net repairs — NOT FIXED; CONFIRM untouched |

Per-case generation is ~0.5 s after the first call (Triton JIT warm-up ~12 s);
peak GPU memory 5 GB.

## Validation run: 2026-08-21, contract32 (32 cases, `--code-only --no-explore`, A6000)

Quick live check of the coded-pipeline selection-guard change (the host guard
now anchors on each case's recorded `baseline_output`; a plain
`model_generate(case_id)` is answered from that record and is free, so the
coder no longer has to make it): `run.py --limit 32 --code-only --no-explore
--judge-provider claude --judge-model sonnet --judge-effort high --run-dir
outputs/contract32`, ~30 min end to end. The host checkout of the package was
bind-mounted read-only over the image's site-packages instead of rebuilding —
a third compose file (the tealab override replaces the whole volume list, so
pass all three with `-f`):

```yaml
services:
  chartqa_qwen3_5_2b:
    volumes:
      - /path/to/evalvitals/evalvitals:/usr/local/lib/python3.11/site-packages/evalvitals:ro
```

| stage | result |
|---|---|
| baseline | 75.0 % (24/32), split 16/16 |
| diagnosis | 2 leads, none verified on CONFIRM; M4 best lead refuted |
| fix, round 1 | coder read `baseline_output` + 3 distinct enhanced calls + 2-of-3 vote — executed first try, **no repair round** (this exact shape was voided as a contract violation in 5/5 earlier rounds) — unsafe 0/1 |
| fix, round 2 | feedback revision (2 enhanced calls) — executed first try, unsafe 0/1; nothing selected, CONFIRM untouched — NOT FIXED |

Each attempt directory under `logs/fixes/` now also carries
`coded_pipeline_result.json` (guard statistics: `n_anchored_from_recorded`,
`n_replayed`, `n_guarded`, `unanchored_ids`) and `frozen_model_control.json`
(runs after this one; contract32 predates the files). Round 2 logged 23
recoverable `expandable_segments` allocator warnings (the process had cached
~42 GB across differently sized upscaled chart images; no hard OOM, 16/16
outputs).

## Validation run: 2026-08-22, merge32 (32 cases, `--code-only`, after merging `main`)

Same 32 cases, judge/coder/explorer all `claude-opus-5`, the rebuilt image
from the merged tree (`0dcb5fe`), explore on: 28 min end to end, exit 0.
Checks that the merge kept our path: with `verified=0` M4 still ran on the
best unverified lead (refuted) and the fix stage executed (`stage_status:
completed`, no `stage_skipped` event); both coded rounds ran first try
(`coded_pipeline_result.json`: 16/16 anchored on the recorded baseline, 0
guarded, no repair round) — unsafe 0/1 and 0/2, NOT FIXED as before. The
diagnosis event carries both critic records (`critic_io` + `review`: 3
proposed, 2 rejected, none removed); the explore charts came out as 1
composition strip, 3 lines, 3 forest plots and 1 count bar under the merged
chart policy.
