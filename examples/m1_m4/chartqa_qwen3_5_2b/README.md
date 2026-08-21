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
