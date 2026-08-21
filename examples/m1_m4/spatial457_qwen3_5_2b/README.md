# Spatial457 × Qwen3.5-2B

The [Spatial457 × Qwen2.5-VL](../spatial457_qwen2_5_vl/) chain — M1 → explore
→ M2 → M3 → held-out M5 → M4 → fix — with **Qwen3.5-2B loaded together with
its vision tower** as the model under test. Same downloader (level-5 6D spatial
questions, 640×480 renders), same scorer (case / punctuation / article / yes-no
normalized exact match), same explore/confirm split (256 rows → 128/128).

What differs from the Qwen2.5-VL example:

| | Qwen2.5-VL-7B example | this example |
|---|---|---|
| spec key | `qwen2.5-vl-7b-instruct` | `qwen3.5-2b-vl` (`Qwen/Qwen3.5-2B`, `Qwen3_5ForConditionalGeneration`) |
| thinking | n/a | **OFF on every template render** — the spec sends `enable_thinking=False` (the 2B template defaults off when the kwarg is absent, the 9B one defaults on, so it is always explicit) |
| attention | full attention every layer | hybrid: `layer_types` = `[linear ×3, full] × 6` — a forward returns 6 attention tensors for 24 layers (position *i* = model layer 4*i*+3); rollouts are partial-path |
| image | transformers 4.57 / torch cu124 | **transformers 5.15 / torch 2.13 + torchvision 0.28 cu129** (`qwen3_5` is unknown to transformers 4.x; the processor's video branch imports torchvision even for images; `gcc` because torch 2.13 JIT-compiles Triton-backed ops on first use) |

```bash
python download_spatial457.py --limit 256 --seed 7457
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

## Load check: 2026-08-21 (A6000)

Same image recipe as `chartqa_qwen3_5_2b` (whose 48-case chain ran end to end);
here the in-container check loaded `qwen3.5-2b-vl`, generated on three frozen
cases (`small → tiny`, `scooter → bus`, `biplane → airplane`, 0.4 s/case after
warm-up) and captured 6 attention tensors + the image-token mask (339 tokens for
a 640×480 render).
