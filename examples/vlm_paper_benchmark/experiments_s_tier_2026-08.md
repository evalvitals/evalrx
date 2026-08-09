# S-tier full-arc experiment record — 2026-08-08

First end-to-end test of the diagnosis system against two S-tier survey rows
([`potential_papers.md`](potential_papers.md) defines the tiers): can the
`run_autofix.py` arc find the model's problem on the paper's own data surface
and repair it with the paper's method family?

- **DyFo → `vstar_bench`**: the runner's built-in black-box search controls
  (`guided_visual_search` / `detector_visual_search`) are the DyFo/V* method
  family, so the arc is executable without implementing DyFo's MCTS.
- **DC² → `hrbench_4k`**: the data surface added with the survey. The serving
  cap below reproduces the paper's failure mechanism (4K source, ~1MP view).

**Model under test**: Qwen3-VL-4B / Qwen3-VL-2B via a local vLLM
OpenAI-compatible endpoint —
`--mm-processor-kwargs '{"max_pixels": 1003520}' --max-model-len 16384`.
Endpoint identity is passed with `AUTOFIX_MODEL_ID` / `AUTOFIX_BASE_URL`.
Reports (gitignored, local): `outputs/{vstar_bench_4b, vstar_bench_2b,
hrbench_4k_2b, hrbench_4k_2b_scaled, hrbench_4k_2b_papermethods,
hrbench_4k_2b_confirm5}.json`.

## Result in one line

Diagnosis identified the papers' failure mechanism in every run, and the
paper-family candidate `detector_visual_search_consensus` was the strongest
repair in **all five disjoint batches** (+40 fixed / −12 broken pooled), but
the combined anytime-valid evidence reached **e = 13.85 < 20**, so the
framework — correctly, by its own pre-registered rule — declines to certify
the fix. "Aligned and replicated in direction, not validated" is the honest
verdict at this model scale and sample size.

## The five batches

Every batch used rows never evaluated in any earlier batch (`--exclude-report`
with a merged exclusion manifest; batch 5 re-derives the untouched
confirmation split of batch 4 deterministically and asserts the
reconstruction). e-values are FixAgent's uniform-mixture martingale on
discordant pairs; independent batches multiply.

| # | run | selection/confirm rows | detector_visual_search pairs | fixed/broken | e |
|---|---|---|---|---|---|
| 1 | 4B × V*Bench (`vstar_bench_4b`) | 36 sel | 34 | +6/−2 | 1.02 |
| 2 | 2B × HR-Bench (`hrbench_4k_2b`) | 48 sel | 31 | +3/−0 | 2.00 |
| 3 | 2B × V*Bench (`vstar_bench_2b`) | 48 sel | 42 | +8/−2 | 2.07 |
| 4 | 2B × HR-Bench, `--paper-methods-only` (`hrbench_4k_2b_papermethods`) | 272 sel | 140 | +10/−3 | 2.05 |
| 5 | 2B × HR-Bench, frozen-candidate confirmation (`hrbench_4k_2b_confirm5`) | 256 confirm | 144 | +13/−5 | 1.61 |
| | **combined** | | | **+40/−12** | **13.85 < 20** |

Batch 5 was run through the public `FixAgent.validate_candidate()` on the
256-row confirmation split that batch 4 never touched (its selection gate did
not pass, so no confirmation calls were spent), with the candidate frozen to
the exact spec attempted in batches 1–4.

Supporting observations, per report:

- Baselines sat at 50–57% (4-way multiple choice), so failures were plentiful
  and real. Diagnosis hypotheses were mechanism-correct every time: "the model
  cannot locate an object that is visibly present" (V*Bench), "fine-grained
  attribute/number misread" (HR-Bench).
- Judge-proposed prompt candidates were **harmful at scale** (`visual_grounding`
  +4/−15, `fix_visual_perception` +4/−12 on 144 pairs, both verdict `unsafe`)
  — the selection gate correctly rejected them, which is the gate earning its
  keep.
- `guided_visual_search` (the model localises for itself) was consistently
  weaker than `detector_visual_search` (external grounding): 2B cannot
  self-localise, consistent with the agent-example finding that small models
  need external grounding tools.
- The search family only engages where the detector finds a target
  (pairs ≈ 51–94% of rows, lower on 4K HR-Bench). This engagement rate, not
  the effect direction, is why single-batch e-values stall around 2: the
  uniform-mixture e needs ~25 discordant pairs at a 4:1 ratio to clear 20.

## Why the powered batch needed `--paper-methods-only`

In batch "scaled" (not counted above: `hrbench_4k_2b_scaled`, 144 pairs) the
2B judge produced *parseable* proposals for the first time, and FixAgent's L1/L2
stages then **displaced the paper-inspired defaults entirely** — the visual
search candidates were never attempted, and every attempted candidate was
judged unsafe. Two practical notes follow:

1. `--paper-methods-only` is the reliable way to test a pre-registered paper
   family; without it, whether paper candidates run depends on whether the
   judge's freeform proposal happens to parse.
2. Upstream improvement worth filing: paper-inspired defaults should compete
   alongside judge proposals rather than being mutually exclusive with them.

## Reproduction

```bash
# serve (shared-GPU settings; see notes below)
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=<idx> \
  vllm serve Qwen/Qwen3-VL-2B-Instruct --served-model-name gpt-qwen3-vl-2b \
  --port 8010 --max-model-len 16384 --mm-processor-kwargs '{"max_pixels": 1003520}' \
  --gpu-memory-utilization 0.30 --max-num-seqs 1 --enforce-eager

python download_benchmarks.py --paper vstar_bench --per-paper 96
python download_benchmarks.py --paper hrbench_4k --per-paper 780 --scan-rows 800

AUTOFIX_MODEL_ID=gpt-qwen3-vl-2b python run_autofix.py vstar_bench \
  --limit 96 --diagnosis-cases 16 --selection-cases 48 --output-name vstar_bench_2b
AUTOFIX_MODEL_ID=gpt-qwen3-vl-2b python run_autofix.py hrbench_4k \
  --exclude-report outputs/<prior>.json --paper-methods-only \
  --limit 544 --diagnosis-cases 16 --selection-cases 272
```

Shared-GPU serving notes (both cost us a crashed engine before being learned):
multimodal vision-tower activations are **not** bounded by
`--gpu-memory-utilization` (a 4K-image workload overshot its budget by ~2 GiB
until `--max-num-seqs 1 --enforce-eager`), and CUDA device ordinals follow
fastest-first, not `nvidia-smi` order — pin with `CUDA_DEVICE_ORDER=PCI_BUS_ID`.

## Status and next steps

V*Bench is exhausted (191 rows total); HR-Bench 4K has 768/780 sampled rows
evaluated. `hrbench_8k` shares sources with the 4K split and must not be
treated as an independent replication. The two obvious routes to a validated
verdict: rerun the arc at 8B on a free GPU (higher engagement and repair rate
expected), or raise the search candidate's engagement rate and pre-register a
fresh replication. Until then this experiment stands as designed behavior:
the system found the right problem, picked the paper's fix family five times
out of five, and refused to certify it on insufficient evidence.
