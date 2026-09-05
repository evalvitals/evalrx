# Spatial457 × Qwen2.5-VL

Full EvalRX M1→M4 diagnosis and held-out Fix validation on 512
deterministically sampled `L5_6d_spatial` questions from Spatial457.  The
explore/confirm split is fixed at 256/256. An agy/Antigravity coding agent sees
only EXPLORE data, runs a controlled subtype-discovery candidate followed by
one feedback-driven revision, and
freezes the best positive-net candidate. Exactly that one candidate is then
executed once on untouched CONFIRM. No confirmation feedback, best-of-N
selection on CONFIRM, or adaptive tier escalation is allowed.
The host bridge also anchors its selection guard on each case's recorded
`baseline_output` (a plain `model_generate(case_id)` is answered from that
record and is free; since 2026-08-21 a pipeline no longer has to make that
call), caps model-hitting calls at four per case, and requires distinct 2-of-3
enhanced-answer support before an agent-written pipeline may override that
baseline.

> **Fix pool (2026-08-20):** by default `run.py` now fields the full candidate
> family — judge-proposed L1/L2 prompts and specs, the `self_consistency_5`
> floor, and the coder-written L2 pipeline — the same shape as
> `examples/dataset_selection/llm_benchmark`. The single-candidate
> autonomous-code-repair protocol described below is `--code-only`.

```bash
python download_spatial457.py --limit 512 --seed 7457 \
  --exclude-json outputs/auto_fix_v7/report/discovery_cases.json \
  --exclude-json outputs/auto_fix_v8/report/discovery_cases.json \
  --exclude-json outputs/auto_fix_v9/report/discovery_cases.json \
  --exclude-json outputs/auto_fix_v10/report/discovery_cases.json
python run.py --smoke-test
CUDA_VISIBLE_DEVICES=4 docker compose up --build -d
docker compose logs -f
```

The downloader uses the official `RyanWW/Spatial457` Hugging Face repository
directly because recent `datasets` versions no longer execute dataset loading
scripts. It writes a frozen manifest plus only the images referenced by it.
