# ChartQA × Qwen2.5-VL

Full EvalVitals M1→M5 diagnosis and held-out Fix validation on 512
deterministically sampled human-authored ChartQA test questions. The scorer
implements normalized exact match and the benchmark's relaxed 5% numeric
tolerance. The fixed explore/confirm split is 256/256.

An agy/Antigravity coding agent sees only EXPLORE data, runs a controlled
subtype-discovery candidate followed by one feedback-driven revision, and
freezes the best positive-net candidate. Exactly that one candidate is then
executed once on untouched
CONFIRM. No confirmation feedback, best-of-N selection on CONFIRM, or adaptive
tier escalation is allowed.
The host bridge also enforces a direct baseline, at most four calls per case,
and distinct 2-of-3 enhanced-answer support before an agent-written pipeline
may override that baseline.

```bash
python download_chartqa.py --limit 512 --seed 5022 \
  --exclude-json outputs/auto_fix_v7/report/discovery_cases.json \
  --exclude-json outputs/auto_fix_v8/report/discovery_cases.json \
  --exclude-json outputs/auto_fix_v9/report/discovery_cases.json
python run.py --smoke-test
CUDA_VISIBLE_DEVICES=6 docker compose up --build -d
docker compose logs -f
```

The downloader reads the official Hugging Face parquet, filters the `human`
ClassLabel, then writes a frozen manifest and local RGB images.
