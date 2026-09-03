# VTC-Bench agent diagnosis — probe → explore → held-out → fix → replicate

The complete agent-under-test arc on real data: a vLLM-served Qwen3-VL drives
the tool loop (`image_zoom_in`, `image_detect`) over one
[VTC-Bench](https://huggingface.co/datasets/zzzhu/VTC-Bench) task at a time,
M1 probes turn the runs into `records.json`, `evalrx explore` finds and
held-out-tests the failure structure, and paired fix experiments close the
loop.

## The recorded result (why the default config is what it is)

Diagnosis on `counting` (Qwen3-VL-2B, 81% fail) found one held-out-confirmed
invariant: **runs that loop until the tool budget dies and never answer**.
The targeted repair — `Agent(block_repeat_calls=True, force_final_answer=True)`
— was inconclusive on the diagnosis task alone (+4.7pp, e = 0.48), then
pre-registered and replicated on five further tasks:

| task | baseline→L2 pass | effect | e |
|---|---|---|---|
| counting | 16→20 / 85 | +0.047 | 0.48 |
| chart | 1→23 / 100 | +0.220 | 27962 |
| color | 8→15 / 90 | +0.078 | 0.97 |
| math (MC) | 3→14 / 66 | +0.167 | 19.5 |
| measure | 7→26 / 105 | +0.181 | 1381 |
| spatial | 6→17 / 44 | +0.250 | 45.0 |

E-values multiply across independent batches: combined e ≈ 1.6×10¹⁰ ≫ 20 —
validated. **`run_m1.py` therefore runs the loop-policy agent by default**;
pass `--no-loop-policy` to reproduce the unfixed baseline (and to record the
baseline that `run_m4.py` fix comparisons require — it refuses a policy-on
baseline via `run_config.json`).

Escalation context: 4B/8B do not beat the ~0.8 fail rate (the capability wall
is flat) but the failure phenotype migrates — identical-repeat loops at 2B,
varied never-concluding churn at 4B, perception-limited at 8B — and the
forced-answer half of the fix GROWS with scale (+0.047 → +0.094 → +0.141).

## Running it

Serve the checkpoint (images here exceed a 16k context at native resolution —
the ~1MP cap is what makes zooming meaningful):

```bash
vllm serve Qwen/Qwen3-VL-2B-Instruct --port 8901 \
    --enable-auto-tool-choice --tool-call-parser hermes \
    --max-model-len 32768 --mm-processor-kwargs '{"max_pixels": 1003520}'
# 8B: use --max-model-len 49152 (one tool-shap conversation exceeded 32k)
```

Then, from this directory (data root expected at
`/tealab-data/jiaqiliu/datasets/vtcbench`; adjust `DATA_ROOT` in run_m1.py):

```bash
python run_m1.py --task counting                  # M1: batch + probes -> outputs/records.json
bash run_explore.sh                               # M2/M3 + 0.6/0.4 held-out confirm
python run_m1.py --task counting --no-loop-policy # unfixed baseline for fix comparisons
python run_m4.py --task counting                  # paired fix arms vs that baseline
python run_m4.py --task chart --arms L2_loop_policy --out outputs_2b_chart
                                                  # pre-registered single-arm replication
```

`run_explore.sh` honors `RECORDS` / `OUT` / `MODEL_DESC` / `BACKEND` /
`TIMEOUT_SEC` (default 3600 — the CLI's 120s default truncates real analyses).
Every stage writes under `outputs*/` (gitignored); view any explore result
with `evalrx serve <out>/explore`.

Only the six four-way multiple-choice tasks are graded (last standalone A-D
letter; a run that never answers counts as FAIL). The free-text tasks
(attention_focusing, ocr, perceptual_Restoration) need a text grader first.
