# musicavqa_videollama2

Audio-visual QA failure diagnosis: **VideoLLaMA2.1-7B-AV** on **Music-AVQA**,
following the failure family studied by AVCD (arXiv 2505.20862) —
unimodal/cross-modal hallucination in audio-visual LLMs, where the model
answers as if it can only see the video (or only hear it), not both.

Question pool: Music-AVQA's `Audio-Visual`, `Audio`, and `Visual` question
types (counting, comparative loudness, existential "is X playing", sound
source, location, temporal). Visual-only questions are the no-free-lunch
control — a fix that improves Audio/Audio-Visual accuracy by breaking
Visual-only accuracy is a different error, not an improvement.

## Files

| File | Role |
|---|---|
| `avqa_data.py` | Music-AVQA record parsing, answer scoring, `FailureCase`/`CaseBatch` builder, `ExperimentProtocol` |
| `videollama2_model.py` | `VideoLLaMA2AVModel` (real, GENERATE-only `evalrx.core.model.Model`) + `MockAVModel` (zero-weight stand-in) |
| `mine_cases.py` | Offline miner: runs the model once over a sampled question pool, labels PASS/FAIL, freezes `data/cases/{model}.json` |
| `run.py` | Loads the frozen manifest, wires `VLDiagnoseLoop` (M1→M2→M3→M4, M5 post-loop), runs the fix module |
| `config.yaml` | Model path, judge/codegen model, fix tier ceiling |
| `docker-compose.yml`, `Dockerfile` | GPU container build (see below) |

No architecture code under `evalrx/` is modified — `VideoLLaMA2AVModel`
implements the public `Model` ABC directly (the same extension point
`hf_local`/`api` backends use); the Music-AVQA adapter only builds public
`FailureCase`/`CaseBatch`/`Inputs` objects.

## Why a custom Model class

VideoLLaMA2 is not a registered EvalRX `ModelSpec` and is not a stock
`transformers` causal LM (`trust_remote_code`-style custom class + its own
`mm_infer()` generation helper, not `model.generate(**tokenizer(...))`), so
neither `evalrx.load(key)` nor `evalrx.wrap(model, tokenizer)`
(text-only VLM bring-your-own-model path, as of this writing) fit. It also
needs the **`audio_visual`** branch of `DAMO-NLP-SG/VideoLLaMA2` (not
`main`, which has no audio path) — not on PyPI, installed from source (see
Dockerfile). Its `mm_infer()` hardcodes `.cuda()` in three places;
`videollama2_model.py` keeps a device-parameterized copy (`_mm_infer_on`) —
a diff against upstream, not a rewrite — so the same class runs on `cuda` or
`cpu`.

The resulting handle is **GENERATE-only** (no attention/hidden-state
capture, since `mm_infer`'s custom path bypasses evalrx' HF-flag-based
`Trace` capture) — M1 analyzer selection and the fix tier ceiling
(`fix_max_tier: L2` in `config.yaml`) are set accordingly.

## Running

```bash
# 1. mine a frozen case pool (CPU wiring smoke test — no real weights):
python mine_cases.py --mock --n 40

# 2. run the loop against the frozen manifest (still --mock: proves M1-M5
#    code paths without a real model; real fail/pass labels are meaningless):
python run.py --mock --max-cycles 1

# Real run (needs enough RAM/VRAM for a 17GB checkpoint — see
# videollama2_model.py's --load-4bit for a quantized-load attempt):
python mine_cases.py --model videollama2.1-7b-av --n 200 --device cuda
python run.py --model videollama2.1-7b-av --device cuda

# GPU container (see docker-compose.yml for volume/env details):
docker compose up
```

`--smoke-test` on `run.py` only checks the frozen manifest exists (matches
the convention in `deco_chair`/`deco_pope`/etc.); the real wiring smoke test
is `--mock`, which runs the actual loop end to end against a fake model.
