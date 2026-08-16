# Examples

Every example is launched with Docker (`docker compose up` from its own
directory) — never `python run.py` directly. Examples that exercise the
diagnosis loop are grouped by **which M-stages they actually run**, verified
against each `run.py`'s own imports, not assumed from the folder name:

- `m1/` — M1 (probe/selection) only. *(Reserved — no example runs only M1
  as a complete deliverable today.)*
- `m1_m3/` — `VLDiagnoseLoop` through M1→M2→M3(→M5): diagnose a failure and
  hold out a verified hypothesis, but attempt no repair.
- `m2_m3/` — standalone M2 (stats) + M3 (hypothesis) via `evalvitals.explore()`,
  outside any loop object. M1 case-selection may be a separate script that
  runs first (see each example's own README), never baked into the same call.
- `m4/` — `FixAgent` direct: propose → validate a repair against a
  **hand-supplied** hypothesis. No M1–M3 discovery stage.
- `m1_m4/` — the full loop, M1→M2→M3→M4→M5. M4's exact position (baked into
  the same call, or a separate script run right after) is noted per example
  below — that distinction matters and doesn't fit in a folder name.

Five more directories don't belong to the M1–M5 progression at all, and are
kept under their own names rather than forced into a stage bucket:

- `analyzer_demos/` — one analyzer, one call, no loop of any kind.
- `agent_demos/` — agent-under-test trajectory capture (tool-calling `Agent`
  loop, e.g. `image_zoom_in`). Produces data for a *later* diagnosis; does
  not diagnose anything itself.
- `preregistered_ab_demo/` — `EvalOrchestrator`'s pre-register → mine →
  validate → confirm protocol. A different orchestrator from
  `VLDiagnoseLoop` entirely, not M-numbered.
- `dataset_selection/` — pre-M1 utilities: is this (model, dataset) pair
  even diagnosable before you spend a probing budget on it.
- `paper_diagnosis_benchmark/` — uses `explore` against the **framework's
  own track record** over research papers, not to diagnose a subject model.

## Stage coverage

| Example | Orchestrator | Stages run | M4 |
|---|---|---|---|
| `m1_m4/deco_pope`, `deco_hallu`, `deco_miss`, `deco_chair` | `VLDiagnoseLoop` | M1→M2→M3→M5 | baked in |
| `m1_m4/qwen_loop_claude` | `VLDiagnoseLoop` | M1→M2→M3→M5 | baked in |
| `m1_m4/mllms_hallucination`, `mllms_small_object` | `VLDiagnoseLoop` | M1→M2→M3→M5 | baked in |
| `m1_m3/qwen_loop_agy`, `qwen_video_temporal` | `VLDiagnoseLoop` | M1→M2→M3→M5 | separate script, run after the loop |
| `m1_m3/vlm_research_topics` | `VLDiagnoseLoop` | M1→M2→M3→M5 | none |
| `m4/qwen2_audio_tcd_mmau` | `FixAgent` only | M4 (hand-supplied hypothesis) | is M4; held-out confirm split |
| `m4/vlm_paper_benchmark/*` | `FixAgent` only | M4 only — **designed as M1→M4, M1–M3 discovery never built** | is M4; held-out confirm split |
| `m2_m3/deco_hallu_explore`, `synthetic_yield_explore` | bare `explore()` | M2→M3 | n/a |
| `m2_m3/vtcbench_diagnosis` | bare scripts | M1 → M2/M3 (`explore`) | n/a |
| `analyzer_demos/*` | none | single analyzer call | n/a |
| `agent_demos/visual_zoom_agent` | `Agent` (tool-calling) | trajectory capture only | n/a — pre-diagnosis |
| `preregistered_ab_demo/eval_agent` | `EvalOrchestrator` | mine → hypothesis → validate → confirm | not M-numbered |
| `dataset_selection/llm_band_probe` | none | pre-M1 | n/a |
| `paper_diagnosis_benchmark/` | mostly bare `explore()` | M2-ish, meta over papers | one script also uses `FixAgent` |

## Run

```bash
cd examples/analyzer_demos/qwen_attention && docker compose up
cd examples/m2_m3/synthetic_yield_explore && docker compose up
cd examples/m2_m3/deco_hallu_explore && docker compose up
cd examples/m2_m3/deco_hallu_explore && bash run_attn.sh          # attention-enriched variant (no GPU)
cd examples/m2_m3/deco_hallu_explore && bash run_attn_pipeline.sh # full held-out pipeline (SKIP_FIX=1 → no GPU)
cd examples/m2_m3/deco_hallu_explore && bash run_web.sh           # ONE web page for all of the above: upload a .zip
                                                                   # to start a new M2+M3 run, plus the script outputs
                                                                   # above attached read-only in the same sidebar
cd examples/m1_m3/qwen_loop_agy && docker compose up
cd examples/m4/qwen2_audio_tcd_mmau && docker compose up  # TCD vs MMAU, FixAgent-only
cd examples/agent_demos/visual_zoom_agent && docker compose up
cd examples/m2_m3/vtcbench_diagnosis && docker compose up
cd examples/paper_diagnosis_benchmark && docker compose up
```

The `deco_hallu_explore` example has three runnable variants: the raw probe
data (categorical signals only); `run_attn.sh` on `data_attn_full/` — the same
606 cases enriched with per-case attention-geometry scalars for all three
checkpoints (committed with the repo), which unlocks FAIL/PASS distribution
views and cross-checkpoint attention comparisons; and `run_attn_pipeline.sh` —
the complete propose → held-out test (frozen recipes + LLM judge) →
surgery/tiered-fix arc. Every explore-shaped result renders with the SAME
five-tab layout (proposal, held-out verdicts and fix each get their own
tab); stages a run never reached grey out as "not available" instead of
disappearing, so an M3-only run and a full pipeline run look alike. See its
[README](m2_m3/deco_hallu_explore/README.md).

For the general standalone exploratory analysis workflow, see
[`docs/m2_analysis.md`](../docs/m2_analysis.md).
