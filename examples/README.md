# Examples

Every example is launched with Docker (`docker compose up` from its own
directory) — never `python run.py` directly. Examples that exercise the
diagnosis loop are grouped by **which M-stages they actually run**, verified
against each `run.py`'s own imports, not assumed from the folder name:

- `m1/` — M1 (probe/selection) only. *(Reserved — no example runs only M1
  as a complete deliverable today.)*
- `m1_m3/` — `VLDiagnoseLoop` through M1→M2→M3(→M4): diagnose a failure and
  hold out a verified hypothesis, but attempt no repair.
- `m2_m3/` — standalone M2 (stats) + M3 (hypothesis) via `evalrx.explore()`,
  outside any loop object. M1 case-selection may be a separate script that
  runs first (see each example's own README), never baked into the same call.
- `m5/` — `FixAgent` direct: propose → validate a repair against a
  **hand-supplied** hypothesis. No M1–M3 discovery stage.
- `m1_m5/` — the full loop, M1→M2→M3→M4→M5. M5's exact position (baked into
  the same call, or a separate script run right after) is noted per example
  below — that distinction matters and doesn't fit in a folder name.

Six more directories don't belong to the M1–M5 progression at all, and are
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
- `benchmark/` — the full loop as a **matrix**: model family (Qwen /
  Gemma 4 / Nemotron 3) × modality (vlm / llm / alm) × dataset, one image per
  family, size and dataset as runtime arguments. See its [README](benchmark/README.md).

## Stage coverage

| Example | Orchestrator | Stages run | M5 |
|---|---|---|---|
| `m1_m5/deco_pope`, `deco_hallu`, `deco_miss`, `deco_chair` | `VLDiagnoseLoop` | M1→M2→M3→M4 | baked in |
| `m1_m5/qwen_loop_claude` | `VLDiagnoseLoop` | M1→M2→M3→M4 | baked in |
| `m1_m5/mllms_hallucination`, `mllms_small_object` | `VLDiagnoseLoop` | M1→M2→M3→M4 | baked in |
| `m1_m5/musicavqa_videollama2` | `VLDiagnoseLoop` | M1→M2→M3→M4 | separate call (`loop.run_m5`/`run_fix`), right after `loop.run()` |
| `m1_m5/mmau_qwen2_audio` | `VLDiagnoseLoop` | M1→M2→M3→M4 | `loop.run_fix()`, right after `loop.run()`; M1 pinned to a static audio-safe analyzer set (see run.py) |
| `m1_m3/qwen_loop_agy`, `qwen_video_temporal` | `VLDiagnoseLoop` | M1→M2→M3→M4 | separate script, run after the loop |
| `m1_m3/vlm_research_topics` | `VLDiagnoseLoop` | M1→M2→M3→M4 | none |
| `m5/vlm_paper_benchmark/*` | `FixAgent` only | M5 only — **designed as M1→M5, M1–M3 discovery never built** | is M5; held-out confirm split |
| `m2_m3/deco_hallu_explore`, `synthetic_yield_explore` | bare `explore()` | M2→M3 | n/a |
| `m2_m3/vtcbench_diagnosis` | bare scripts | M1 → M2/M3 (`explore`) | n/a |
| `analyzer_demos/*` | none | single analyzer call | n/a |
| `agent_demos/visual_zoom_agent` | `Agent` (tool-calling) | trajectory capture only | n/a — pre-diagnosis |
| `preregistered_ab_demo/eval_agent` | `EvalOrchestrator` | mine → hypothesis → validate → confirm | not M-numbered |
| `dataset_selection/llm_band_probe` | none | pre-M1 | n/a |
| `benchmark/<modality>/<family>` | `VLDiagnoseLoop` | M1→M2→M3→M4 | `loop.run_m5`/`run_fix`, right after `loop.run()`; pinned M1 per dataset |
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
cd examples/m1_m5/musicavqa_videollama2 && docker compose up  # audio-visual QA, VideoLLaMA2.1-7B-AV
cd examples/m1_m5/mmau_qwen2_audio && docker compose up  # TCD vs MMAU, full M1->M5 loop
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

## Catalog M2 + explore inside one loop run

Two different programs produce "M2" in this repo, and they are not the same
thing:

| | loop M2 — `StatsAnalysisAgent` | standalone `evalrx explore` — `ExploratoryAnalysisAgent` |
|---|---|---|
| input | M1 analyzer per-case findings (`StatsInput`) | a flat per-case records table (any source) |
| method | **confirmatory**: judge-picked tools from a fixed statistical catalog, e-BH/BH multiplicity, optional codegen tools | **exploratory**: a coder agent writes free-form pandas EDA in a sandbox; the host recomputes candidate verdicts (`adjudicate`) and renders its chart specs |
| output | JSON (`artifacts/c0_m2_stats_results.json` …) + chart *specs* the dashboard plots live; `m2_effects.png` only with `figure_dir=` | `exploratory_report.json` + `tables/*.csv` + `figures/*.png` + `analysis.py` |
| verdict | yes — the evidence M4 tests | no — "the explorer never decides" |

A `VLDiagnoseLoop` run can carry **both**: keep the catalog M2 as the evidence
and add the explorer as a descriptive lane that runs between M1 and M2 over
the *same* per-case table (M1 signals + PASS/FAIL labels). Its observations
and rendered charts reach M3 as an `ExploreContext` (which hypotheses to
propose — never *whether* one is true), and land on disk for the dashboard.
It never enters M2's confirmatory family, M4, or the fix gate. Wiring, for a
`RunContext`-style example:

```python
from evalrx.agent_runtime.sandbox import ExperimentSandbox
from evalrx.analysis import ExploratoryAnalysisAgent, StatsAnalysisAgent
from evalrx.eval_agent import DiagnosisAgent, RunContext, VLDiagnoseLoop

with RunContext("examples/foo/outputs", verbose=True) as ctx:
    explorer = ExploratoryAnalysisAgent(
        cli_config=codegen,                       # the same CliAgentConfig M2's codegen uses
        sandbox=ExperimentSandbox(workdir=ctx.explore_dir / "sandbox", cleanup=False),
        timeout_sec=900, max_attempts=2,
    )
    loop = VLDiagnoseLoop(
        model=model, protocol=protocol,
        stats_agent=StatsAnalysisAgent(judge=judge, allow_codegen=True, codegen_config=codegen,
                                       figure_dir=str(ctx.figures_dir)),  # catalog M2, unchanged; + m2_effects.png
        diagnosis_agent=DiagnosisAgent(judge=judge),
        run_logger=ctx.logger,
        explorer=explorer,                        # ← turns the explore step on
        explore_dir=ctx.explore_dir,              # ← <root>/explore (also the default with ctx.logger)
        # explore_question="...",                 # optional; default is built from the protocol
    )
    report = loop.run(cases)                      # or loop.run_analysis(cases) — explore runs in both
```

What you get per run:

- `<root>/explore/exploratory_report.json`, `tables/*.csv`, `figures/*.png`,
  `analysis.py`, `sandbox/` — the dashboard's loop view finds them by path
  (`<root>/*/exploratory_report.json`) and stops saying "No explore report
  was found"; the Analysis panel shows the explorer's candidate signals
  (descriptive), charts and tables next to the catalog M2 verdicts.
- `run_log.jsonl`: an `explore` event per cycle (counts, observations,
  rendered figure paths, `report_path`); the `diagnosis` event records the
  explore figures M3 was shown (`explore_figures`, `referenced_charts`).
- M3's prompt carries an "EXPLORATORY MECHANISM NOTES … UNCONFIRMED" block
  with the observations/caveats, and the PNGs are attached as images.

Rules of the road:

- The step is best-effort: an explorer failure logs a warning and the cycle
  continues on M2 alone; `run_confirm()` (M4 → fix) never explores.
- The explorer's recipes are **not** bridged into M2's family here — the loop
  discovers and confirms on the same rows, which would be double-dipping. To
  confirm explorer recipes on a held-out split use the fused pipeline
  (`run_fused_analysis` → `signal_recipes=` + `explore_report=`, as
  `m1_m5/deco_hallu/run_fused.py` does).
- `explore/` is rewritten every cycle (it holds what the latest M3 saw); the
  per-cycle `explore` events keep every cycle's counts.
- Cost: one coder-agent call per cycle (minutes) plus host-side matplotlib
  rendering. Omit `explorer=` (or set it to `None`) and the loop is exactly
  what it was.

`dataset_selection/llm_benchmark` has this wired behind `config.yaml`
`explore: true` (`EXPLORE=0` / `--no-explore` to turn it off) — see its
[README](dataset_selection/llm_benchmark/README.md).
