# Examples

Examples are grouped by the layer they exercise:

- `analyzer_demos/` — single analyzer or analysis demos. These run one diagnostic
  capability directly, without the M1-M5 diagnosis loop.
- `agent_demos/` — agent-under-test demos: a VLM drives the tool-calling
  `Agent` loop (visual tools like `image_zoom_in`) and the run is captured as
  a serialized `Trajectory` for trajectory-level diagnosis.
- `m2_statistics/` — standalone M2 (exploratory analysis) + M3 (hypothesis
  proposal) examples via `evalvitals explore`, outside the loop.
- `diagnosis_loops/` — full diagnosis-loop examples (`AutoDiagnoseLoop`,
  `VLDiagnoseLoop`, DeCo/Qwen scenarios, and related loop demos).

Run each example from its own directory, for example:

```bash
cd examples/analyzer_demos/qwen_attention && docker compose up
cd examples/m2_statistics/synthetic_yield_explore && docker compose up
cd examples/m2_statistics/deco_hallu_explore && docker compose up
cd examples/m2_statistics/deco_hallu_explore && bash run_attn.sh          # attention-enriched variant (no GPU)
cd examples/m2_statistics/deco_hallu_explore && bash run_attn_pipeline.sh # full held-out pipeline (SKIP_FIX=1 → no GPU)
cd examples/m2_statistics/deco_hallu_explore && bash run_web.sh           # ONE web page for all of the above: upload a .zip
                                                                          # to start a new M2+M3 run, plus the script outputs
                                                                          # above attached read-only in the same sidebar
cd examples/diagnosis_loops/qwen_loop_agy && docker compose up
cd examples/agent_demos/visual_zoom_agent && python run.py --device cuda:0  # minimal VLM tool-loop trajectory
cd examples/agent_demos/vtcbench_diagnosis && python run_m1.py && bash run_explore.sh
                                                                          # full agent diagnosis on a VTC-Bench task:
                                                                          # batch + probes -> records -> explore with
                                                                          # held-out confirm (needs a vllm serve endpoint)
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
[README](m2_statistics/deco_hallu_explore/README.md).

For the general standalone exploratory analysis workflow, see
[`docs/m2_analysis.md`](../docs/m2_analysis.md).
