"""deco_hallu — STEP 1 of the LAMBDA×M2 validation: fused discover -> confirm.

Reuses the FROZEN M1 result (outputs/m1_state.pkl from run_m1.py) and runs the
standalone fused pipeline:

  1. transpose the M1 analyzer per-case signals + PASS/FAIL labels into records
  2. a REAL explorer (claude/codex/agy CLI agent) freely writes EDA code in a
     sandbox and proposes candidate signals, each with a deterministic recipe
  3. each recipe is compiled on a held-out CONFIRM split and adjudicated by the
     validated M2 engine (the explorer never decides)

Output: outputs/fused/fused_report.json (full report) and
outputs/fused/confirmed_recipes.json (the recipes whose signals were confirmed —
fed to run_m2-5.py --recipes for STEP 2, the in-loop repair).

    python run_m1.py    --model qwen3-vl-2b-instruct --device cuda   # once (GPU)
    python run_fused.py --backend claude                            # this step

No GPU and no judge needed here — M1 is frozen and M2 confirmation is the
deterministic catalog. Only the explorer makes CLI-agent calls.
"""

from __future__ import annotations

import argparse
import json
import pickle

import run  # build_codegen, OUT

OUT = run.OUT
M1_STATE = OUT / "m1_state.pkl"
FUSED_DIR = OUT / "fused"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="claude", choices=["claude", "codex", "agy"],
                    help="explorer coder backend (default claude)")
    ap.add_argument("--confirm-split", type=float, default=0.4,
                    help="held-out fraction for confirmation (~matches the manifest split)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--timeout-sec", type=int, default=240)
    ap.add_argument("--question", default=(
        "Among these labeled cases, what per-case signals distinguish the FAIL cases "
        "(false 'Yes' on an absent object) from the PASS cases? Propose composite "
        "signals as deterministic recipes over the available numeric columns."))
    ap.add_argument("--dashboard", action="store_true",
                    help="open the Streamlit dashboard on the fused output when done")
    ap.add_argument("--skill", action="append", default=[], metavar="DIR",
                    help="Agent-Skill dir (e.g. nature-figure) to style agent figures; repeatable")
    ap.add_argument("--allow-skills", action="store_true",
                    help="enable the Skill tool so ~/.claude/skills are usable too")
    args = ap.parse_args()

    if not M1_STATE.exists():
        raise SystemExit(f"{M1_STATE} missing — run `python run_m1.py` first (needs GPU)")
    with open(M1_STATE, "rb") as fh:
        state = pickle.load(fh)
    probe_results = state["probe_results"]
    cases = state["cases"]

    from evalvitals.agent_runtime.sandbox import ExperimentSandbox
    from evalvitals.analysis import ExploratoryAnalysisAgent, run_fused_analysis
    from evalvitals.analysis.operationalize import per_case_to_records
    from evalvitals.analysis.stats_tools import build_stats_input

    # 1) frozen analyzer signals + labels -> records the explorer/catalog can use.
    inp = build_stats_input(probe_results, cases)
    records = per_case_to_records(inp.per_case, inp.labels)
    n_fail = sum(1 for v in inp.labels.values() if v)
    print(f"frozen M1: analyzers={list(probe_results)} signals={list(inp.per_case)}")
    print(f"records={len(records)} labeled={len(inp.labels)} (fail={n_fail}, pass={len(inp.labels) - n_fail})")
    if len(inp.per_case) == 0:
        raise SystemExit("no per-case signals in the frozen M1 — nothing for the explorer to compose")

    FUSED_DIR.mkdir(parents=True, exist_ok=True)
    explorer = ExploratoryAnalysisAgent(
        cli_config=run.build_codegen(args.backend, skills=args.skill, allow_skills=args.allow_skills),
        sandbox=ExperimentSandbox(workdir=FUSED_DIR / "sandbox", cleanup=False),
        timeout_sec=args.timeout_sec,
        max_attempts=2,
    )

    print(f"\nrunning fused discover->confirm (backend={args.backend}, "
          f"confirm_split={args.confirm_split}) ...")
    report = run_fused_analysis(
        records, explorer=explorer,
        question=args.question,
        confirm_split=args.confirm_split, seed=args.seed,
    )

    # ── report ──
    print(f"\nsplit: {report.split}")
    print(f"adjudication: {report.adjudication}")
    print(f"\nobservations ({len(report.observations)}):")
    for o in report.observations[:8]:
        print(f"  - {o}")
    print(f"\ncandidate signals ({len(report.candidate_signals)}):")
    for s in report.candidate_signals:
        verdict = "REJECT H0" if s.reject else "inconclusive"
        tag = "e-BH" if s.e_value is not None else "CI"
        print(f"  - [{s.source}] {s.name}: {verdict} ({tag})"
              + (f"  recipe={s.recipe.get('expr')!r}" if s.recipe else ""))
    if report.recommended_confirmatory_tests:
        print(f"\nrecommended_confirmatory_tests ({len(report.recommended_confirmatory_tests)}):")
        for r in report.recommended_confirmatory_tests[:8]:
            print(f"  - {r}")
    for c in report.caveats:
        print(f"  [caveat] {c}")

    # ── render the explorer's chart specs (host-side, spec + CSV -> PNG) so the
    #    fused_report.json carries figure_path for Step 2's M3 and the dashboard ──
    from evalvitals.viz.renderer import render_chart_specs

    try:
        report.charts = render_chart_specs(report.charts, FUSED_DIR / "sandbox", FUSED_DIR)
        n_rendered = sum(1 for c in report.charts if c.get("figure_path"))
        print(f"rendered {n_rendered}/{len(report.charts)} chart(s) -> {FUSED_DIR / 'figures'}")
    except Exception as exc:  # rendering must never block persisting the report below
        print(f"[WARN] chart rendering failed ({exc}); persisting report without figures")

    # ── persist full report + the CONFIRMED recipes for Step 2 ──
    (FUSED_DIR / "fused_report.json").write_text(
        json.dumps(report.to_dict(), indent=2, default=str), encoding="utf-8")
    confirmed = [
        s.recipe for s in report.candidate_signals
        if s.reject and s.recipe and s.source in ("explorer", "both")
    ]
    (FUSED_DIR / "confirmed_recipes.json").write_text(
        json.dumps(confirmed, indent=2), encoding="utf-8")
    print(f"\nwrote {FUSED_DIR / 'fused_report.json'}")
    print(f"wrote {len(confirmed)} confirmed recipe(s) -> {FUSED_DIR / 'confirmed_recipes.json'}")
    if confirmed:
        print("Step 2:  python run_m2-5.py --recipes outputs/fused/confirmed_recipes.json "
              "--explore-report outputs/fused/fused_report.json")
    else:
        print("no explorer recipe was confirmed on the held-out split — "
              "nothing to feed Step 2 (this is an honest negative, not an error)")
        print("(you can still feed M3 the charts/observations: "
              "run_m2-5.py --explore-report outputs/fused/fused_report.json)")

    if args.dashboard:
        from evalvitals.analysis.dashboard import launch_dashboard

        raise SystemExit(launch_dashboard(FUSED_DIR))


if __name__ == "__main__":
    main()
