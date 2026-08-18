#!/usr/bin/env python3
"""mmau_qwen2_audio — Qwen2-Audio-7B-Instruct x MMAU test-mini: full M1→M4 loop.

Was ``examples/m4/qwen2_audio_tcd_mmau`` — a FixAgent-only run against a
hand-supplied hypothesis ("temporal smoothing bias", TCD's own framing from
its paper, Li et al. 2026, arXiv:2604.15383). This version drops the
hand-supplied hypothesis and lets the loop discover its own:

    1. download_mmau.py          (run first, standalone — writes
                                   data/mmau_test_mini.jsonl + data/audio/*.wav)
    2. Fresh baseline pass        generate_tcd_baseline (forced greedy) on every
                                   row → PASS/FAIL labels, computed live every
                                   run, never cached (Qwen2-Audio-7B-Instruct's
                                   own generation_config defaults to sampling,
                                   and TCD's candidate is greedy by
                                   construction — a sampled baseline would not
                                   be a valid paired comparison, see
                                   generate_tcd_baseline's docstring)
    3. ExperimentProtocol         OBSERVATION ONLY — describes the MMAU task
                                   (sound/music/speech, 4-way multiple choice)
                                   and the measured baseline accuracy; does not
                                   name "temporal smoothing bias" or point at
                                   TCD. That mechanism, if it appears, has to
                                   come out of M3.
    4. VLDiagnoseLoop M1→M5       M1 uses a PINNED static analyzer set (see
                                   PINNED_M1_ANALYZERS below — why LLM-guided
                                   catalog selection is not used here yet); M2
                                   StatsAnalysisAgent (e-BH-corrected); M3
                                   DiagnosisAgent (Claude judge); M5
                                   HypothesisTester (protocol-consistency +
                                   statistical significance)
    5. loop.run_fix               FixAgent's tiered candidates — including the
                                   paper-registered ``tcd_temporal_blur`` L3a
                                   candidate (fix_agent.py's admission gate:
                                   has_audio + task=="multiple_choice" +
                                   paper_method_fidelity — see
                                   ``evalvitals/eval_agent/stages/fix_agent.py``,
                                   search ``_l3_candidates``) — validated on a
                                   held-out CONFIRM split the loop's own M1–M5
                                   discovery never saw (``--confirm-split``).

Usage:
    python download_mmau.py --limit 120
    python run.py --model qwen2-audio-7b-instruct --limit 120
    python run.py --smoke-test     # fast wiring check, no GPU/model/judge —
                                    # exercises the REAL VLDiagnoseLoop + FixAgent
                                    # against a synthetic model+cases, specifically
                                    # to catch the tcd_temporal_blur admission gate
                                    # silently never firing (wrong metadata key,
                                    # wrong max_tier, model missing generate_tcd*).
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any, Callable

HERE = Path(__file__).parent
DATA = HERE / "data"
MANIFEST = DATA / "mmau_test_mini.jsonl"


# ---------------------------------------------------------------------------
# Dataset glue — unchanged from the old FixAgent-only run.py
# ---------------------------------------------------------------------------

def load_records(limit: int | None = None) -> list[dict[str, Any]]:
    if not MANIFEST.is_file():
        raise SystemExit(f"{MANIFEST} does not exist -- run download_mmau.py first")
    records = [json.loads(line) for line in MANIFEST.read_text(encoding="utf-8").splitlines() if line]
    return records[:limit] if limit else records


def shuffled_rows(rows: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    shuffled = list(rows)
    random.Random(seed).shuffle(shuffled)
    return shuffled


def task_prompt(row: dict[str, Any]) -> str:
    options = "\n".join(row["choices"])
    return (
        f"{row['instruction']}\n\n{options}\n\n"
        "Listen to the audio and reply with only the option letter (A, B, C, or D)."
    )


def parsed_choice(output: str) -> str:
    marked = re.findall(r"(?:answer|choice|final)\s*[:=-]?\s*([A-D])\b", output.upper())
    letters = re.findall(r"\b([A-D])\b", output.upper())
    return marked[-1] if marked else (letters[-1] if letters else "")


def score(row: dict[str, Any], output: str) -> bool:
    return parsed_choice(output) == str(row["expected"]).strip().upper()


def score_case(case: "Any", output: str) -> bool:
    return score({"expected": case.expected}, output)


def row_inputs(row: dict[str, Any]) -> "Any":
    from evalvitals.core.case import Inputs

    return Inputs(prompt=task_prompt(row), audio=str(DATA / row["audio_path"]))


def evaluate(rows: list[dict[str, Any]], strategy: Callable[[dict[str, Any]], str]) -> dict[str, Any]:
    """One fresh generation pass over *rows* — never cached across runs."""
    cases = []
    for row in rows:
        output = strategy(row)
        cases.append({"id": row["id"], "output": output, "correct": score(row, output)})
    correct = sum(case["correct"] for case in cases)
    return {"n": len(cases), "correct": correct, "accuracy": correct / len(cases), "cases": cases}


def make_cases(rows: list[dict[str, Any]], baseline: dict[str, Any]) -> "Any":
    from evalvitals.core.case import CaseBatch, FailureCase, Label

    baseline_by_id = {case["id"]: case for case in baseline["cases"]}
    return CaseBatch([
        FailureCase(
            id=row["id"],
            inputs=row_inputs(row),
            expected=row["expected"],
            observed=baseline_by_id[row["id"]]["output"],
            label=Label.PASS if baseline_by_id[row["id"]]["correct"] else Label.FAIL,
            metadata={**row["metadata"], "task": row["task"]},
        )
        for row in rows
    ])


# ---------------------------------------------------------------------------
# M1 — pinned analyzer set
# ---------------------------------------------------------------------------
#
# ProbeAgent(judge=<claude>, protocol=...) would let the judge pick from the
# FULL analyzer catalog — every analyzer whose applies_to_modalities overlaps
# the model's modalities. Because most catalog analyzers declare
# applies_to_modalities={"text","image"} (the modality gate is an OR, and
# every model has "text"), that catalog is NOT actually filtered down to
# audio-safe analyzers by the gate alone. Auditing each one against a live
# Qwen2-Audio handle for the first time is out of scope here, so M1 is pinned
# to a static, code-reviewed subset instead (ProbeAgent(judge=None, ...) forces
# the StrategyProbe fallback — LLM-guided catalog selection never runs):
#
#   - all eight preserve non-image Inputs fields correctly (verified by
#     reading each analyzer's _run: dataclasses.replace(case.inputs, ...) or
#     case.inputs unmodified — never a bare Inputs(prompt=..., image=...)
#     that would silently drop .audio; prompt_contrast needed a fix for this,
#     see evalvitals/analyzers/perturbation/prompt_contrast.py)
#   - none depend on vision-specific internals (attention/logit_lens/cka/
#     mm_shap and friends are excluded — untested against an audio tower)
#   - format_sensitivity is *built* for exactly this task shape (4-way MC
#     option-order bias); the rest are black-box output/logprob diagnostics
#     that don't care what the input modality was
#   - prompt_contrast/cot_faithfulness are excluded even though audio-safe
#     now: their DEFAULT strategy templates are image-phrased ("describe what
#     you see in the relevant region of the image..."), which is not just
#     irrelevant but actively confusing prompt content for an audio model —
#     the old FixAgent-only run measured exactly this failure mode for
#     generic fallback candidates (see README's "what does the agent propose
#     on its own?" section).
PINNED_M1_ANALYZERS = [
    "answer_extraction_audit",   # is the FAIL label real or a parse miss?
    "termination_audit",          # truncated / degenerate / gave up?
    "selfcheck_consistency",      # black-box hallucination signal
    "format_sensitivity",         # MC option-order bias vs content-tracking
    "self_consistency",           # resample stability
    "calibration",                 # stated/derived confidence vs correctness
    "logprob_entropy",            # answer-token uncertainty
    "coverage_verification_gap",  # cannot-solve vs cannot-select
]


def build_protocol() -> "Any":
    from evalvitals.eval_agent import ExperimentProtocol

    return ExperimentProtocol(
        description=(
            "We evaluate an audio-language model (Qwen2-Audio-7B-Instruct) on "
            "MMAU test-mini: four-way multiple-choice questions about a short "
            "audio clip, spanning three domains — sound, music, and speech. "
            "The model must select the option that best matches what is "
            "actually audible in the clip. We want to know what distinguishes "
            "the questions it gets right from the ones it gets wrong."
        ),
        task_domain="audio question answering (MMAU)",
        success_criteria=(
            "The selected option letter must match the gold answer for the clip."
        ),
        target_modalities=frozenset({"text", "audio"}),
    )


def build_judge(model_name: str, effort: str) -> "Any":
    from evalvitals.eval_agent import ClaudeModel

    judge = ClaudeModel(model=model_name, effort=effort)
    if not judge.generate("Reply with exactly the word OK").strip():
        raise SystemExit(
            f"judge probe: claude --model {model_name} returned empty "
            f"(rate-limited?) — try --judge-model sonnet or haiku"
        )
    print(f"judge: claude model={model_name} effort={effort or 'default'}")
    return judge


# ---------------------------------------------------------------------------
# Smoke test — real VLDiagnoseLoop + FixAgent, synthetic model/judge
# ---------------------------------------------------------------------------

def _run_smoke_test(args: argparse.Namespace) -> None:
    from evalvitals.core.capability import Capability
    from evalvitals.core.case import CaseBatch, FailureCase, Inputs, Label
    from evalvitals.core.result import Result
    from evalvitals.eval_agent import (
        DiagnosisResult,
        ExperimentProtocol,
        FixAgent,
        Hypothesis,
        HypothesisTester,
        ProbingSchema,
        RunContext,
        StatsAnalysisAgent,
        StatsToolAgent,
        VLDiagnoseLoop,
    )

    class _SmokeAudioModel:
        """Baseline always answers 'A'; generate_tcd 'hears' GOLD= in the prompt."""

        def __init__(self) -> None:
            self.capabilities = frozenset({Capability.GENERATE})
            self.modalities = frozenset({"text", "audio"})

        def __repr__(self) -> str:
            return "SmokeAudioModel()"

        def generate(self, inputs, **kwargs) -> str:
            return "A"

        def generate_tcd_baseline(self, inputs, **kwargs) -> str:
            return "A"

        def generate_tcd(self, inputs, **kwargs) -> str:
            m = re.search(r"GOLD=([A-D])", str(getattr(inputs, "prompt", inputs)))
            return m.group(1) if m else "A"

        def paper_method_fidelity(self, name: str) -> str:
            return "native_layer_matched_stability" if name == "tcd" else "unavailable"

        def forward(self, inputs, capture, spec=None):
            raise NotImplementedError("SmokeAudioModel only supports generate().")

    def _smoke_case(id_: str, gold: str) -> FailureCase:
        # Baseline ("A" always) only matches gold=="A" rows -> PASS; the rest FAIL.
        return FailureCase(
            id=id_,
            inputs=Inputs(prompt=f"Pick one: A/B/C/D. GOLD={gold}", audio="smoke://clip.wav"),
            expected=gold,
            observed="A",
            label=Label.PASS if gold == "A" else Label.FAIL,
            metadata={"task": "multiple_choice"},
        )

    cases = CaseBatch([
        _smoke_case("s0", "A"),
        _smoke_case("s1", "A"),
        _smoke_case("s2", "B"),
        _smoke_case("s3", "B"),
    ])
    model = _SmokeAudioModel()

    class _SmokeProbe:
        last_schema = None

        def probe(self, model, data, **kwargs):
            fail_ids = [c.id for c in data if c.label == Label.FAIL]
            self.last_schema = ProbingSchema(
                selected_analyzers=["format_sensitivity"],
                rationale="Smoke probe marks discovered failures as a signal.",
                protocol=kwargs.get("protocol"),
            )
            findings = {
                "per_case": [{"sample_id": cid, "defaults_to_a": True} for cid in fail_ids],
            }
            return {"format_sensitivity": Result(
                analyzer="format_sensitivity", model=repr(model), cases=data, findings=findings,
            )}

    class _SmokeDiagnosisAgent:
        def diagnose(self, analysis, prior_cycles=None):
            h = Hypothesis(
                statement="The model defaults to option A when it cannot resolve the clip.",
                target_model=analysis.model_name,
                predicted_failure_mode="temporal_smoothing_bias",
            )
            return DiagnosisResult(
                model_name=analysis.model_name,
                hypotheses=[h],
                findings_summary={n: r.findings for n, r in analysis.raw_results.items()},
                raw_judge_output="HYPOTHESIS: defaults to A.\nFAILURE_MODE: temporal_smoothing_bias",
            )

    ctx = RunContext(args.run_dir, verbose=True, config={"smoke_test": True})
    loop = VLDiagnoseLoop(
        model=model,
        protocol=ExperimentProtocol(
            description="Smoke: synthetic 4-choice audio QA.",
            task_domain="audio question answering (MMAU smoke)",
            success_criteria="the selected letter must match GOLD.",
            target_modalities=frozenset({"text", "audio"}),
        ),
        probe_agent=_SmokeProbe(),
        stats_agent=StatsAnalysisAgent(stats_tool_agent=StatsToolAgent(max_tools=3)),
        diagnosis_agent=_SmokeDiagnosisAgent(),
        hypothesis_tester=HypothesisTester(min_effect=0.05),
        fix_agent=FixAgent(
            score_fn=score_case,
            max_tier=args.fix_max_tier,
            candidate_allowlist=None if args.unrestricted else ["tcd_temporal_blur"],
        ),
        max_cycles=1,
        run_logger=ctx.logger,
    )
    report = loop.run(cases)
    ctx.write_diagnose_report(report, cases)

    print(f"\nSmoke test result: stopped_by={report.stopped_by} cycles={report.cycles} "
          f"verified={len(report.verified_hypotheses)}")

    outcome = loop.run_fix(report, cases)
    ctx.finalize()
    names = [v.candidate.name for v in outcome.attempted]
    print(f"  fix candidates attempted: {names}")
    if "tcd_temporal_blur" not in names:
        raise SystemExit(
            "Smoke test FAILED: tcd_temporal_blur was never proposed -- the L3a "
            "admission gate (fix_agent.py _l3_candidates) did not fire. Check "
            "case.metadata['task']=='multiple_choice', --fix-max-tier>=L3a, and "
            "the model's generate_tcd/generate_tcd_baseline/paper_method_fidelity."
        )
    tcd = next(v for v in outcome.attempted if v.candidate.name == "tcd_temporal_blur")
    if tcd.n_fixed < 1:
        raise SystemExit("Smoke test FAILED: tcd_temporal_blur fixed 0 of the synthetic FAIL cases.")
    effect_str = f"{tcd.effect:+.3f}" if tcd.effect is not None else "n/a"
    print(f"  tcd_temporal_blur: fixed={tcd.n_fixed} broken={tcd.n_broken} effect={effect_str}")
    print("Smoke test passed.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="qwen2-audio-7b-instruct")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, default=120)
    parser.add_argument(
        "--max-tokens", type=int, default=24,
        help="subject-model generation budget (baseline scoring AND every M1 "
             "probe re-ask). 24, not the old FixAgent-only run's 8: the pinned "
             "M1 set's re-asks (calibration's confidence elicitation, "
             "self-consistency resamples) need a little more room than a bare "
             "option letter, but stay well short of the image-phrased "
             "'describe first' templates this run deliberately excludes.",
    )
    parser.add_argument("--max-cycles", type=int, default=2)
    parser.add_argument("--max-analyzers", type=int, default=len(PINNED_M1_ANALYZERS))
    parser.add_argument(
        "--confirm-split", type=float, default=0.6,
        help="fraction of --limit rows held out from M1-M5 discovery, reserved "
             "for loop.run_fix's validation (VLDiagnoseLoop's own leak-#3 split). "
             "Ignored (forced to 0.0) under --analysis-only, since nothing "
             "downstream would ever read the held-out partition.",
    )
    parser.add_argument("--fix-max-tier", default="L3a")
    parser.add_argument("--judge-model", default="claude-fable-5")
    parser.add_argument("--judge-effort", default="low")
    parser.add_argument("--seed", type=int, default=20260814)
    parser.add_argument(
        "--allow-adapted-paper-methods", action="store_true",
        help="also admit TCD on non-layer-matched audio specs (reported as adapted, never native)",
    )
    parser.add_argument(
        "--unrestricted", action="store_true",
        help="drop candidate_allowlist=['tcd_temporal_blur'] so every admissible "
             "L0-L3a candidate (paper defaults AND judge-proposed ones) competes "
             "on equal footing",
    )
    parser.add_argument(
        "--analysis-only", action="store_true",
        help="run M1+M2 only (skip M3/M5/fix)",
    )
    parser.add_argument("--run-dir", default=str(HERE / "outputs"))
    parser.add_argument(
        "--smoke-test", action="store_true",
        help="fast wiring check (no GPU/model/judge) -- see module docstring",
    )
    args = parser.parse_args()

    if args.smoke_test:
        _run_smoke_test(args)
        return 0

    rows = shuffled_rows(load_records(), args.seed)
    if len(rows) < args.limit:
        raise SystemExit(f"only {len(rows)} rows available, fewer than --limit {args.limit}")
    rows = rows[: args.limit]

    from evalvitals.models.backends.base import RuntimeConfig
    from evalvitals.models.backends.hf_local import HFLocalModel
    from evalvitals.specs import get_spec

    model = HFLocalModel(
        get_spec(args.model),
        RuntimeConfig(device=args.device, dtype="bfloat16", max_new_tokens=args.max_tokens),
    )
    model.load()
    print(f"paper_method_fidelity(tcd) = {model.paper_method_fidelity('tcd')!r}")

    def baseline_generate(row: dict[str, Any]) -> str:
        return model.generate_tcd_baseline(row_inputs(row), max_new_tokens=args.max_tokens)

    baseline = evaluate(rows, baseline_generate)
    print(f"baseline accuracy on {len(rows)} rows: {baseline['accuracy']:.1%} "
          f"({baseline['correct']}/{baseline['n']})")

    cases = make_cases(rows, baseline)
    # The tcd_temporal_blur admission gate (fix_agent.py _l3_candidates) is an
    # EXACT set match on case.metadata["task"] -- if this ever drifts (a case
    # loses the key, or gains a second task tag), TCD silently stops being
    # proposed with no error anywhere downstream. Fail loud, here, first.
    tasks = {c.metadata.get("task") for c in cases}
    assert tasks == {"multiple_choice"}, (
        f"expected every case to carry metadata['task']=='multiple_choice', got {tasks} "
        "-- tcd_temporal_blur will never be proposed by loop.run_fix with this batch"
    )

    judge = build_judge(args.judge_model, args.judge_effort)

    from evalvitals.eval_agent import (
        DiagnosisAgent,
        FixAgent,
        HypothesisTester,
        ProbeAgent,
        RunContext,
        StatsAnalysisAgent,
        StrategyProbe,
        VLDiagnoseLoop,
    )

    ctx = RunContext(
        args.run_dir, verbose=True,
        config={
            "model": args.model,
            "judge_model": args.judge_model,
            "limit": args.limit,
            "max_cycles": args.max_cycles,
            "confirm_split": args.confirm_split,
            "fix_max_tier": args.fix_max_tier,
            "analysis_only": args.analysis_only,
        },
    )

    protocol = build_protocol()
    print("\nExperimentProtocol:")
    print(f"  task_domain : {protocol.task_domain}")
    print(f"  description : {protocol.description[:100]}...")

    # M1 pinned static selection -- see PINNED_M1_ANALYZERS' comment above for why.
    probe_agent = ProbeAgent(
        probe=StrategyProbe(priority_override={k: PINNED_M1_ANALYZERS for k in ("vlm", "agent", "llm")}),
        judge=None,
        max_analyzers=args.max_analyzers,
    )
    stats_agent = StatsAnalysisAgent(
        judge=None if args.analysis_only else judge,
        figure_dir=str(ctx.figures_dir),
        allow_codegen=False,
    )
    diagnosis_agent = None if args.analysis_only else DiagnosisAgent(judge=judge)
    hypothesis_tester = None if args.analysis_only else HypothesisTester(judge=judge, min_effect=0.05)

    fix_agent = FixAgent(
        judge=judge,
        max_tier=args.fix_max_tier,
        score_fn=score_case,
        run_logger=ctx.logger,
        run_context=ctx,
        candidate_allowlist=None if args.unrestricted else ["tcd_temporal_blur"],
        allow_adapted_paper_methods=args.allow_adapted_paper_methods,
        # a coding backend is out of scope for what this run is testing --
        # see the README's "what does the agent propose on its own?" section.
        allow_codegen=False,
    )

    # --analysis-only never reaches run_fix, so nothing would ever read the
    # held-out CONFIRM partition -- reserving one anyway would silently starve
    # M1/M2 of rows the user believes they gave it (--limit 120 but M1/M2 only
    # ever seeing the ~40% EXPLORE share). Force it off in that mode.
    confirm_split = 0.0 if args.analysis_only else args.confirm_split
    n_confirm = round(len(cases) * confirm_split) if 0.0 < confirm_split < 1.0 else 0
    print(f"  explore/confirm split: {len(cases) - n_confirm} explore "
          f"(M1-M5 discovery) / {n_confirm} confirm (held out for run_fix)")

    loop = VLDiagnoseLoop(
        model=model,
        protocol=protocol,
        probe_agent=probe_agent,
        stats_agent=stats_agent,
        diagnosis_agent=diagnosis_agent,
        hypothesis_tester=hypothesis_tester,
        fix_agent=fix_agent,
        max_cycles=args.max_cycles,
        run_logger=ctx.logger,
        analysis_only=args.analysis_only,
        confirm_split=confirm_split,
        confirm_split_seed=args.seed,
    )

    print(f"\n{'='*64}\nVLDiagnoseLoop  model={args.model}  max_cycles={args.max_cycles}\n{'='*64}")
    report = loop.run(cases)
    ctx.write_diagnose_report(report, cases)

    print(f"\n{'='*64}\nLOOP RESULT  stopped_by={report.stopped_by}  cycles={report.cycles}\n{'='*64}")
    print(f"  total hypotheses proposed : {len(report.all_hypotheses)}")
    print(f"  verified (protocol-consistent + statistically supported): "
          f"{len(report.verified_hypotheses)}")
    for vr in report.verified_hypotheses:
        print(f"    [{vr.status.value}] {vr.hypothesis.statement}")
        print(f"           effect={vr.effect_size}  confidence={vr.confidence:.2f}"
              f"  protocol_ok={vr.is_consistent_with_protocol}")

    if not args.analysis_only:
        print(f"\n{'='*64}\nFIX  Tiered repair attempts (max tier = {args.fix_max_tier})\n{'='*64}")
        outcome = loop.run_fix(report, cases)
        for v in outcome.attempted:
            tag = "FIXED" if v.fixed else "no"
            e_str = f"{v.e_value:.2f}" if v.e_value is not None else "n/a"
            print(f"  [{v.candidate.tier.label:3s}] {v.candidate.name:24s} "
                  f"({v.candidate.source})  fixed={tag:5s} "
                  f"repairs={v.n_fixed} breaks={v.n_broken} e={e_str}")
        if outcome.fixed and outcome.best is not None:
            best = outcome.best
            print(f"  VERDICT    : fixed by [{best.candidate.tier.label}] "
                  f"{best.candidate.name} (effect={best.effect:+.3f}, "
                  f"repaired {best.fixed_cases})")
        elif outcome.recommendation is not None:
            rec = outcome.recommendation
            print(f"  VERDICT    : not fixed within {args.fix_max_tier}")
            print(f"  RECOMMEND  : raise the intervention tier to {rec['recommend_tier']}")
        else:
            print(f"  VERDICT    : not fixed; already at the highest tier ({args.fix_max_tier})")

    ctx.finalize()
    print(f"\n  Full guide -> {ctx.root / 'README.txt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
