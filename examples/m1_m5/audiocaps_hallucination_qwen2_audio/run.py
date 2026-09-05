#!/usr/bin/env python3
"""audiocaps_hallucination_qwen2_audio — frozen Q/A + protocol -> VLDiagnoseLoop -> AAD.

Qwen2-Audio-7B-Instruct answers discriminative yes/no questions ("Is there a
sound of X in the audio?") over AudioCaps clips (Kuan et al. 2024,
Interspeech 2024, arXiv:2406.08402 -- the exact benchmark AAD, Hsu et al.
2025 arXiv:2506.07233, evaluates against, on this exact model). This
run.py only supplies INPUTS -- frozen cases + an observation-only protocol;
detection, diagnosis and repair are the loop's own job. No hand-supplied
hypothesis, no forced candidate (unless --paper-method-only locks the pool,
see below): the point is finding out whether the loop discovers "language
priors override audio evidence" on its own and reaches for AAD.

    1. download_audiohallucination.py --limit 300   (run first, standalone --
                                                       writes data/audiohallucination.jsonl
                                                       + data/audio/*.wav)
    2. Fresh baseline pass        plain model.generate() (greedy) on every row
                                   -> PASS/FAIL labels, computed live every run,
                                   never cached
    3. ExperimentProtocol         OBSERVATION ONLY -- describes the task
                                   (binary object-presence judgment) and the
                                   measured baseline accuracy; does not name
                                   "language priors override audio evidence"
                                   or point at AAD. That mechanism, if it
                                   appears, has to come out of M3.
    4. VLDiagnoseLoop M1->M4      M1 uses a PINNED static analyzer set (see
                                   PINNED_M1_ANALYZERS below); M2
                                   StatsAnalysisAgent (e-BH-corrected); M3
                                   DiagnosisAgent (Claude judge); M4
                                   HypothesisTester (protocol-consistency +
                                   statistical significance)
    5. loop.run_fix               FixAgent's tiered candidates -- including
                                   the paper-registered aad_silence_contrast /
                                   aad_silence_contrast_gated_false_yes L0
                                   candidates (fix_agent.py's admission gate:
                                   tasks=={"yes_no"} + generate_aad +
                                   paper_method_fidelity('aad')=='native_silence_contrast'
                                   -- see evalrx/eval_agent/stages/fix_agent.py,
                                   search "aad_silence_contrast"), validated
                                   on a held-out CONFIRM split the loop's own
                                   M1-M4 discovery never saw (--confirm-split).

Usage:
    python download_audiohallucination.py --limit 300 --scan-rows 2000
    python run.py --model qwen2-audio-7b-instruct --limit 300
    python run.py --smoke-test     # fast wiring check, no GPU/model/judge --
                                    # exercises the REAL VLDiagnoseLoop + FixAgent
                                    # against a synthetic model+cases, specifically
                                    # to catch the aad_silence_contrast admission
                                    # gate silently never firing (wrong metadata
                                    # key, wrong max_tier, model missing generate_aad).
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
MANIFEST = DATA / "audiohallucination.jsonl"


# ---------------------------------------------------------------------------
# Dataset glue
# ---------------------------------------------------------------------------

def load_records(limit: int | None = None) -> list[dict[str, Any]]:
    if not MANIFEST.is_file():
        raise SystemExit(f"{MANIFEST} does not exist -- run download_audiohallucination.py first")
    records = [json.loads(line) for line in MANIFEST.read_text(encoding="utf-8").splitlines() if line]
    return records[:limit] if limit else records


def shuffled_rows(rows: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    shuffled = list(rows)
    random.Random(seed).shuffle(shuffled)
    return shuffled


def task_prompt(row: dict[str, Any]) -> str:
    return row["instruction"]


def parsed_yes_no(output: str) -> str:
    m = re.search(r"\b(yes|no)\b", output, re.IGNORECASE)
    return m.group(1).capitalize() if m else ""


def score(row: dict[str, Any], output: str) -> bool:
    return parsed_yes_no(output) == str(row["expected"]).strip().capitalize()


def score_case(case: "Any", output: str) -> bool:
    return score({"expected": case.expected}, output)


def row_inputs(row: dict[str, Any]) -> "Any":
    from evalrx.core.case import Inputs

    return Inputs(prompt=task_prompt(row), audio=str(DATA / row["audio_path"]))


def evaluate(rows: list[dict[str, Any]], strategy: Callable[[dict[str, Any]], str]) -> dict[str, Any]:
    """One fresh generation pass over *rows* -- never cached across runs."""
    cases = []
    for row in rows:
        output = strategy(row)
        cases.append({"id": row["id"], "output": output, "correct": score(row, output)})
    correct = sum(case["correct"] for case in cases)
    return {"n": len(cases), "correct": correct, "accuracy": correct / len(cases), "cases": cases}


def make_cases(rows: list[dict[str, Any]], baseline: dict[str, Any]) -> "Any":
    from evalrx.core.case import CaseBatch, FailureCase, Label

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
# M1 -- pinned analyzer set
# ---------------------------------------------------------------------------
#
# Same reasoning as mmau_qwen2_audio's PINNED_M1_ANALYZERS (LLM-guided
# catalog selection isn't safely audited against a live Qwen2-Audio handle
# yet), minus format_sensitivity: that analyzer targets 4-way multiple-choice
# OPTION-ORDER bias, which doesn't exist on a binary yes/no task -- keeping
# it would just add a meaningless "no options to reorder" no-op.
PINNED_M1_ANALYZERS = [
    "answer_extraction_audit",   # is the FAIL label real or a parse miss?
    "termination_audit",          # truncated / degenerate / gave up?
    "selfcheck_consistency",      # black-box hallucination signal
    "self_consistency",           # resample stability
    "calibration",                 # stated/derived confidence vs correctness
    "logprob_entropy",            # answer-token uncertainty
    "perturbation_battery",       # invariance under paraphrase/no-op edits
]


def build_protocol() -> "Any":
    from evalrx.eval_agent import ExperimentProtocol

    return ExperimentProtocol(
        description=(
            "We evaluate an audio-language model (Qwen2-Audio-7B-Instruct) on "
            "a discriminative object-hallucination benchmark: for a short "
            "AudioCaps clip, the model is asked a binary yes/no question about "
            "whether a specific sound is present ('Is there a sound of X in "
            "the audio?'). Half the questions name a sound that IS present, "
            "half name one that is absent (random/popular/adversarial "
            "negative sampling). We want to know what distinguishes the "
            "questions it gets right from the ones it gets wrong."
        ),
        task_domain="audio object-hallucination detection (AudioCaps discriminative)",
        success_criteria=(
            "The model's Yes/No answer must match the gold label for whether "
            "that sound is actually present in the clip."
        ),
        failure_patterns=(
            "wrong answers concentrated on absent-sound (gold=No) questions, "
            "where the model answers Yes anyway -- a model that asserts sounds "
            "it has not actually heard, falling back on what the question "
            "text alone makes plausible rather than the audio evidence, will "
            "show exactly this asymmetric pattern; a fix that trades away "
            "present-sound (gold=Yes) accuracy to gain absent-sound accuracy "
            "is a different error, not an improvement"
        ),
        target_modalities=frozenset({"text", "audio"}),
    )


def build_judge(provider: str, model_name: str, effort: str) -> "Any":
    if provider == "agy":
        from evalrx.agent_runtime.judges import AgyModel

        judge = AgyModel(model=model_name, timeout_sec=300)
        label = f"agy model={model_name or 'session default'}"
        empty_hint = "agy is likely rate-limited/quota-exhausted -- try --judge-model with a different agy model"
    else:
        from evalrx.eval_agent import ClaudeModel

        model_name = model_name or "sonnet"
        judge = ClaudeModel(model=model_name, effort=effort)
        label = f"claude model={model_name} effort={effort or 'default'}"
        empty_hint = f"claude --model {model_name} returned empty (rate-limited?) -- try --judge-model sonnet or haiku"
    if not judge.generate("Reply with exactly the word OK").strip():
        raise SystemExit(f"judge probe: {empty_hint}")
    print(f"judge: {label}")
    return judge


# ---------------------------------------------------------------------------
# Smoke test -- real VLDiagnoseLoop + FixAgent, synthetic model/judge
# ---------------------------------------------------------------------------

def _run_smoke_test(args: argparse.Namespace) -> None:
    from evalrx.core.capability import Capability
    from evalrx.core.case import CaseBatch, FailureCase, Inputs, Label
    from evalrx.core.result import Result
    from evalrx.eval_agent import (
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
        """Baseline always answers 'Yes' (over-affirms, AAD's real target
        direction); generate_aad 'hears' GOLD= in the prompt."""

        def __init__(self) -> None:
            self.capabilities = frozenset({Capability.GENERATE, Capability.LOGPROBS})
            self.modalities = frozenset({"text", "audio"})

        def __repr__(self) -> str:
            return "SmokeAudioModel()"

        def generate(self, inputs, **kwargs) -> str:
            return "Yes"

        def generate_aad(self, inputs, **kwargs) -> str:
            m = re.search(r"GOLD=(Yes|No)", str(getattr(inputs, "prompt", inputs)))
            return m.group(1) if m else "Yes"

        def paper_method_fidelity(self, name: str) -> str:
            return "native_silence_contrast" if name == "aad" else "unavailable"

        def forward(self, inputs, capture, spec=None):
            raise NotImplementedError("SmokeAudioModel only supports generate().")

    def _smoke_case(id_: str, gold: str) -> FailureCase:
        # Baseline ("Yes" always) only matches gold=="Yes" rows -> PASS; the
        # gold=="No" rows FAIL as false-Yes -- object hallucination, AAD's
        # actual target direction (over-affirming a sound that isn't there).
        return FailureCase(
            id=id_,
            inputs=Inputs(prompt=f"Is there a sound of X? GOLD={gold}", audio="smoke://clip.wav"),
            expected=gold,
            observed="Yes",
            label=Label.PASS if gold == "Yes" else Label.FAIL,
            metadata={"task": "yes_no"},
        )

    cases = CaseBatch([
        _smoke_case("s0", "Yes"),
        _smoke_case("s1", "Yes"),
        _smoke_case("s2", "No"),
        _smoke_case("s3", "No"),
    ])
    model = _SmokeAudioModel()

    class _SmokeProbe:
        last_schema = None

        def probe(self, model, data, **kwargs):
            fail_ids = [c.id for c in data if c.label == Label.FAIL]
            self.last_schema = ProbingSchema(
                selected_analyzers=["answer_extraction_audit"],
                rationale="Smoke probe marks discovered failures as a signal.",
                protocol=kwargs.get("protocol"),
            )
            findings = {
                "per_case": [{"sample_id": cid, "defaults_to_yes": True} for cid in fail_ids],
            }
            return {"answer_extraction_audit": Result(
                analyzer="answer_extraction_audit", model=repr(model), cases=data, findings=findings,
            )}

    class _SmokeDiagnosisAgent:
        def diagnose(self, analysis, prior_cycles=None):
            h = Hypothesis(
                statement="The model defaults to Yes when it cannot resolve the clip, "
                          "overriding audio evidence with a language-plausibility guess.",
                target_model=analysis.model_name,
                predicted_failure_mode="language_prior_bias",
            )
            return DiagnosisResult(
                model_name=analysis.model_name,
                hypotheses=[h],
                findings_summary={n: r.findings for n, r in analysis.raw_results.items()},
                raw_judge_output="HYPOTHESIS: defaults to Yes.\nFAILURE_MODE: language_prior_bias",
            )

    ctx = RunContext(args.run_dir, verbose=True, config={"smoke_test": True})
    loop = VLDiagnoseLoop(
        model=model,
        protocol=ExperimentProtocol(
            description="Smoke: synthetic binary audio hallucination QA.",
            task_domain="audio object-hallucination detection (smoke)",
            success_criteria="the Yes/No answer must match GOLD.",
            # Shares a 6+ char word ("overriding") with the smoke diagnosis
            # agent's hypothesis text below -- HypothesisTester's
            # judge=None fallback (_heuristic_consistency_check) is a plain
            # word-overlap check against description+failure_patterns, so an
            # empty/unrelated failure_patterns here makes every hypothesis
            # protocol_consistent=False regardless of content, silently
            # emptying report.verified_hypotheses and starving run_fix of
            # anything to route candidates against (caught live: this exact
            # gap already exists in examples/m1_m5/mmau_qwen2_audio's smoke
            # test too, pre-existing there, not introduced here).
            failure_patterns="the model overriding audio evidence with a default guess",
            target_modalities=frozenset({"text", "audio"}),
        ),
        probe_agent=_SmokeProbe(),
        stats_agent=StatsAnalysisAgent(stats_tool_agent=StatsToolAgent(max_tools=3)),
        diagnosis_agent=_SmokeDiagnosisAgent(),
        hypothesis_tester=HypothesisTester(min_effect=0.05),
        fix_agent=FixAgent(
            score_fn=score_case,
            max_tier=args.fix_max_tier,
            candidate_allowlist=(["aad_silence_contrast"] if args.paper_method_only else None),
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
    if "aad_silence_contrast" not in names:
        raise SystemExit(
            "Smoke test FAILED: aad_silence_contrast was never proposed -- the L0 "
            "admission gate (fix_agent.py _l0_candidates) did not fire. Check "
            "case.metadata['task']=='yes_no', and the model's generate_aad/"
            "paper_method_fidelity."
        )
    aad = next(v for v in outcome.attempted if v.candidate.name == "aad_silence_contrast")
    if aad.n_fixed < 1:
        raise SystemExit("Smoke test FAILED: aad_silence_contrast fixed 0 of the synthetic FAIL cases.")
    effect_str = f"{aad.effect:+.3f}" if aad.effect is not None else "n/a"
    print(f"  aad_silence_contrast: fixed={aad.n_fixed} broken={aad.n_broken} effect={effect_str}")
    print("Smoke test passed.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="qwen2-audio-7b-instruct")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, default=300)
    parser.add_argument(
        "--max-tokens", type=int, default=16,
        help="subject-model generation budget (baseline scoring AND every M1 "
             "probe re-ask) -- a bare Yes/No answer needs very little room.",
    )
    parser.add_argument("--max-cycles", type=int, default=1)
    parser.add_argument("--max-analyzers", type=int, default=len(PINNED_M1_ANALYZERS))
    parser.add_argument(
        "--confirm-split", type=float, default=0.6,
        help="fraction of --limit rows held out from M1-M4 discovery, reserved "
             "for loop.run_fix's validation (VLDiagnoseLoop's own leak-#3 split). "
             "Ignored (forced to 0.0) under --analysis-only, since nothing "
             "downstream would ever read the held-out partition.",
    )
    parser.add_argument(
        "--fix-max-tier", default="L3a",
        help="highest intervention tier FixAgent may propose. AAD reads and "
             "combines logits from paired forwards, so it is L3a; L0 admits "
             "runtime-configuration repairs only.",
    )
    parser.add_argument(
        "--judge-provider", choices=["claude", "agy"], default="agy",
        help="'agy' (default, Antigravity CLI, no Anthropic API key -- see "
             "evalrx/agent_runtime/judges/agy.py) or 'claude' (native "
             "claude CLI). Matches vlm_benchmark_common.py's default.",
    )
    parser.add_argument(
        "--judge-model", default="",
        help="model name passed to the judge CLI. Empty = provider default "
             "(sonnet for --judge-provider claude; agy session default for "
             "--judge-provider agy).",
    )
    parser.add_argument("--judge-effort", default="high")
    parser.add_argument("--seed", type=int, default=20260814)
    parser.add_argument(
        "--paper-method-only", action="store_true",
        help="restrict the fix pool to candidate_allowlist=['aad_silence_contrast'] "
             "(the pre-registered paper method) and disable the coder. Default: "
             "every admissible candidate competes (judge L1/L2, floor, AAD, coded).",
    )
    parser.add_argument(
        "--unrestricted", action="store_true",
        help="(no-op, kept for compatibility: the open pool is the default now)",
    )
    parser.add_argument(
        "--explore", action=argparse.BooleanOptionalAction, default=True,
        help="In-cycle free-form EDA between M1 and M2 (claude coder); charts "
             "+ tables land under <run-dir>/explore. --no-explore disables.",
    )
    parser.add_argument(
        "--analysis-only", action="store_true",
        help="run M1+M2 only (skip M3/M4/fix)",
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

    from evalrx.models.backends.base import RuntimeConfig
    from evalrx.models.backends.hf_local import HFLocalModel
    from evalrx.specs import get_spec

    model = HFLocalModel(
        get_spec(args.model),
        RuntimeConfig(device=args.device, dtype="bfloat16", max_new_tokens=args.max_tokens),
    )
    model.load()
    print(f"paper_method_fidelity(aad) = {model.paper_method_fidelity('aad')!r}")

    def baseline_generate(row: dict[str, Any]) -> str:
        return model.generate(row_inputs(row), max_new_tokens=args.max_tokens)

    baseline = evaluate(rows, baseline_generate)
    print(f"baseline accuracy on {len(rows)} rows: {baseline['accuracy']:.1%} "
          f"({baseline['correct']}/{baseline['n']})")

    cases = make_cases(rows, baseline)
    # The aad_silence_contrast admission gate (fix_agent.py _l0_candidates) is
    # an EXACT set match on case.metadata["task"] -- if this ever drifts (a
    # case loses the key, or gains a second task tag), AAD silently stops
    # being proposed with no error anywhere downstream. Fail loud, here, first.
    tasks = {c.metadata.get("task") for c in cases}
    assert tasks == {"yes_no"}, (
        f"expected every case to carry metadata['task']=='yes_no', got {tasks} "
        "-- aad_silence_contrast will never be proposed by loop.run_fix with this batch"
    )

    judge = build_judge(args.judge_provider, args.judge_model, args.judge_effort)

    from evalrx.eval_agent import (
        DiagnosisAgent,
        FixAgent,
        HypothesisTester,
        ProbeAgent,
        RunContext,
        StatsAnalysisAgent,
        StrategyProbe,
        VLDiagnoseLoop,
    )

    # llm_benchmark layout: run log + artifacts under <run-dir>/logs, explore
    # as its sibling <run-dir>/explore (dashboard merges logs*/run_log.jsonl).
    ctx = RunContext(
        Path(args.run_dir) / "logs", verbose=True,
        config={
            "model": args.model,
            "judge_provider": args.judge_provider,
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

    from evalrx.eval_agent import CliAgentConfig, SurgeryAgent
    from evalrx.eval_agent.stages.experiment_writer import ExperimentWriterConfig

    coder_cfg = CliAgentConfig(
        provider="antigravity" if args.judge_provider == "agy" else "claude_code",
        model=args.judge_model,
        timeout_sec=900,
        extra_args=(
            () if args.judge_provider == "agy"
            else (("--effort", args.judge_effort) if args.judge_effort else ())
        ),
    )
    fix_agent = FixAgent(
        judge=judge,
        max_tier=args.fix_max_tier,
        score_fn=score_case,
        run_logger=ctx.logger,
        run_context=ctx,
        candidate_allowlist=(["aad_silence_contrast"] if args.paper_method_only else None),
        # Full candidate family by default (judge L1/L2, the self_consistency
        # floor, AAD when its gate admits it, and a coded pipeline from the
        # same coder that runs explore/M5) — the llm_benchmark shape.
        # --paper-method-only restores the AAD-only pool.
        allow_codegen=not args.paper_method_only,
        cli_config=None if args.paper_method_only else coder_cfg,
    )

    # --analysis-only never reaches run_fix, so nothing would ever read the
    # held-out CONFIRM partition -- reserving one anyway would silently starve
    # M1/M2 of rows the user believes they gave it. Force it off in that mode.
    confirm_split = 0.0 if args.analysis_only else args.confirm_split
    n_confirm = round(len(cases) * confirm_split) if 0.0 < confirm_split < 1.0 else 0
    print(f"  explore/confirm split: {len(cases) - n_confirm} explore "
          f"(M1-M4 discovery) / {n_confirm} confirm (held out for run_fix)")

    explorer = None
    if args.explore and not args.analysis_only:
        from evalrx.agent_runtime.sandbox import ExperimentSandbox
        from evalrx.analysis import ExploratoryAnalysisAgent

        explorer = ExploratoryAnalysisAgent(
            cli_config=coder_cfg,
            sandbox=ExperimentSandbox(
                workdir=Path(args.run_dir) / "explore" / "sandbox",
                cleanup=False),
            timeout_sec=900,
            max_attempts=2,
        )
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
        surgery_agent=None if args.analysis_only else SurgeryAgent(
            judge=judge, writer_config=ExperimentWriterConfig(cli_agent=coder_cfg)),
        explorer=explorer,
        explore_dir=Path(args.run_dir) / "explore",
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
        print(f"\n{'='*64}\nM4  Intervention experiment\n{'='*64}")
        fix_proposal = loop.run_m5(report, cases, allow_unverified=True)
        if fix_proposal is not None:
            tag = "verified" if report.verified_hypotheses else "UNVERIFIED (best lead)"
            print(f"  M5 experiment on the {tag} hypothesis: status={fix_proposal.status}")
        else:
            print("  M5: no hypothesis to experiment on")
        print(f"\n{'='*64}\nFIX  Tiered repair attempts (max tier = {args.fix_max_tier})\n{'='*64}")
        outcome = loop.run_fix(report, cases, allow_unverified=True)
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
            if str(rec["recommend_tier"]).lower() == str(args.fix_max_tier).lower():
                print(f"  RECOMMEND  : stay within {rec['recommend_tier']} -- {rec['reason']}")
            else:
                print(f"  RECOMMEND  : raise the intervention tier to {rec['recommend_tier']} "
                      f"-- {rec['reason']}")
        else:
            print(f"  VERDICT    : not fixed; already at the highest tier ({args.fix_max_tier})")

    ctx.finalize()
    print(f"\n  Full guide -> {ctx.root / 'README.txt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
