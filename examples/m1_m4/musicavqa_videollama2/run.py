"""musicavqa_videollama2 — Mode-1 container input: frozen AVQA answers + protocol -> VLDiagnoseLoop.

Audio-visual analogue of examples/m1_m4/deco_chair: an audio-visual
LLM (VideoLLaMA2.1-7B-AV) answers Music-AVQA questions that need audio AND
video evidence (unimodal/cross-modal hallucination in AV-LLMs -- the failure
mode AVCD, arXiv:2505.20862, studies; that paper's own released code turned
out unusable here, see the Dockerfile's comment). This run.py only
supplies INPUTS — frozen cases + an observation-only protocol; detection,
diagnosis and repair are the loop's own job.

    1. load data/cases/{model}.json     (frozen Q/A + pass/fail labels, from mine_cases.py)
    2. drift check                      (re-answer a sample, compare correctness)
    3. ExperimentProtocol                (OBSERVATION ONLY — no mechanism named)
    4. VLDiagnoseLoop M1->M5             (M1 pinned to PINNED_M1_ANALYZERS, see below)
    5. run_m4 + run_fix                  (loop proposes + validates its own fix)

M1 is pinned (ProbeAgent(judge=None, probe=StrategyProbe(priority_override=...)))
rather than left to LLM-guided catalog selection. Measured on real runs: the
judge sometimes picks a weak, purely-hygiene combo (answer_extraction_audit /
arith_audit / termination_audit) that surfaces no signal at all, and
VLDiagnoseLoop.run() does not retry a fresh M1 selection on the next cycle
when M3 finds zero hypotheses (loop.py: `stopped_by=no_hypotheses` on cycle 1
is terminal, not "try again with different analyzers") -- so a weak first
pick silently ends the whole run with no fix attempted. Pinning to a
combination that DID surface a real, statistically-supported signal on this
model/dataset (self_consistency + perturbation_battery) makes the loop
reproducible instead of a coin flip on which cycle-1 draw the judge makes.

The fix module is given an exact/substring answer-match score_fn: a candidate
output counts as a success only if it still contains the gold answer for
Visual-only controls too (no free-lunch guard — a fix that improves
Audio/Audio-Visual accuracy by degrading Visual-only accuracy is a different
error, not an improvement).

Usage:
    python run.py --mock                              # CPU wiring smoke test
    python run.py --model videollama2.1-7b-av --device cuda
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from avqa_data import build_protocol, load_manifest, make_avqa_score_fn, answers_match

OUT = Path(__file__).parent / "outputs"
CFG = yaml.safe_load((Path(__file__).parent / "config.yaml").read_text())

# See the module docstring for why this is pinned rather than judge-selected.
# self_consistency + perturbation_battery are the two analyzers that produced
# a verified (statistically-supported, protocol-consistent) hypothesis on a
# real run; answer_extraction_audit/termination_audit stay as cheap hygiene
# checks (do the FAIL labels reflect a real wrong answer, or a truncated/
# degenerate one?) since a real finding is not interpretable until those are
# ruled out. prompt_contrast/cot_faithfulness are intentionally excluded here
# too -- same reasoning as examples/m1_m4/mmau_qwen2_audio/run.py: their
# default strategy templates are image-phrased ("describe what you see in the
# image..."), nonsensical prompt content for a model that only ever gets
# Inputs.video/.audio, never Inputs.image.
PINNED_M1_ANALYZERS = [
    "answer_extraction_audit",
    "termination_audit",
    "self_consistency",
    "perturbation_battery",
]


def drift_check(model, cases, n: int = 3) -> None:
    """Re-answer a few questions; warn if the pass/fail verdict flips."""
    from evalvitals.core.case import Label

    stale = 0
    for case in list(cases)[:n]:
        gold = case.metadata.get("gold_answer", "")
        now_ok = answers_match(str(model.generate(case.inputs)), gold)
        if now_ok != (case.label == Label.PASS):
            stale += 1
    if stale:
        print(f"[WARN] {stale}/{n} frozen pass/fail labels no longer reproduce — "
              f"re-run mine_cases.py")


def build_judge(model_name: str, effort: str):
    from evalvitals.eval_agent import ClaudeModel

    judge = ClaudeModel(model=model_name, effort=effort)
    if not judge.generate("Reply with exactly the word OK").strip():
        raise SystemExit(f"judge probe: claude --model {model_name} returned empty "
                          f"(rate-limited?) — try --judge-model sonnet or haiku")
    print(f"judge: claude model={model_name} effort={effort or 'default'}")
    return judge


def build_model(args):
    if args.mock:
        from videollama2_model import MockAVModel
        print("model: MockAVModel (no real weights loaded)")
        return MockAVModel()
    from videollama2_model import VideoLLaMA2AVModel
    print(f"model: VideoLLaMA2AVModel path={args.model_path} device={args.device} "
          f"want_attention={args.want_attention}")
    return VideoLLaMA2AVModel(args.model_path, device=args.device,
                               load_4bit=args.load_4bit,
                               want_attention=args.want_attention)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=CFG["model"],
                     help="manifest label (matches mine_cases.py --model)")
    ap.add_argument("--model-path", default=CFG.get("model_path"))
    ap.add_argument("--max-cycles", type=int, default=CFG.get("max_cycles", 2))
    ap.add_argument("--max-analyzers", type=int, default=3)
    ap.add_argument("--smoke-test", action="store_true",
                     help="only check the manifest exists; do not run the loop")
    ap.add_argument("--mock", action="store_true",
                     help="use MockAVModel instead of the real checkpoint (drift_check "
                          "+ any loop step that calls the model stay wiring-only)")
    ap.add_argument("--skip-m4", action="store_true")
    ap.add_argument("--judge-model", default=CFG.get("judge_model", "claude-opus-4-8"))
    ap.add_argument("--judge-effort", default=CFG.get("judge_effort", "low"))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--load-4bit", action="store_true")
    ap.add_argument("--want-attention", action="store_true",
                     help="force eager attention at load time so ATTENTION capture "
                          "(not just HIDDEN_STATES/LOGITS) is available for M1/L3a; "
                          "costs more memory/compute, off by default")
    ap.add_argument("--candidate-allowlist", default=None,
                     help="comma-separated FixCandidate names to restrict run_fix to -- "
                          "isolates a specific candidate's execution/effect from "
                          "whether the judge would have picked it unprompted. Unset "
                          "(default) = full free exploration, every admissible "
                          "candidate competes.")
    args = ap.parse_args()
    candidate_allowlist = (
        [name.strip() for name in args.candidate_allowlist.split(",") if name.strip()]
        if args.candidate_allowlist else None
    )
    OUT.mkdir(exist_ok=True)

    manifest_path = Path(__file__).parent / "data" / "cases" / f"{args.model}.json"
    if args.smoke_test:
        print("smoke ok" if manifest_path.exists()
              else f"smoke: no manifest yet at {manifest_path} (run mine_cases.py)")
        return

    from evalvitals.eval_agent import (
        CliAgentConfig,
        ExperimentWriterConfig,
        FixAgent,
        RunContext,
        SurgeryAgent,
        VLDiagnoseLoop,
    )
    from evalvitals.eval_agent.stages.diagnosis import DiagnosisAgent
    from evalvitals.eval_agent.stages.probe_agent import ProbeAgent
    from evalvitals.eval_agent.stages.probe import StrategyProbe
    from evalvitals.analysis.stats_agent import StatsAnalysisAgent

    judge = build_judge(args.judge_model, args.judge_effort)  # probe BEFORE weights load
    model = build_model(args)
    cases, raw = load_manifest(manifest_path)
    n_fail = raw["n_fail"]
    print(f"cases={len(list(cases))} fail={n_fail} pass={len(list(cases)) - n_fail}")
    drift_check(model, cases)

    ctx = RunContext(
        OUT, verbose=True,
        config={"model": args.model, "judge_model": args.judge_model,
                "max_cycles": args.max_cycles, "mock": args.mock},
    )
    codegen_effort = str(CFG.get("codegen_effort", "") or "")
    codegen = CliAgentConfig(
        provider="claude_code",
        model=str(CFG.get("codegen_model", "claude-opus-4-8")),
        max_budget_usd=float(CFG.get("codegen_budget_usd", 2.0)),
        timeout_sec=int(CFG.get("codegen_timeout_sec", 240)),
        extra_args=(("--effort", codegen_effort) if codegen_effort else ()),
    )
    print(f"codegen: claude_code model={codegen.model} effort={codegen_effort or 'default'}")
    avqa_score = make_avqa_score_fn()

    loop = VLDiagnoseLoop(
        model=model,
        probe_agent=ProbeAgent(
            probe=StrategyProbe(priority_override={k: PINNED_M1_ANALYZERS for k in ("vlm", "agent", "llm")}),
            judge=None,  # pinned static selection -- see PINNED_M1_ANALYZERS above
            max_analyzers=len(PINNED_M1_ANALYZERS),
        ),
        stats_agent=StatsAnalysisAgent(judge=judge, allow_codegen=True,
                                        codegen_config=codegen, figure_dir=str(ctx.figures_dir)),
        diagnosis_agent=DiagnosisAgent(judge=judge),
        surgery_agent=SurgeryAgent(
            judge=judge, writer_config=ExperimentWriterConfig(cli_agent=codegen),
            run_context=ctx),
        fix_agent=FixAgent(judge=judge, score_fn=avqa_score,
                            max_tier=str(CFG.get("fix_max_tier", "L2")),
                            candidate_allowlist=candidate_allowlist,
                            cli_config=codegen, run_logger=ctx.logger,
                            max_validation_cases=int(CFG.get("fix_validation_cases", 12)),
                            exec_timeout_sec=int(CFG.get("fix_exec_timeout_sec", 1200)),
                            max_repair_rounds=int(CFG.get("fix_repair_rounds", 2)),
                            run_context=ctx),
        max_cycles=args.max_cycles,
        protocol=build_protocol(),
        run_logger=ctx.logger,
    )
    report = loop.run(cases)
    print(f"cycles={report.cycles} stopped_by={report.stopped_by} "
          f"verified={len(report.verified_hypotheses)}/{len(report.all_test_results)}")
    for t in report.all_test_results:
        stmt = getattr(t.hypothesis, "statement", str(t.hypothesis))
        print(f" - [{t.status}] conf={t.confidence:.2f} grade={t.evidence_grade} {stmt[:110]}")

    if not args.skip_m4:
        fix = loop.run_m4(report, cases)
        print("m4:", fix)
        outcome = loop.run_fix(report, cases)
        print("fix outcome:", getattr(outcome, "recommendation", None) or outcome)

    ctx.write_diagnose_report(report, cases)
    ctx.finalize()


if __name__ == "__main__":
    main()
