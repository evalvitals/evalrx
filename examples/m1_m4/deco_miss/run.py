"""deco_miss — Mode-1 container input: the POPE MISS subset -> VLDiagnoseLoop.

Sibling of examples/m1_m4/deco_pope, but the failure slice is the *miss* (a present
object answered "No") rather than the adversarial hallucination. This is the
slice where an internals-write fix can actually work, so the loop should be
able to close detect -> analyse -> *validated* repair.

    1. load data/cases/{model}.json   (present-probe slice, build_cases.py)
    2. drift check                    (re-generate a sample, compare frozen labels)
    3. ExperimentProtocol             (OBSERVATION-ONLY — no mechanism named)
    4. VLDiagnoseLoop M1->M5          (loop selects its own analyzers)
    5. run_m4 + run_fix               (loop proposes + validates its own fix)

The protocol describes only what is OBSERVED (which answers are wrong, on what
inputs). It must NOT name a mechanism or cause — that would hand the loop the
answer. Any reference analysis lives outside this repo.

Usage:
    python build_cases.py                                   # once, offline
    python run.py --model qwen3-vl-2b-instruct --device cuda
    python run.py --smoke-test
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

HERE = Path(__file__).parent
DATA = HERE / "data"
IMAGES = HERE.parent / "deco_pope" / "data" / "images"   # shared, not copied
OUT = HERE / "outputs"
CFG = yaml.safe_load((HERE / "config.yaml").read_text())


# ---------------------------------------------------------------------------
# 1. Frozen manifest -> CaseBatch (mixed PASS/FAIL — M2/M5 need both groups)
# ---------------------------------------------------------------------------

def load_manifest(model_key: str, max_correct: int = 0, seed: int = 42):
    """Build the CaseBatch; optionally cap the correct (PASS) cases for speed.

    Every MISS (FAIL) is kept; PASS cases are subsampled (split-stratified) to
    ``max_correct`` so the enriched batch stays balanced enough for the group
    contrast without re-generating the whole present-probe pool.
    """
    import random as _random

    from PIL import Image

    from evalrx.core.case import CaseBatch, FailureCase, Inputs, Label

    path = DATA / "cases" / f"{model_key}.json"
    if not path.exists():
        raise SystemExit(f"{path} missing — run `python build_cases.py` first")
    raw = json.loads(path.read_text())
    rows = raw["cases"]
    if max_correct:
        fails = [r for r in rows if r["label"] == "fail"]
        passes = [r for r in rows if r["label"] == "pass"]
        rng = _random.Random(seed)
        kept_pass = []
        for split, frac in (("explore", 0.6), ("validate", 0.4)):
            pool = [r for r in passes if r["split"] == split]
            rng.shuffle(pool)
            kept_pass += pool[: round(max_correct * frac)]
        rows = fails + kept_pass
        print(f"subset: {len(fails)} miss + {len(kept_pass)} correct -> {len(rows)} cases")

    cases = []
    pil_cache: dict[str, object] = {}
    for r in rows:
        if r["file_name"] not in pil_cache:
            pil_cache[r["file_name"]] = Image.open(IMAGES / r["file_name"]).convert("RGB")
        cases.append(FailureCase(
            inputs=Inputs(prompt=raw["prompt_template"].format(obj=r["object"]),
                          image=pil_cache[r["file_name"]]),
            expected=r["pope_label"],
            observed=r["observed"],
            label=Label.FAIL if r["label"] == "fail" else Label.PASS,
            tags={"miss", "object-presence"},
            metadata={k: r[k] for k in (
                "image_id", "object", "probe_type", "pope_label",
                "gt_token_ids", "out_token_ids", "split")},
        ))
    return CaseBatch(cases), raw


def drift_check(model, cases, n: int = 10) -> None:
    from evalrx.analyzers.hallucination.pope import parse_yes_no
    stale = 0
    for case in list(cases)[:n]:
        if parse_yes_no(model.generate(case.inputs)) != parse_yes_no(case.observed):
            stale += 1
    if stale:
        print(f"[WARN] {stale}/{n} frozen labels no longer reproduce — re-run "
              f"deco_pope/mine_cases.py + build_cases.py (check transformers version)")


# ---------------------------------------------------------------------------
# 2. Judge + Protocol (OBSERVATION ONLY — naming a mechanism leaks the answer)
# ---------------------------------------------------------------------------

def build_judge(model_name: str, effort: str):
    from evalrx.eval_agent import ClaudeModel

    judge = ClaudeModel(model=model_name, effort=effort)
    if not judge.generate("Reply with exactly the word OK").strip():
        raise SystemExit(f"judge probe: claude --model {model_name} returned empty "
                         f"(rate-limited?) — try --judge-model sonnet or haiku")
    print(f"judge: claude model={model_name} effort={effort or 'default'}")
    return judge


def build_protocol():
    from evalrx.eval_agent.stages.protocol import ExperimentProtocol

    # OBSERVATION ONLY: state the wrong-answer pattern and the input conditions.
    # Do NOT name a suspected mechanism (layers, suppression, priors, DeCo) —
    # that is the loop's job to discover; supplying it would leak the answer.
    return ExperimentProtocol(
        description=(
            "For yes/no questions of the form 'Is there a {object} in the image?', "
            "the VLM sometimes answers 'No' for an object that IS visibly present "
            "in the image (a missed detection). Failure cases are present-object "
            "questions the model answers 'No'; success cases are present-object "
            "questions it answers 'Yes', drawn from the same images and question "
            "template."
        ),
        task_domain="object-presence recognition",
        success_criteria="parse_yes_no(answer) matches the POPE gold label ('yes')",
        failure_patterns=(
            "the answer is 'No' although the queried object is actually in the "
            "image; the same model answers many other present-object questions "
            "correctly"
        ),
        target_modalities=frozenset({"text", "image"}),
    )


# ---------------------------------------------------------------------------
# 3. Loop wiring (mirrors examples/m1_m4/deco_pope/run.py)
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=CFG["model"])
    ap.add_argument("--max-cycles", type=int, default=2)
    ap.add_argument("--max-analyzers", type=int, default=3)
    ap.add_argument("--smoke-test", action="store_true")
    ap.add_argument("--skip-m4", action="store_true")
    ap.add_argument("--judge-model", default=CFG.get("judge_model", "claude-opus-4-8"))
    ap.add_argument("--judge-effort", default=CFG.get("judge_effort", "low"))
    ap.add_argument("--max-correct", type=int, default=CFG.get("max_correct", 120),
                    help="cap PASS (correct) cases; every MISS is always kept")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16")
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)

    if args.smoke_test:
        exists = (DATA / "cases" / f"{args.model}.json").exists()
        print("smoke ok" if exists else "smoke: no manifest yet (run build_cases.py)")
        return

    from evalrx import compose
    from evalrx.core.capability import Capability
    from evalrx.eval_agent import (
        CliAgentConfig,
        ExperimentWriterConfig,
        FixAgent,
        RunContext,
        SurgeryAgent,
        VLDiagnoseLoop,
    )
    from evalrx.eval_agent.stages.diagnosis import DiagnosisAgent
    from evalrx.eval_agent.stages.probe_agent import ProbeAgent
    from evalrx.analysis.stats_agent import StatsAnalysisAgent
    from evalrx.models.backends.base import RuntimeConfig

    judge = build_judge(args.judge_model, args.judge_effort)

    model = compose(args.model, "hf_local",
                    runtime=RuntimeConfig(device=args.device, dtype=args.dtype),
                    want={Capability.GENERATE, Capability.HIDDEN_STATES,
                          Capability.ATTENTION})
    cases, raw = load_manifest(args.model, max_correct=args.max_correct)
    print(f"cases={len(list(cases))} yields={raw['yields']}")
    drift_check(model, cases)

    codegen_effort = str(CFG.get("codegen_effort", "") or "")
    codegen = CliAgentConfig(
        provider="claude_code",
        model=str(CFG.get("codegen_model", "claude-opus-4-8")),
        max_budget_usd=float(CFG.get("codegen_budget_usd", 2.0)),
        timeout_sec=int(CFG.get("codegen_timeout_sec", 240)),
        extra_args=(("--effort", codegen_effort) if codegen_effort else ()),
    )
    print(f"codegen: claude_code model={codegen.model} effort={codegen_effort or 'default'}")
    ctx = RunContext(
        OUT, verbose=True,
        config={"model": args.model, "judge_model": args.judge_model, "max_cycles": args.max_cycles},
    )
    loop = VLDiagnoseLoop(
        model=model,
        probe_agent=ProbeAgent(judge=judge, max_analyzers=args.max_analyzers,
                               allow_codegen=True, codegen_config=codegen),
        stats_agent=StatsAnalysisAgent(judge=judge, allow_codegen=True,
                                       codegen_config=codegen, figure_dir=str(ctx.figures_dir)),
        diagnosis_agent=DiagnosisAgent(judge=judge),
        surgery_agent=SurgeryAgent(
            judge=judge, writer_config=ExperimentWriterConfig(cli_agent=codegen),
            run_context=ctx),
        fix_agent=FixAgent(judge=judge,
                           max_tier=str(CFG.get("fix_max_tier", "L3b")),
                           cli_config=codegen, run_logger=ctx.logger,
                           max_validation_cases=int(CFG.get("fix_validation_cases", 60)),
                           exec_timeout_sec=int(CFG.get("fix_exec_timeout_sec", 900)),
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
