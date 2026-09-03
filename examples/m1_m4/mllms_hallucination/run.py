"""VLDiagnoseLoop: POPE adversarial object hallucination failure.

This example loads the POPE adversarial split (Li et al., EMNLP 2023) and
runs VLDiagnoseLoop to discover and diagnose why a VLM produces wrong answers
to yes/no object-presence questions.

The POPE adversarial split is the hardest partition: absent objects are
sampled from objects that frequently co-occur with objects that *are* in the
image.  Models systematically answer "Yes" to absent objects here more than
on the random or popular splits.

Setup (run once, outside Docker):
    python examples/m1_m4/mllms_hallucination/setup_data.py

Usage (inside Docker):
    docker compose run --rm mllms_hallucination
    docker compose run --rm mllms_hallucination python run.py --max-data-cases 100
"""

from __future__ import annotations

import json
import re
import textwrap
from pathlib import Path
from typing import Any

_OUTPUTS_DIR = Path(__file__).parent / "outputs"
_POPE_ROOT = Path("/data/rjin02/evalrx/pope_coco")
_DEFAULT_POPE_JSONL = _POPE_ROOT / "coco_pope_adversarial.json"
_DEFAULT_IMAGE_DIR = _POPE_ROOT / "images"


# ---------------------------------------------------------------------------
# Scorer
# ---------------------------------------------------------------------------

def _parse_yes_no(text: str) -> str | None:
    """Extract 'yes' or 'no' from the first few tokens of a VLM response."""
    prefix = text.strip().lower()[:40]
    if re.search(r"\byes\b", prefix):
        return "yes"
    if re.search(r"\bno\b", prefix):
        return "no"
    return None


def _score_case(case: Any, observed: Any) -> Any:
    from evalrx.core.case import Label

    pred = _parse_yes_no(str(observed))
    gold = str(case.expected).strip().lower()
    if pred is None:
        return Label.UNKNOWN
    return Label.PASS if pred == gold else Label.FAIL


# ---------------------------------------------------------------------------
# Case loading
# ---------------------------------------------------------------------------

def _build_pope_cases(args: Any) -> list:
    """Load POPE adversarial cases from local JSONL + images."""
    from PIL import Image

    from evalrx.core.case import FailureCase, Inputs

    jsonl_path = Path(args.pope_jsonl)
    image_dir = Path(args.image_dir)

    if not jsonl_path.exists():
        raise SystemExit(
            f"POPE JSONL not found: {jsonl_path}\n"
            "Run `python examples/m1_m4/mllms_hallucination/setup_data.py` first."
        )
    if not image_dir.exists():
        raise SystemExit(
            f"Image directory not found: {image_dir}\n"
            "Run `python examples/m1_m4/mllms_hallucination/setup_data.py` first."
        )

    raw_records: list[dict] = []
    with open(jsonl_path, encoding="utf-8") as fh:
        content = fh.read().strip()
    # Support both JSONL (one JSON object per line) and a JSON array
    if content.startswith("["):
        raw_records = json.loads(content)
    else:
        for line in content.splitlines():
            line = line.strip()
            if line:
                raw_records.append(json.loads(line))

    if args.max_data_cases:
        raw_records = raw_records[: args.max_data_cases]

    cases: list[FailureCase] = []
    missing: list[str] = []
    pil_cache: dict[str, Any] = {}

    for rec in raw_records:
        fname = rec["image"]
        img_path = image_dir / fname
        if not img_path.exists():
            missing.append(fname)
            continue
        if fname not in pil_cache:
            pil_cache[fname] = Image.open(img_path).convert("RGB")
        question_id = rec.get("question_id", len(cases))
        gold_label = str(rec.get("label", "")).lower()
        cases.append(
            FailureCase(
                id=f"pope_{question_id}",
                inputs=Inputs(
                    prompt=rec["text"] + " Please answer with Yes or No.",
                    image=pil_cache[fname],
                ),
                expected=gold_label,
                tags={"hallucination", "pope", "adversarial"},
                metadata={
                    "question_id": question_id,
                    "image": fname,
                    "category": rec.get("category", "adversarial"),
                    "pope_label": gold_label,
                },
            )
        )

    if missing:
        print(
            f"  WARNING: {len(missing)} image(s) missing from {image_dir}. "
            "Re-run setup_data.py to download them."
        )
    if not cases:
        raise SystemExit(
            f"No cases loaded from {jsonl_path}. "
            "Check that images are present in {image_dir}."
        )
    return cases


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------

def _build_protocol():
    from evalrx.eval_agent import ExperimentProtocol

    return ExperimentProtocol(
        description=(
            "A vision-language model answers yes/no questions of the form "
            "'Is there a [object] in the image?' using the POPE adversarial "
            "split.  In this split, the absent objects queried are specifically "
            "chosen to be objects that frequently co-occur with objects that "
            "ARE present in the image (e.g., keyboard is present, model is "
            "asked about mouse).  The model produces incorrect 'Yes' answers "
            "at a substantially higher rate on this adversarial split than on "
            "random absent-object probes.  The goal is to identify what "
            "internal mechanism drives this systematic error and find a "
            "training-free intervention that reduces it."
        ),
        task_domain="object hallucination / POPE adversarial",
        success_criteria=(
            "Model's first-token 'Yes' or 'No' matches the ground-truth label "
            "in the POPE annotation."
        ),
        failure_patterns=(
            "Model answers 'Yes' to absent objects, especially those that "
            "co-occur statistically with present objects.  Failures are "
            "concentrated on adversarial probes; random-split accuracy is "
            "typically higher."
        ),
        target_modalities=frozenset({"text", "image"}),
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _bar(title: str, width: int = 64) -> None:
    pad = width - len(title) - 4
    print(f"\n-- {title} {'-' * max(pad, 2)}")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="VLDiagnoseLoop - POPE adversarial object hallucination"
    )
    parser.add_argument("--model", default="qwen3-vl-4b-instruct")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument(
        "--judge-model", default="Gemini 3.1 Pro (Low)",
        help="agy model for M1-M5 judge.",
    )
    parser.add_argument("--max-cycles", type=int, default=3)
    parser.add_argument("--max-analyzers", type=int, default=3)
    parser.add_argument(
        "--pope-jsonl",
        default=str(_DEFAULT_POPE_JSONL),
        help="Path to POPE adversarial JSONL annotation file.",
    )
    parser.add_argument(
        "--image-dir",
        default=str(_DEFAULT_IMAGE_DIR),
        help="Directory containing COCO val2014 images.",
    )
    parser.add_argument(
        "--max-data-cases",
        type=int,
        default=None,
        help="Cap on number of POPE cases to load (default: all).",
    )
    parser.add_argument("--run-dir", default=str(_OUTPUTS_DIR))
    args = parser.parse_args()

    import evalrx
    from evalrx.eval_agent import (
        AgyModel,
        CaseDiscoveryAgent,
        CliAgentConfig,
        DiagnosisAgent,
        ExperimentWriterConfig,
        FixAgent,
        HypothesisTester,
        ProbeAgent,
        RunContext,
        StatsAnalysisAgent,
        SurgeryAgent,
        VLDiagnoseLoop,
    )

    print(f"\nLoading {args.model!r} ...")
    model = evalrx.load(
        args.model,
        backend="hf_local",
        device=args.device,
        dtype=args.dtype,
        want=["attention"],
    )

    try:
        judge = AgyModel(model=args.judge_model)
        judge_desc = f"{args.judge_model}  (antigravity)"
    except RuntimeError as agy_err:
        import warnings as _w

        _w.warn(
            f"agy not available ({agy_err}). Falling back to loaded model as judge.",
            stacklevel=2,
        )
        judge = model
        judge_desc = f"{args.model}  (fallback; agy unavailable)"

    protocol = _build_protocol()
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"  model       {args.model}  [{args.device} / {args.dtype}]")
    print(f"  judge       {judge_desc}")
    print(f"  data        {Path(args.pope_jsonl).resolve()}")
    print(f"  image_dir   {Path(args.image_dir).resolve()}")
    print(f"  output      {run_dir.resolve()}")

    candidates = _build_pope_cases(args)

    discovery = CaseDiscoveryAgent(
        scorer=_score_case,
        include_unknown=False,
    ).discover(model, candidates, protocol=protocol)
    cases = discovery.cases

    discovery_rows = []
    for case in cases:
        observed = str(case.observed)
        discovery_rows.append({
            "id": case.id,
            "prompt": case.inputs.prompt,
            "expected": case.expected,
            "observed": observed,
            "pred_yes_no": _parse_yes_no(observed),
            "label": case.label.value,
            "metadata": getattr(case, "metadata", {}),
        })

    _bar(f"CASES  {len(cases)} total / {discovery.n_fail} fail / {discovery.n_pass} pass")
    id_w = max((len(c.id) for c in cases), default=20)
    for case in cases[:40]:
        label = "FAIL" if case.label.value == "fail" else "pass"
        obs = textwrap.shorten(str(case.observed), width=70, placeholder="...")
        gold = case.expected
        print(f"  {label}  {case.id:<{id_w}}  gold={gold:<3}  got: {obs!r}")
    if len(cases) > 40:
        print(f"  ... ({len(cases) - 40} more)")

    if not discovery.has_m5_groups:
        print(
            "\n  WARNING: need both FAIL and PASS cases for diagnosis. "
            "Try --max-data-cases with a larger sample."
        )

    ctx = RunContext(
        run_dir,
        verbose=True,
        config={
            "model": args.model,
            "judge": judge_desc,
            "max_cycles": args.max_cycles,
            "max_analyzers": args.max_analyzers,
        },
    )
    probe_agent = ProbeAgent(judge=judge, max_analyzers=args.max_analyzers)
    stats_agent = StatsAnalysisAgent(
        judge=judge,
        figure_dir=str(ctx.figures_dir),
    )
    diagnosis_agent = DiagnosisAgent(judge=judge)
    hypothesis_tester = HypothesisTester(judge=judge, min_effect=0.05)
    writer_cfg = ExperimentWriterConfig(
        cli_agent=CliAgentConfig(
            provider="antigravity", timeout_sec=300, model=args.judge_model
        ),
        exec_fix_timeout_sec=90,
    )
    surgery_agent = SurgeryAgent(judge=judge, writer_config=writer_cfg, run_context=ctx)
    fix_agent = FixAgent(
        judge=judge,
        score_fn=_score_case,
        run_logger=ctx.logger,
        cli_config=CliAgentConfig(
            provider="antigravity", timeout_sec=300, model=args.judge_model
        ),
        allow_codegen=True,
        exec_timeout_sec=300,
        run_context=ctx,
    )

    loop = VLDiagnoseLoop(
        model=model,
        protocol=protocol,
        probe_agent=probe_agent,
        stats_agent=stats_agent,
        diagnosis_agent=diagnosis_agent,
        hypothesis_tester=hypothesis_tester,
        surgery_agent=surgery_agent,
        fix_agent=fix_agent,
        max_cycles=args.max_cycles,
        run_logger=ctx.logger,
    )

    _bar(f"RUNNING  max {args.max_cycles} cycles / {args.max_analyzers} analyzers")
    report = loop.run(cases)
    ctx.write_diagnose_report(report, cases, discovery=discovery_rows)

    n_verified = len(report.verified_hypotheses)
    _bar(f"DIAGNOSIS  {report.cycles} cycle(s) / stopped: {report.stopped_by}")
    print(f"  {len(report.all_hypotheses)} hypothesis/es generated / {n_verified} verified")
    for vr in report.verified_hypotheses:
        stmt = textwrap.shorten(vr.hypothesis.statement, width=90, placeholder="...")
        print(f"\n  SUPPORTED: {stmt}")
        print(
            f"    effect {vr.effect_size}  confidence {vr.confidence:.2f}  "
            f"{'protocol-consistent' if vr.is_consistent_with_protocol else 'protocol-inconsistent'}"
        )
        if vr.verdict:
            print(f"    {textwrap.shorten(str(vr.verdict), width=90, placeholder='...')}")

    _bar("EXPERIMENT  (M4 mechanism verification)")
    if report.verified_hypotheses:
        fix = loop.run_m4(report, cases)
        if fix is not None:
            print(f"  verdict   {fix.status.value.upper()}")
            skip = {"validation_log", "llm_calls", "sandbox_runs", "returncode", "timed_out"}
            for k, v in (fix.evidence or {}).items():
                if k not in skip:
                    print(f"  {k:<22}  {v}")
        else:
            print("  no result returned")
    else:
        print("  skipped: no verified hypotheses")

    _bar("FIX  (auto-escalating / L2 -> L3b)")
    if report.verified_hypotheses:
        fix_outcome = loop.run_fix(report, cases, auto_escalate=True, max_tier="L3b")
        if fix_outcome is not None:
            n_fail_cases = sum(1 for c in cases if c.label.value == "fail")
            best = fix_outcome.best
            if fix_outcome.fixed and best is not None:
                cand = getattr(best, "candidate", None)
                name = getattr(cand, "name", "?")
                tier = getattr(cand, "tier", "?")
                print(f"  {best.n_fixed} of {n_fail_cases} failure cases fixed / {best.n_broken} regression(s)")
                print(f"  method  {name}  [{tier}]")
                print(f"  effect  {best.effect}")
            else:
                print(f"  no fix found  ({len(fix_outcome.attempted or [])} approach(es) tried)")
            if fix_outcome.recommendation:
                print(f"  recommendation  {fix_outcome.recommendation}")
        else:
            print("  no result returned")
    else:
        print("  skipped: no verified hypotheses")

    ctx.finalize()
    print(f"\n  Full guide -> {ctx.root / 'README.txt'}")
    print("\nDone.")


if __name__ == "__main__":
    main()
