"""VLDiagnoseLoop: TextVQA small-detail perception failure.

This example uses the real TextVQA GT-bbox data released with arXiv:2502.17422
("MLLMs Know Where to Look").  The task is to answer TextVQA questions where
the answer text occupies a very small fraction of the image.  Cases are loaded
from local JSONL manifests under /data and are partitioned by the paper's bbox
area ratio S = answer_bbox_area / image_area:

  small   S < 0.005
  medium  0.005 <= S < 0.05
  large   S >= 0.05

Usage (direct):
    python examples/m1_m5/mllms_small_object/run.py
    python examples/m1_m5/mllms_small_object/run.py --max-data-cases 32
    python examples/m1_m5/mllms_small_object/run.py --textvqa-size-split all

Investigation findings (qwen3-vl-4b-instruct, small split, 2026-06-15)
-----------------------------------------------------------------------
Model accuracy on the small split is ~79% (64/300 fail), but diagnosis
is blocked by an M4 statistical power gap:

  - The diagnosis analyzers (prompt_contrast, relative_attention) operate
    on a 32-case stratified head.  With ~6 FAIL cases in that window,
    McNemar produces only 2–3 discordant pairs — far below the threshold
    needed to reject H0 at alpha=0.05 with e-value correction.

  - Despite this, all 3 diagnosis cycles converge on the same finding:
    ``describe_first`` (ask the model to describe the image before
    answering) achieves 100% success on the stratified head vs. 93.75%
    baseline.  The mechanism hypothesis is premature text-token anchoring:
    image_token_attention_ratio ≈ 0.006, so the model defaults to its
    language prior and ignores fine-grained visual detail unless forced
    to do a visual scan first.

  - To break through the power gap: either (a) increase the analyzer
    ``max_cases`` above 32 so M4 sees more discordant pairs, or (b) run
    a direct L1 fix validation with ``describe_first`` as the prompt
    template against the full failure set (skips M4 gating entirely).

  - Two earlier bugs masked these results on prior runs:
      * ``502fdfe`` — fix-agent scoring: ``Label`` enum passed raw to
        numpy caused stats to silently fail for 26/27 fix attempts.
      * ``de3cb01`` — analyzer sampling: ``list(cases)[:32]`` head on
        a 773-case batch contained ~1 FAIL; switched to stratified_head.
"""

from __future__ import annotations

import re
import textwrap
from pathlib import Path
from typing import Any

_OUTPUTS_DIR = Path(__file__).parent / "outputs"
_TEXTVQA_ROOT = Path("/data/rjin02/evalrx/textvqa_mllms_know")
_DEFAULT_TEXTVQA_ANNOTATIONS = _TEXTVQA_ROOT / "textvqa_gt_bbox_small.jsonl"
_DEFAULT_TEXTVQA_IMAGE_DIR = _TEXTVQA_ROOT / "images"


# ---------------------------------------------------------------------------
# Scoring helpers (word-boundary safe)
# ---------------------------------------------------------------------------

def _normalize_answer(value: Any) -> str:
    text = str(value).lower()
    text = re.sub(r"\*\*|__|`", "", text)
    text = text.replace("\u201c", '"').replace("\u201d", '"').replace("\u2019", "'")
    text = re.sub(r"[^a-z0-9:%#.'+-]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip(" .,:;\"'")
    return text


def _answer_candidates(observed: str) -> list[str]:
    """Extract likely final-answer spans from verbose VLM output.

    TextVQA models often answer in prose. Scoring the whole explanation is too
    permissive because a wrong main answer can mention a gold answer as a
    rejected alternative. Prefer answer-like spans, then only short lead text.
    """
    text = str(observed).strip()
    out: list[str] = []
    first_para = re.split(r"\n\s*\n", text, maxsplit=1)[0].strip()

    for span in re.findall(r"\*\*([^*]{1,120})\*\*", first_para or text):
        out.append(span)
    for match in re.finditer(r"(?im)^\s*(?:answer|final answer)\s*[:\-]\s*(.{1,120})$", text):
        out.append(match.group(1))

    if first_para:
        first_line = first_para.splitlines()[0].strip()
        first_sentence = re.split(r"(?<=[.!?])\s+", first_line, maxsplit=1)[0].strip()
        out.extend([first_line, first_sentence])

    if len(text) <= 120:
        out.append(text)

    seen: set[str] = set()
    cleaned: list[str] = []
    for candidate in out:
        candidate = candidate.strip(" \t\r\n-*#>\"'")
        norm = _normalize_answer(candidate)
        if norm and norm not in seen:
            cleaned.append(candidate)
            seen.add(norm)
    return cleaned


def _matches_answer(gold: str, candidate: str) -> bool:
    gold_norm = _normalize_answer(gold)
    cand_norm = _normalize_answer(candidate)
    if not gold_norm or not cand_norm:
        return False
    if cand_norm == gold_norm:
        return True
    # For short/numeric answers, require the extracted answer span to be short;
    # otherwise explanations like "388 ... number 22 is separate" become false
    # positives for gold "22".
    if len(gold_norm) <= 3 or re.fullmatch(r"[0-9.:%+-]+", gold_norm):
        if len(cand_norm) > max(12, len(gold_norm) + 6):
            return False
        return re.search(rf"(?<![a-z0-9]){re.escape(gold_norm)}(?![a-z0-9])", cand_norm) is not None
    if len(cand_norm) > max(80, len(gold_norm) * 4):
        return False
    return re.search(rf"(?<![a-z0-9]){re.escape(gold_norm)}(?![a-z0-9])", cand_norm) is not None


def _score_case(case, observed):
    from evalrx.core.case import Label

    candidates = _answer_candidates(str(observed))
    expected = case.expected
    if isinstance(expected, dict):
        if any(_matches_answer(t, c) for t in expected.get("none_of", []) for c in candidates):
            return Label.FAIL
        if not all(any(_matches_answer(t, c) for c in candidates) for t in expected.get("all_of", [])):
            return Label.FAIL
        any_of = expected.get("any_of", [])
        if any_of and not any(_matches_answer(t, c) for t in any_of for c in candidates):
            return Label.FAIL
        return Label.PASS
    if isinstance(expected, str):
        return Label.PASS if any(_matches_answer(expected, c) for c in candidates) else Label.FAIL
    return Label.UNKNOWN


# ---------------------------------------------------------------------------
# Real TextVQA cases
# ---------------------------------------------------------------------------

def _build_textvqa_cases(args):
    """Load paper-style TextVQA answer-bbox cases from a local annotation file."""
    from evalrx.datasets import TextVQASizeDataset

    annotations = Path(args.textvqa_annotations)
    image_dir = Path(args.textvqa_image_dir)
    if not annotations.exists():
        raise SystemExit(
            f"TextVQA annotations not found: {annotations}\n"
            "Download/prep the real TextVQA GT-bbox data under /data first."
        )
    if not image_dir.exists():
        raise SystemExit(
            f"TextVQA image directory not found: {image_dir}\n"
            "Download/prep the real TextVQA GT-bbox images under /data first."
        )

    cases = TextVQASizeDataset.from_jsonl(
        str(annotations),
        image_dir=str(image_dir),
        size_split=args.textvqa_size_split,
        max_samples=args.max_data_cases,
        bbox_format=args.textvqa_bbox_format,
    ).load()
    if not cases:
        raise SystemExit(
            "No TextVQA cases loaded. Check that records contain question, answers, "
            "image dimensions or readable local images, and answer_bbox/bbox fields "
            f"matching split={args.textvqa_size_split!r}."
        )
    return cases


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------

def _build_protocol():
    from evalrx.eval_agent import ExperimentProtocol

    return ExperimentProtocol(
        description=(
            "A vision-language model answers real TextVQA questions that require "
            "reading the textual answer from an image. Following arXiv 2502.17422, "
            "each case carries an answer bounding box and is partitioned by "
            "S = answer_bbox_area / image_area: small S < 0.005, medium "
            "0.005 <= S < 0.05, large S >= 0.05. The main target split is small, "
            "where the answer text occupies less than 0.5% of the image. The "
            "repair goal is a training-free visual intervention, such as cropping "
            "and enhancing the annotated answer region, without changing model "
            "weights."
        ),
        task_domain="fine-grained visual perception / TextVQA small text reading",
        success_criteria=(
            "Response must contain one of the accepted TextVQA answers. Minor "
            "surrounding text is acceptable."
        ),
        failure_patterns=(
            "Small-answer cases: model produces an incorrect or partially correct "
            "OCR answer, says it cannot read the text, or guesses from context "
            "instead of resolving the annotated visual detail."
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
        description="VLDiagnoseLoop - real TextVQA small-detail perception failure"
    )
    parser.add_argument("--model", default="qwen3-vl-4b-instruct")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument(
        "--judge-model", default="Gemini 3.1 Pro (Low)",
        help="agy model for M1-M4 judge.",
    )
    parser.add_argument("--max-cycles", type=int, default=3)
    parser.add_argument("--max-analyzers", type=int, default=3)
    parser.add_argument(
        "--textvqa-annotations",
        default=str(_DEFAULT_TEXTVQA_ANNOTATIONS),
        help="Local real TextVQA GT-bbox JSONL annotations.",
    )
    parser.add_argument(
        "--textvqa-image-dir",
        default=str(_DEFAULT_TEXTVQA_IMAGE_DIR),
        help="Directory containing the real TextVQA images referenced by the annotations.",
    )
    parser.add_argument(
        "--textvqa-size-split",
        choices=("small", "medium", "large", "all"),
        default="small",
        help="TextVQA answer-bbox size split from arXiv 2502.17422.",
    )
    parser.add_argument(
        "--textvqa-bbox-format",
        choices=("auto", "xywh", "xyxy"),
        default="xywh",
        help="Format for list-valued bbox fields; dict bboxes infer from key names.",
    )
    parser.add_argument(
        "--max-data-cases",
        type=int,
        default=None,
        help="Optional cap for loaded real TextVQA cases.",
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
    print(f"  data        {Path(args.textvqa_annotations).resolve()}")
    print(f"  image_dir   {Path(args.textvqa_image_dir).resolve()}")
    print(f"  split       {args.textvqa_size_split}")
    print(f"  output      {run_dir.resolve()}")

    candidates = _build_textvqa_cases(args)
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
            "label": case.label.value,
            "metadata": getattr(case, "metadata", {}),
        })

    _bar(f"CASES  {len(cases)} total / {discovery.n_fail} fail / {discovery.n_pass} pass")
    id_w = max((len(c.id) for c in cases), default=20)
    for case in cases:
        label = "FAIL" if case.label.value == "fail" else "pass"
        obs = textwrap.shorten(str(case.observed), width=80, placeholder="...")
        exp = textwrap.shorten(str(case.expected), width=32, placeholder="...")
        split = (case.metadata or {}).get("size_split", "?")
        print(f"  {label}  {case.id:<{id_w}}  split={split:<6} expected {exp!r:<34} got {obs!r}")

    if not discovery.has_m4_groups:
        print(
            "\n  WARNING: need both FAIL and PASS cases for diagnosis. "
            "Try --max-data-cases with a larger sample, or use --textvqa-size-split all."
        )

    ctx = RunContext(
        run_dir,
        verbose=True,
        config={
            "model": args.model,
            "judge": judge_desc,
            "max_cycles": args.max_cycles,
            "max_analyzers": args.max_analyzers,
            "textvqa_size_split": args.textvqa_size_split,
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
        print(f"    effect {vr.effect_size}  confidence {vr.confidence:.2f}  "
              f"{'protocol-consistent' if vr.is_consistent_with_protocol else 'protocol-inconsistent'}")
        if vr.verdict:
            print(f"    {textwrap.shorten(str(vr.verdict), width=90, placeholder='...')}")

    _bar("EXPERIMENT  (M5 mechanism verification)")
    if report.verified_hypotheses:
        fix = loop.run_m5(report, cases)
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
