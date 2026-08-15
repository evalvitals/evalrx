"""deco_chair — Mode-1 container input: frozen captions + protocol -> VLDiagnoseLoop.

Captioning sibling of examples/m1_m4/deco_pope / deco_miss / deco_hallu.  Same DeCo
scenario family (arXiv 2410.11779); the failure here is **object hallucination
in an open-ended description** (the model names an object that is not in the
image).  This run.py only provides INPUTS — frozen cases + an observation-only
protocol — and a CHAIR scorer; detection, diagnosis and repair are the loop's
own job (its agents select analyzers, write probes, propose + validate fixes).

    1. load data/cases/{model}.json   (frozen captions + CHAIR-matched mentions)
    2. drift check                    (re-caption a sample, compare hallucination)
    3. ExperimentProtocol             (OBSERVATION ONLY — no mechanism named)
    4. VLDiagnoseLoop M1->M5          (loop selects its own analyzers; tier-(a)
                                       should pick `chair`)
    5. run_m4 + run_fix               (loop proposes + validates its own fix;
                                       feedback-driven, up to fix_repair_rounds)

The fix module is given a CHAIR `score_fn` (object hallucination with a RECALL
FLOOR): a candidate output counts as a success only if it names NO absent object
AND still names at least one present object — so a degenerate "say nothing /
stay vague" fix loses recall on the clean controls and cannot win (the
no-free-lunch guard, the captioning analogue of deco_hallu's present-detection
controls).

Reference analyses live OUTSIDE this repo so the loop's coding agents can't read
them — they grade the loop's conclusions, they do not feed them.

Usage:
    python run.py --model qwen3-vl-2b-instruct --device cuda
    python run.py --smoke-test
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import yaml

DATA = Path(__file__).parent / "data"
OUT = Path(__file__).parent / "outputs"
CFG = yaml.safe_load((Path(__file__).parent / "config.yaml").read_text())


# ---------------------------------------------------------------------------
# CHAIR scoring (synonym map -> COCO category) with a recall floor
# ---------------------------------------------------------------------------

def _load_synonyms() -> "dict[str, list[str]]":
    """COCO category -> surface synonyms (Rohrbach et al.), longest-first so a
    multi-word surface ('dining table') is matched before a sub-word."""
    raw = json.loads((DATA / "chair_synonyms.json").read_text())["synonyms"]
    return {cat: sorted({s.lower() for s in surfaces}, key=len, reverse=True)
            for cat, surfaces in raw.items()}


def _normalize(text: str) -> str:
    """Lowercase, punctuation -> spaces, single-spaced and padded for word match."""
    return " " + re.sub(r"[^a-z0-9 ]+", " ", text.lower()) + " "


def _mentioned_categories(caption: str, syn: "dict[str, list[str]]") -> "set[str]":
    norm = re.sub(r"\s+", " ", _normalize(caption))
    out: "set[str]" = set()
    for cat, surfaces in syn.items():
        for s in surfaces:
            if f" {s} " in norm or f" {s}s " in norm:
                out.add(cat)
                break
    return out


def make_chair_score_fn(syn: "dict[str, list[str]]"):
    """Build ``(case, output) -> bool | None`` for the fix module.

    True  = the description names only present objects AND still names >= 1
            present object (no hallucination, recall preserved).
    False = it names an absent object, OR it names no present object at all
            (vague/empty/evasive — the degenerate "fix" the guard must reject).
    None  = no ground-truth object list (unscorable).
    """

    def score_fn(case, output: str):
        gt = {o.lower() for o in (getattr(case, "metadata", {}) or {}).get("gt_objects", [])}
        if not gt:
            return None
        mentioned = _mentioned_categories(str(output), syn)
        hallucinated = mentioned - gt
        grounded = mentioned & gt
        return (not hallucinated) and bool(grounded)

    return score_fn


# ---------------------------------------------------------------------------
# 1. Frozen manifest -> CaseBatch (image-level; mention records stay in raw)
# ---------------------------------------------------------------------------

def load_manifest(model_key: str):
    from PIL import Image

    from evalvitals.core.case import CaseBatch, FailureCase, Inputs, Label

    path = DATA / "cases" / f"{model_key}.json"
    if not path.exists():
        raise SystemExit(f"{path} missing — run `python mine_cases.py --model {model_key}`")
    raw = json.loads(path.read_text())
    gt_objects = json.loads((DATA / "gt_objects.json").read_text())
    cases = []
    for img in raw["images"]:
        pil = Image.open(DATA / "images" / img["file_name"]).convert("RGB")
        cases.append(FailureCase(
            inputs=Inputs(prompt=raw["prompt"], image=pil),
            expected="description naming only objects present in the image",
            observed=img["caption"],
            label=Label.FAIL if img["hallucinated"] else Label.PASS,
            tags={"hallucination", "captioning", "object-presence"},
            metadata={
                "image_id": img["image_id"],
                "gt_objects": gt_objects[str(img["image_id"])],
                "mentions": img["mentions"],
                "split": img.get("split", "explore"),
            },
        ))
    return CaseBatch(cases), raw


def force_greedy(model) -> None:
    """Match the greedy decoding the manifest was mined with.

    Qwen3-VL defaults to sampling; on long captions that drifts the output (and
    the hallucination verdict) run-to-run and adds noise to the fix module's
    paired baseline.  Pin deterministic greedy so analyzers, baseline and
    candidate all decode the same way."""
    try:
        hf, _ = model._loaded
        gc = hf.generation_config
        gc.do_sample = False
        gc.temperature = 1.0
        gc.top_p = 1.0
        gc.top_k = 0
        print("decoding: greedy (do_sample=False)")
    except Exception as exc:
        print(f"[WARN] could not force greedy decoding: {exc}")


def drift_check(model, cases, syn, n: int = 3) -> None:
    """Re-caption a few images; warn if the hallucinated-vs-clean verdict flips
    (long generations drift — compare the boolean, not the text)."""
    from evalvitals.core.case import Label

    stale = 0
    for case in list(cases)[:n]:
        gt = {o.lower() for o in case.metadata.get("gt_objects", [])}
        hallu_now = bool(_mentioned_categories(str(model.generate(case.inputs)), syn) - gt)
        if hallu_now != (case.label == Label.FAIL):
            stale += 1
    if stale:
        print(f"[WARN] {stale}/{n} frozen hallucination labels no longer reproduce — "
              f"re-run mine_cases.py (check transformers version)")


# ---------------------------------------------------------------------------
# 2. Judge + Protocol (OBSERVATION ONLY — naming a mechanism leaks the answer)
# ---------------------------------------------------------------------------

def build_judge(model_name: str, effort: str):
    from evalvitals.eval_agent import ClaudeModel

    judge = ClaudeModel(model=model_name, effort=effort)
    if not judge.generate("Reply with exactly the word OK").strip():
        raise SystemExit(f"judge probe: claude --model {model_name} returned empty "
                         f"(rate-limited?) — try --judge-model sonnet or haiku")
    print(f"judge: claude model={model_name} effort={effort or 'default'}")
    return judge


def build_protocol():
    from evalvitals.eval_agent.stages.protocol import ExperimentProtocol

    # OBSERVATION ONLY: describe the wrong-output pattern and the recall guard.
    # Do NOT name a suspected mechanism (layers, suppression, language/co-occurrence
    # prior, DeCo) — discovering any such cause is the loop's job; supplying it
    # would leak the answer. (Same answer-no-leak rule as deco_pope/miss/hallu.)
    return ExperimentProtocol(
        description=(
            "When asked to describe an image in detail, the VLM's description "
            "sometimes names one or more objects that are NOT present in the "
            "image, while other objects it names in the same description are "
            "correct. Failure cases are descriptions that name at least one "
            "absent object; success cases are descriptions, drawn from the same "
            "image pool and the same prompt, that name only objects actually "
            "present. Each description is scored against its image's ground-truth "
            "object list."
        ),
        task_domain="object hallucination in open-ended image description",
        success_criteria=(
            "the description names only objects present in the image (no "
            "hallucinated object) while still describing the objects that are "
            "present"
        ),
        failure_patterns=(
            "the description asserts an object that is not in the image; the "
            "correctly-named present objects in the same description are not the "
            "problem — a change that drops correct object mentions, or makes the "
            "description vague or empty to avoid naming the absent object, is not "
            "an improvement, only a different error (loss of recall)"
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
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16")
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)

    if args.smoke_test:
        exists = (DATA / "cases" / f"{args.model}.json").exists()
        print("smoke ok" if exists else "smoke: no manifest yet (run mine_cases.py)")
        return

    from evalvitals import compose
    from evalvitals.core.capability import Capability
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
    from evalvitals.analysis.stats_agent import StatsAnalysisAgent
    from evalvitals.models.backends.base import RuntimeConfig

    judge = build_judge(args.judge_model, args.judge_effort)  # probe BEFORE weights load

    model = compose(args.model, "hf_local",
                    runtime=RuntimeConfig(device=args.device, dtype=args.dtype),
                    want={Capability.GENERATE, Capability.HIDDEN_STATES,
                          Capability.ATTENTION})
    force_greedy(model)
    syn = _load_synonyms()
    cases, raw = load_manifest(args.model)
    n_fail = sum(1 for c in cases if str(getattr(c.label, "value", "")) == "fail")
    print(f"cases={len(list(cases))} hallucinated(FAIL)={n_fail} "
          f"clean(PASS)={len(list(cases)) - n_fail}")
    drift_check(model, cases, syn)

    ctx = RunContext(
        OUT, verbose=True,
        config={"model": args.model, "judge_model": args.judge_model, "max_cycles": args.max_cycles},
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
    chair_score = make_chair_score_fn(syn)
    # The `chair` analyzer needs an object vocabulary, so it can't auto-instantiate
    # with default args — give M1 a ready instance (COCO category names) via the
    # override map, so tier-(a) selection of `chair` actually runs.
    from evalvitals.analyzers.hallucination.chair import CHAIRAnalyzer
    chair_analyzer = CHAIRAnalyzer(object_vocab=list(syn))
    loop = VLDiagnoseLoop(
        model=model,
        probe_agent=ProbeAgent(judge=judge, max_analyzers=args.max_analyzers,
                               allow_codegen=True, codegen_config=codegen,
                               analyzer_overrides={"chair": chair_analyzer}),
        stats_agent=StatsAnalysisAgent(judge=judge, allow_codegen=True,
                                       codegen_config=codegen, figure_dir=str(ctx.figures_dir)),
        diagnosis_agent=DiagnosisAgent(judge=judge),
        surgery_agent=SurgeryAgent(
            judge=judge, writer_config=ExperimentWriterConfig(cli_agent=codegen),
            run_context=ctx),
        # CHAIR scorer (recall-floored) replaces the yes/no rubric scorer so the
        # fix module can validate open-ended captions; feedback-driven repair
        # runs up to fix_repair_rounds rounds within the allowed tier.
        fix_agent=FixAgent(judge=judge, score_fn=chair_score,
                           max_tier=str(CFG.get("fix_max_tier", "L2")),
                           cli_config=codegen, run_logger=ctx.logger,
                           max_validation_cases=int(CFG.get("fix_validation_cases", 24)),
                           exec_timeout_sec=int(CFG.get("fix_exec_timeout_sec", 1200)),
                           max_repair_rounds=int(CFG.get("fix_repair_rounds", 3)),
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
