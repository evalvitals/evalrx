"""deco_hallu — Mode-1 container input: the POPE HALLUCINATION subset -> VLDiagnoseLoop.

The hard, complementary slice to deco_miss. The failure is a *false Yes* (the
model says an absent object is present). A confident hallucination has no latent
correct signal to amplify, so this slice tests whether the loop can find a
mitigation that survives the no-free-lunch guard — the batch mixes the
hallucinations with present-object DETECTIONS, so a degenerate "always answer
No" fix is penalised for breaking recall and cannot win.

    1. load data/cases/{model}.json   (hallucination + controls, build_cases.py)
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

def load_manifest(model_key: str):
    """Load the pre-balanced hallucination batch (build_cases.py already capped
    the controls, so nothing is subsampled here — the present-Yes detections are
    the no-free-lunch guard and must all stay in)."""
    from PIL import Image

    from evalvitals.core.case import CaseBatch, FailureCase, Inputs, Label

    path = DATA / "cases" / f"{model_key}.json"
    if not path.exists():
        raise SystemExit(f"{path} missing — run `python build_cases.py` first")
    raw = json.loads(path.read_text())

    cases = []
    pil_cache: dict[str, object] = {}
    for r in raw["cases"]:
        if r["file_name"] not in pil_cache:
            pil_cache[r["file_name"]] = Image.open(IMAGES / r["file_name"]).convert("RGB")
        cases.append(FailureCase(
            inputs=Inputs(prompt=raw["prompt_template"].format(obj=r["object"]),
                          image=pil_cache[r["file_name"]]),
            expected=r["pope_label"],
            observed=r["observed"],
            label=Label.FAIL if r["label"] == "fail" else Label.PASS,
            tags={"hallucination", "object-presence"},
            metadata={k: r[k] for k in (
                "image_id", "object", "probe_type", "pope_label",
                "gt_token_ids", "out_token_ids", "split")},
        ))
    return CaseBatch(cases), raw


def drift_check(model, cases, n: int = 10) -> None:
    from evalvitals.analyzers.hallucination.pope import parse_yes_no
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
    from evalvitals.eval_agent import ClaudeModel

    judge = ClaudeModel(model=model_name, effort=effort)
    if not judge.generate("Reply with exactly the word OK").strip():
        raise SystemExit(f"judge probe: claude --model {model_name} returned empty "
                         f"(rate-limited?) — try --judge-model sonnet or haiku")
    print(f"judge: claude model={model_name} effort={effort or 'default'}")
    return judge


# Coder/explorer backend is selectable across the three local CLI agents; the
# chat judge stays claude (no codex/agy chat-model wrapper exists).
_PROVIDER = {"claude": "claude_code", "codex": "codex", "agy": "antigravity"}


def build_codegen(backend: str = "claude", skills=(), allow_skills: bool = False):
    """CliAgentConfig for the coder/explorer backend. ``backend`` is one of
    ``claude`` | ``codex`` | ``agy`` (default claude). model/effort are claude-only
    knobs; codex/agy use their own defaults. ``skills`` are Agent-Skill dirs (e.g.
    a nature-figure skill) vendored to style agent-authored figures."""
    from evalvitals.eval_agent import CliAgentConfig

    provider = _PROVIDER.get(backend, backend)
    is_claude = provider == "claude_code"
    effort = str(CFG.get("codegen_effort", "") or "")
    cfg = CliAgentConfig(
        provider=provider,
        model=str(CFG.get("codegen_model", "claude-opus-4-8")) if is_claude else "",
        max_budget_usd=float(CFG.get("codegen_budget_usd", 2.0)),
        timeout_sec=int(CFG.get("codegen_timeout_sec", 240)),
        extra_args=(("--effort", effort) if (effort and is_claude) else ()),
        skills=tuple(skills or ()),
        allow_skills=allow_skills,
    )
    print(f"coder backend: {provider}" + (f" model={cfg.model} effort={effort or 'default'}" if is_claude else ""))
    return cfg


def build_protocol():
    from evalvitals.eval_agent.stages.protocol import ExperimentProtocol

    # OBSERVATION ONLY: state the wrong-answer pattern and the input conditions.
    # Do NOT name a suspected mechanism (layers, suppression, priors, DeCo) —
    # that is the loop's job to discover; supplying it would leak the answer.
    return ExperimentProtocol(
        description=(
            "For yes/no questions of the form 'Is there a {object} in the image?', "
            "the VLM sometimes answers 'Yes' for an object that is NOT in the "
            "image (a false detection). Failure cases are absent-object questions "
            "the model answers 'Yes'. The batch also contains, as controls, "
            "absent-object questions it correctly answers 'No' and present-object "
            "questions it correctly answers 'Yes', drawn from the same images and "
            "question template; the answer is scored against each question's own "
            "ground truth."
        ),
        task_domain="object-presence recognition",
        success_criteria="parse_yes_no(answer) matches the POPE gold label",
        failure_patterns=(
            "the answer is 'Yes' although the queried object is not in the image; "
            "present-object questions are mostly answered 'Yes' correctly — a "
            "change that suppresses these correct 'Yes' answers is not an "
            "improvement, only a different error"
        ),
        target_modalities=frozenset({"text", "image"}),
    )


# ---------------------------------------------------------------------------
# Frozen-M1 handoff helpers (shared by run_m2-5.py / run_analysis.py /
# run_confirm_fix.py — the staged scripts that replay outputs/m1_state.pkl).
# ---------------------------------------------------------------------------

class ReplayProbeAgent:
    """Stands in for ProbeAgent: returns the frozen M1 result instead of
    re-running analyzers. Exposes exactly the attributes the loop reads off the
    probe agent after M1 (last_schema / last_selection_* / _failed_analyzers /
    _generated_probes / run_logger), so the loop's M1 step is unmodified."""

    def __init__(self, state: dict):
        self._results = state["probe_results"]
        self.last_schema = state.get("schema")
        self.last_selection_prompt = state.get("selection_prompt", "")
        self.last_selection_raw = state.get("selection_raw", "")
        self._failed_analyzers = dict(state.get("failed_analyzers", {}) or {})
        self._generated_probes = []        # tier-(b) codegen already happened in M1
        self.run_logger = None             # set by the loop's _attach_run_logger

    def probe(self, model, data, **kwargs):  # noqa: ARG002 - signature parity only
        return self._results


class FrozenModel:
    """Lightweight stand-in for the VLM during the GPU-free analysis phase.

    M1 is replayed from the pickle and M2/M3 only need the model's repr string
    (model_name in the stats narrative, the run_start provenance), so the
    analysis phase never has to load weights or touch the GPU. The repair phase
    (run_confirm_fix.py) loads the real model because run_fix calls it."""

    def __init__(self, model_key: str):
        self._key = model_key

    def __repr__(self) -> str:
        return self._key


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
        print("smoke ok" if exists else "smoke: no manifest yet (run build_cases.py)")
        return

    from evalvitals import compose
    from evalvitals.core.capability import Capability
    from evalvitals.eval_agent import (
        CliAgentConfig,
        ExperimentWriterConfig,
        FixAgent,
        RunLogger,
        SurgeryAgent,
        VLDiagnoseLoop,
    )
    from evalvitals.eval_agent.stages.diagnosis import DiagnosisAgent
    from evalvitals.eval_agent.stages.probe_agent import ProbeAgent
    from evalvitals.analysis.stats_agent import StatsAnalysisAgent
    from evalvitals.models.backends.base import RuntimeConfig

    judge = build_judge(args.judge_model, args.judge_effort)

    model = compose(args.model, "hf_local",
                    runtime=RuntimeConfig(device=args.device, dtype=args.dtype),
                    want={Capability.GENERATE, Capability.HIDDEN_STATES,
                          Capability.ATTENTION})
    cases, raw = load_manifest(args.model)
    print(f"cases={len(list(cases))} yields={raw['yields']} "
          f"hallu={raw['n_hallucination']} reject={raw['n_correct_reject']} "
          f"present={raw['n_present_detect']}")
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
    run_logger = RunLogger(run_dir=OUT / "logs", verbose=True)
    loop = VLDiagnoseLoop(
        model=model,
        probe_agent=ProbeAgent(judge=judge, max_analyzers=args.max_analyzers,
                               allow_codegen=True, codegen_config=codegen),
        stats_agent=StatsAnalysisAgent(judge=judge, allow_codegen=True,
                                       codegen_config=codegen),
        diagnosis_agent=DiagnosisAgent(judge=judge),
        surgery_agent=SurgeryAgent(
            judge=judge, writer_config=ExperimentWriterConfig(cli_agent=codegen)),
        fix_agent=FixAgent(judge=judge,
                           max_tier=str(CFG.get("fix_max_tier", "L3b")),
                           cli_config=codegen, run_logger=run_logger,
                           max_validation_cases=int(CFG.get("fix_validation_cases", 60)),
                           exec_timeout_sec=int(CFG.get("fix_exec_timeout_sec", 900))),
        max_cycles=args.max_cycles,
        protocol=build_protocol(),
        run_logger=run_logger,
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


if __name__ == "__main__":
    main()
