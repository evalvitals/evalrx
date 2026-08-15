"""VLDiagnoseLoop on Qwen3-VL: temporal video understanding failure diagnosis.

Pipeline (same as qwen_loop_agy):

    ExperimentProtocol  ← NL description of the temporal reasoning task
         │
    M1  ProbeAgent           LLM-guided analyzer selection + execute
    M2  StatsAnalysisAgent   stats tools + e-BH FDR-correct + evidence chain
    M3  DiagnosisAgent       hypothesis generation
    M5  HypothesisTester     statistical test + protocol consistency check
         │
    loop exits when M5 finds a verified, protocol-consistent hypothesis,
    or after --max-cycles cycles
         │
    M4  SurgeryAgent         post-loop fix proposal (called separately)

Key difference vs qwen_loop_agy:
  Inputs carry a *video* field (list of 3 PIL frames) instead of a single
  image.  The evalvitals backend maps each frame to a separate image-token
  block in the VLM prompt, so the model sees a temporal sequence.
  Cases probe temporal understanding:
    - easy  (→ PASS): static properties visible in any single frame
                      (shape colour, frame count, shape presence)
    - hard  (→ FAIL): precise cross-frame measurements the model cannot
                      know (exact pixel coords, exact movement in pixels,
                      exact fraction of width travelled)

Outputs written to --run-dir (default: ./outputs/), see manifest.json + the
auto-generated README.txt for the full index:
    run_log.jsonl          ← one JSON line per M1/M2/M3/M5 event
    artifacts/              ← per-cycle analyzer artifacts

Usage (via Docker — preferred):
    docker compose up

Usage (direct):
    python run.py
    python run.py --smoke-test          # fast local wiring test, no GPU/agy
    python run.py --model qwen2.5-vl-7b-instruct --device cuda
    python run.py --max-cycles 3 --max-analyzers 3
"""

from __future__ import annotations

import re
import textwrap
from pathlib import Path

_OUTPUTS_DIR = Path(__file__).parent / "outputs"
_N_FRAMES = 3          # synthetic clip length


# ---------------------------------------------------------------------------
# Synthetic video
# ---------------------------------------------------------------------------

def _synthetic_video() -> list:
    """Return a list of ``_N_FRAMES`` PIL images showing a red rectangle
    moving from left to right across a neutral-grey background.

    Frame layout (224 × 224 px):
      - Background: medium grey  (180, 180, 180)
      - Red box (60 × 100 px) at y = 62..162; x advances 62 px per frame:
          frame 0: x = 20..80    (left)
          frame 1: x = 82..142   (centre)
          frame 2: x = 144..204  (right)
      - White frame-number label in the bottom-left corner.
    """
    from PIL import Image, ImageDraw

    frames = []
    xs = [(20, 80), (82, 142), (144, 204)]
    for i, (x1, x2) in enumerate(xs):
        img = Image.new("RGB", (224, 224), color=(180, 180, 180))
        draw = ImageDraw.Draw(img)
        draw.rectangle([x1, 62, x2, 162], fill=(210, 50, 50))  # red box
        try:
            draw.text((4, 204), f"frame {i + 1}", fill=(255, 255, 255))
        except Exception:
            pass
        frames.append(img)
    return frames


# ---------------------------------------------------------------------------
# Word-boundary scorer (identical to qwen_loop_agy)
# ---------------------------------------------------------------------------

def _contains(term: str, text: str) -> bool:
    term = term.lower().strip()
    if not term:
        return False
    if re.fullmatch(r"[a-z0-9]+", term):
        return re.search(rf"\b{re.escape(term)}\b", text) is not None
    return term in text


def _score_case(case, observed):
    from evalvitals.core.case import Label

    text = re.sub(r"\s+", " ", str(observed).lower())
    expected = case.expected
    if isinstance(expected, dict):
        if any(_contains(t, text) for t in expected.get("none_of", [])):
            return Label.FAIL
        if not all(_contains(t, text) for t in expected.get("all_of", [])):
            return Label.FAIL
        any_of = expected.get("any_of", [])
        if any_of and not any(_contains(t, text) for t in any_of):
            return Label.FAIL
        return Label.PASS
    if isinstance(expected, str):
        return Label.PASS if _contains(expected, text) else Label.FAIL
    return Label.UNKNOWN


# ---------------------------------------------------------------------------
# Cases
# ---------------------------------------------------------------------------

def _build_candidate_cases(frames: list):
    """Balanced cases: easy (salient static properties → PASS) +
    hard (precise cross-frame measurements → FAIL).

    All prompts reference the frame sequence explicitly so the model
    knows it is looking at a temporal clip, not unrelated images.
    """
    from evalvitals.core.case import CaseBatch, FailureCase, Inputs

    prefix = (
        f"You are shown {_N_FRAMES} consecutive frames from a short video clip "
        "(displayed left-to-right as frame 1, frame 2, frame 3). "
    )
    return CaseBatch([
        # ── Easy: salient static properties the model reliably gets right ──
        FailureCase(
            id="q_color",
            inputs=Inputs(
                prompt=prefix + "What color is the moving shape? Answer with one word.",
                video=frames,
            ),
            expected={"any_of": ["red", "crimson", "scarlet"]},
        ),
        FailureCase(
            id="q_n_frames",
            inputs=Inputs(
                prompt=prefix + "How many frames are shown in this video clip? "
                                "Answer with one number.",
                video=frames,
            ),
            expected={"any_of": ["3", "three"]},
        ),
        FailureCase(
            id="q_present",
            inputs=Inputs(
                prompt=prefix + "Is the coloured shape visible in every frame? "
                                "Answer yes or no.",
                video=frames,
            ),
            expected={"all_of": ["yes"], "none_of": ["no"]},
        ),
        # ── Hard: precise pixel-level details the model cannot know ──
        FailureCase(
            id="q_coords_f2",
            inputs=Inputs(
                prompt=prefix + "What are the exact pixel coordinates (x1, y1, x2, y2) "
                                "of the shape in frame 2? Answer with four numbers only.",
                video=frames,
            ),
            # exact source values: x1=82, y1=62, x2=142, y2=162
            expected={"all_of": ["82", "62", "142", "162"]},
        ),
        FailureCase(
            id="q_movement_px",
            inputs=Inputs(
                prompt=prefix + "How many pixels did the left edge of the shape move "
                                "from frame 1 to frame 3? Answer with one number only.",
                video=frames,
            ),
            # left edge: 20 → 144, movement = 124 px
            expected={"any_of": ["124"]},
        ),
        FailureCase(
            id="q_travel_pct",
            inputs=Inputs(
                prompt=prefix + "What percentage of the frame width did the shape's "
                                "centre travel from frame 1 to frame 3? "
                                "Answer with one integer.",
                video=frames,
            ),
            # centre moves: 50 → 174, Δ=124, frame width=224 → 55 %
            expected={"any_of": ["55"]},
        ),
    ])


# ---------------------------------------------------------------------------
# Smoke-test stubs
# ---------------------------------------------------------------------------

class _SmokeVLM:
    """Deterministic VLM stand-in — exercises the wiring without GPU/weights."""

    def __init__(self) -> None:
        from evalvitals.core.capability import Capability

        self.capabilities = frozenset({Capability.GENERATE, Capability.ATTENTION})
        self.modalities = frozenset({"text", "image", "video"})

    def __repr__(self) -> str:
        return "SmokeVLM()"

    def generate(self, inputs, **kwargs) -> str:
        prompt = str(getattr(inputs, "prompt", inputs)).lower()
        if "what color" in prompt:
            return "red"
        if "how many frames" in prompt:
            return "3"
        if "visible in every frame" in prompt:
            return "Yes, the shape is visible in every frame."
        if "exact pixel coordinates" in prompt:
            return "10 20 50 80"                # wrong
        if "how many pixels did" in prompt:
            return "42"                          # wrong
        if "percentage of the frame width" in prompt:
            return "20"                          # wrong
        return "Unknown."

    def forward(self, inputs, capture, spec=None):
        raise NotImplementedError("SmokeVLM only supports generate().")


class _SmokeProbe:
    last_schema = None

    def probe(self, model, data, **kwargs):
        from evalvitals.core.case import Label
        from evalvitals.core.result import Result
        from evalvitals.eval_agent import ProbingSchema

        fail_ids = [case.id for case in data if case.label == Label.FAIL]
        self.last_schema = ProbingSchema(
            selected_analyzers=["attention"],
            rationale="Smoke probe: attention signals on discovered failures.",
            protocol=kwargs.get("protocol"),
        )
        findings = {
            "mean_entropy": 0.3,
            "per_case": [
                {"sample_id": cid, "attention_signal": True}
                for cid in fail_ids
            ],
        }
        return {
            "attention": Result(
                analyzer="attention",
                model=repr(model),
                cases=data,
                findings=findings,
            )
        }


class _SmokeDiagnosisAgent:
    def diagnose(self, analysis, prior_cycles=None):
        from evalvitals.eval_agent import DiagnosisResult, Hypothesis

        h = Hypothesis(
            statement=(
                "The model attends only to visual tokens in the first frame "
                "and ignores temporal context from later frames."
            ),
            target_model=analysis.model_name,
            predicted_failure_mode="temporal_attention",
        )
        return DiagnosisResult(
            model_name=analysis.model_name,
            hypotheses=[h],
            findings_summary={name: r.findings for name, r in analysis.raw_results.items()},
            raw_judge_output=(
                "HYPOTHESIS: The model ignores temporal context.\n"
                "FAILURE_MODE: temporal_attention"
            ),
        )


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------

def _build_protocol():
    from evalvitals.eval_agent import ExperimentProtocol

    return ExperimentProtocol(
        description=(
            "We show a vision-language model three consecutive frames from a "
            "short synthetic video clip.  The clip shows a red rectangle moving "
            "steadily from left to right across a grey background.  The model "
            "must answer questions about both static properties (colour, presence) "
            "and precise temporal measurements (exact pixel coordinates per frame, "
            "exact displacement in pixels, percentage of frame width travelled).  "
            "The model reliably identifies colour and frame count but consistently "
            "fails on measurements that require integrating information across "
            "frames.  We want to understand whether the failures stem from "
            "ignoring later frames, misaligning image tokens, or an inability to "
            "reason about spatial change across a sequence."
        ),
        task_domain="temporal video question answering",
        success_criteria=(
            "Colour identification, frame count, and shape-presence answers must "
            "be correct.  Cross-frame measurements (pixel coords, displacement, "
            "travel percentage) must match the ground-truth values."
        ),
        target_modalities=frozenset({"text", "image", "video"}),
    )


# ---------------------------------------------------------------------------
# Smoke-test runner
# ---------------------------------------------------------------------------

def _run_smoke_test(args) -> None:
    from evalvitals.eval_agent import (
        CaseDiscoveryAgent,
        HypothesisTester,
        RunContext,
        StatsAnalysisAgent,
        StatsToolAgent,
        SurgeryAgent,
        VLDiagnoseLoop,
    )

    model = _SmokeVLM()
    frames = _synthetic_video()
    protocol = _build_protocol()

    discovery = CaseDiscoveryAgent(
        scorer=_score_case,
        include_unknown=False,
    ).discover(model, _build_candidate_cases(frames), protocol=protocol)
    cases = discovery.cases

    print("\nSmoke test data:")
    print(
        f"  discovered {len(cases)} labeled cases "
        f"(PASS={discovery.n_pass}, FAIL={discovery.n_fail}, UNKNOWN={discovery.n_unknown})"
    )
    if not discovery.has_m5_groups:
        raise SystemExit("Smoke test requires both PASS and FAIL cases.")

    ctx = RunContext(args.run_dir, verbose=True, config={"smoke_test": True})

    loop = VLDiagnoseLoop(
        model=model,
        protocol=protocol,
        probe_agent=_SmokeProbe(),
        stats_agent=StatsAnalysisAgent(stats_tool_agent=StatsToolAgent(max_tools=3)),
        diagnosis_agent=_SmokeDiagnosisAgent(),
        hypothesis_tester=HypothesisTester(min_effect=0.05),
        surgery_agent=SurgeryAgent(),
        max_cycles=1,
        run_logger=ctx.logger,
    )
    report = loop.run(cases)
    ctx.write_diagnose_report(report, cases)
    ctx.finalize()

    print("\nSmoke test result:")
    print(f"  stopped_by={report.stopped_by}  cycles={report.cycles}")
    print(f"  verified={len(report.verified_hypotheses)}")
    if report.stopped_by != "criteria_met" or not report.verified_hypotheses:
        raise SystemExit("Smoke test failed: no verified hypothesis.")

    fix = loop.run_m4(report, cases)
    if fix is None or fix.status.value != "supported":
        raise SystemExit("Smoke test failed: M4 did not support the verified hypothesis.")

    print("  m4_status=supported")
    print("Smoke test passed.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="VLDiagnoseLoop — temporal video understanding on Qwen VL"
    )
    parser.add_argument("--model", default="qwen3-vl-4b-instruct")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument(
        "--judge-model", default="Gemini 3.1 Pro (Low)",
        help="agy model for the M1–M5 judge.",
    )
    parser.add_argument("--max-cycles", type=int, default=2)
    parser.add_argument("--max-analyzers", type=int, default=2)
    parser.add_argument(
        "--smoke-test", action="store_true",
        help="Fast local wiring test — no GPU, no model weights, no agy.",
    )
    parser.add_argument("--run-dir", default=str(_OUTPUTS_DIR))
    args = parser.parse_args()

    if args.smoke_test:
        _run_smoke_test(args)
        return

    import evalvitals
    from evalvitals.eval_agent import (
        AgyModel,
        CaseDiscoveryAgent,
        CliAgentConfig,
        DiagnosisAgent,
        ExperimentWriterConfig,
        HypothesisTester,
        ProbeAgent,
        RunContext,
        StatsAnalysisAgent,
        SurgeryAgent,
        VLDiagnoseLoop,
    )

    # ── Load model ────────────────────────────────────────────────────────
    print(f"\nLoading {args.model!r} on {args.device} ({args.dtype}) …")
    model = evalvitals.load(
        args.model,
        backend="hf_local",
        device=args.device,
        dtype=args.dtype,
        want=["attention"],
    )
    print(f"  capabilities : {sorted(str(c.name) for c in model.capabilities)}")
    print(f"  modalities   : {sorted(model.modalities)}")

    # ── Judge ─────────────────────────────────────────────────────────────
    try:
        judge = AgyModel(model=args.judge_model)
        print(f"\n  judge : antigravity CLI ({judge._binary})  "
              f"model={args.judge_model or 'session default'}  [M1–M5, no API key]")
    except RuntimeError as _agy_err:
        import warnings as _w
        _w.warn(
            f"agy not available ({_agy_err}). Falling back to the loaded model as judge.",
            stacklevel=2,
        )
        judge = model
        print(f"\n  judge : {args.model} (agy unavailable — using evaluated model as fallback)")

    protocol = _build_protocol()
    run_dir = Path(args.run_dir)
    ctx = RunContext(
        run_dir, verbose=True,
        config={
            "model": args.model,
            "judge_model": args.judge_model,
            "max_cycles": args.max_cycles,
            "max_analyzers": args.max_analyzers,
        },
    )

    # ── Video frames + cases ──────────────────────────────────────────────
    print(f"\nGenerating {_N_FRAMES}-frame synthetic video clip …")
    frames = _synthetic_video()
    print(f"  frame size: {frames[0].size} px  ({_N_FRAMES} frames)")

    candidate_cases = _build_candidate_cases(frames)
    discovery = CaseDiscoveryAgent(
        scorer=_score_case,
        include_unknown=False,
    ).discover(model, candidate_cases)
    cases = discovery.cases
    print(
        f"  discovered {len(cases)} labeled cases "
        f"(PASS={discovery.n_pass}, FAIL={discovery.n_fail}, UNKNOWN={discovery.n_unknown})"
    )
    discovery_rows = []
    for case in cases:
        observed = str(case.observed)
        discovery_rows.append({
            "id": case.id,
            "prompt": case.inputs.prompt,
            "expected": case.expected,
            "observed": observed,
            "label": case.label.value,
        })
        print(
            f"    [{case.label.value.upper()}] {case.id}: "
            f"{textwrap.shorten(observed, width=110, placeholder='...')}"
        )
    if not discovery.has_m5_groups:
        print(
            "  WARNING: M5 needs both PASS and FAIL cases; "
            "this run may stop without verified hypotheses."
        )

    print("\nExperimentProtocol:")
    print(f"  task_domain : {protocol.task_domain}")
    print(f"  description : {protocol.description[:80]}...")

    # ── Agents ────────────────────────────────────────────────────────────
    probe_agent = ProbeAgent(judge=judge, max_analyzers=args.max_analyzers)

    stats_agent = StatsAnalysisAgent(
        judge=judge,
        figure_dir=str(ctx.figures_dir),
    )

    diagnosis_agent = DiagnosisAgent(judge=judge)

    hypothesis_tester = HypothesisTester(judge=judge, min_effect=0.05)

    writer_cfg = ExperimentWriterConfig(
        cli_agent=CliAgentConfig(
            provider="antigravity", timeout_sec=120, model=args.judge_model
        ),
        exec_fix_timeout_sec=60,
    )
    surgery_agent = SurgeryAgent(judge=judge, writer_config=writer_cfg, run_context=ctx)

    # ── Run ───────────────────────────────────────────────────────────────
    print(f"\nOutput directory: {run_dir.resolve()}")
    print("  run_log.jsonl   ← one JSON line per M1/M2/M3/M5 event")
    print("  artifacts/      ← per-cycle analyzer artifacts")

    loop = VLDiagnoseLoop(
        model=model,
        protocol=protocol,
        probe_agent=probe_agent,
        stats_agent=stats_agent,
        diagnosis_agent=diagnosis_agent,
        hypothesis_tester=hypothesis_tester,
        surgery_agent=surgery_agent,
        max_cycles=args.max_cycles,
        run_logger=ctx.logger,
    )

    print(f"\n{'='*64}")
    print(f"VLDiagnoseLoop  model={args.model}  max_cycles={args.max_cycles}")
    print(f"{'='*64}")

    report = loop.run(cases)
    ctx.write_diagnose_report(report, cases, discovery=discovery_rows)

    print(f"\n{'='*64}")
    print(f"LOOP RESULT  stopped_by={report.stopped_by}  cycles={report.cycles}")
    print(f"{'='*64}")
    print(f"  total hypotheses proposed : {len(report.all_hypotheses)}")
    print(f"  verified                  : {len(report.verified_hypotheses)}")
    for vr in report.verified_hypotheses:
        print(f"    [{vr.status.value}] {vr.hypothesis.statement}")
        print(f"           effect={vr.effect_size}  confidence={vr.confidence:.2f}"
              f"  protocol_ok={vr.is_consistent_with_protocol}")
        print(f"           {vr.verdict}")

    # ── M4 ────────────────────────────────────────────────────────────────
    print(f"\n{'='*64}")
    print("M4  Fix proposal (post-loop)")
    print(f"{'='*64}")
    if report.verified_hypotheses:
        fix = loop.run_m4(report, cases)
        if fix is not None:
            print(f"  hypothesis : {fix.hypothesis.statement}")
            print(f"  status     : {fix.status.value}  fixed={fix.fixed}")
            for k, v in list((fix.evidence or {}).items())[:6]:
                print(f"  {k:20s}: {v}")
        else:
            print("  SurgeryAgent returned None.")
    else:
        print("  No verified hypotheses — skipping M4.")

    ctx.finalize()
    print(f"\n  Full guide -> {ctx.root / 'README.txt'}")
    print("\nDone.")


if __name__ == "__main__":
    main()
