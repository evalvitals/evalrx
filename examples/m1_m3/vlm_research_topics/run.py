"""VLM Research Topics: three paper-grounded VLM failure scenarios for VLDiagnoseLoop.

Each scenario reproduces a known failure mode from the vision-language model
literature.  Running each scenario through VLDiagnoseLoop (M1 → M2 → M3 → M5)
shows which evalvitals analyzers best characterise the failure.

Scenarios (select with --scenario):

  spatial   Spatial relationship failures.
            Paper: CV-Bench (Tong et al., "Eyes Wide Shut? Exploring the Visual
            Shortcomings of Multimodal LLMs", ICLR 2025).
            Failure: VLMs confuse left/right and above/below when processing
            relational queries, even for clearly placed objects.
            Expected analyzers: attention, relative_attention (WB);
                                self_consistency (BB fallback).

  counting  Object counting hallucination.
            Papers: POPE (Li et al., "Evaluating Object Hallucination in Large
            Vision-Language Models", EMNLP 2023);
            TallyQA (Acharya et al., AAAI 2019).
            Failure: VLMs undercount / overcount objects beyond the subitizing
            limit (~4); attention diffuses rather than tracking each instance.
            Expected analyzers: attention (WB); self_consistency,
                                token_entropy (BB).

  binding   Attribute binding / compositional VQA.
            Papers: Winoground (Thrush et al., "Winoground: Probing Vision and
            Language Models for Visio-Linguistic Compositionality", ACL 2022);
            ARO (Yuksekgonul et al., "When and Why Vision-Language Models Behave
            like Bags-of-Words", ICLR 2023).
            Failure: VLMs recognise objects and attributes as independent
            bag-of-words features but fail to bind them to the correct referent.
            Expected analyzers: attention_rollout, relative_attention (WB);
                                self_consistency (BB fallback).

Pipeline (identical to qwen_loop_agy):

    ExperimentProtocol  ← paper-grounded NL description of the failure mode
         │
    M1  ProbeAgent           selects analyzers from the evalvitals catalog
    M2  StatsAnalysisAgent   stats tools + e-BH FDR-correction + evidence chain
    M3  DiagnosisAgent       LLM judge proposes hypotheses
    M5  HypothesisTester     statistical test + protocol consistency check
         │
    loop exits on verified hypothesis or after --max-cycles
         │
    M4  SurgeryAgent         writes + runs targeted fix script (post-loop)

Outputs (default: <project-root>/runs/<scenario>/), see manifest.json + the
auto-generated README.txt for the full index:
    run_log.jsonl                ← one JSON line per M1/M2/M3/M5 event
    artifacts/                   ← per-cycle analyzer artifacts
    report/discovery_cases.json  ← labeled case set after model execution
    report/hypotheses.json       ← all hypotheses generated
    report/m5_results.json       ← hypothesis test results
    report/summary.md            ← human-readable summary

Usage (via Docker — preferred):
    SCENARIO=spatial  docker compose up   # default
    SCENARIO=counting docker compose up
    SCENARIO=binding  docker compose up

Usage (direct):
    python run.py --scenario spatial
    python run.py --scenario counting --max-cycles 3
    python run.py --scenario binding --analysis-only
    python run.py --smoke-test --scenario counting
"""

from __future__ import annotations

import argparse
import re
import textwrap
from pathlib import Path

_OUTPUTS_DIR = Path(__file__).parent.parent.parent / "runs"
_SCENARIOS = ("spatial", "counting", "binding")


# ---------------------------------------------------------------------------
# Robust case scorer (word-boundary aware)
# ---------------------------------------------------------------------------

def _contains(term: str, text: str) -> bool:
    """Word-boundary match for plain alphanumerics; substring for punctuated terms."""
    term = term.lower().strip()
    if not term:
        return False
    if re.fullmatch(r"[a-z0-9]+", term):
        return re.search(rf"\b{re.escape(term)}\b", text) is not None
    return term in text


def _score_case(case, observed) -> "Label":
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
# Image builders
# ---------------------------------------------------------------------------

def _spatial_image():
    """Synthesize image for the *spatial* scenario (224 × 224).

    Layout:
      - Red filled circle:    bbox [20, 140, 100, 210]          → LEFT-BOTTOM
      - Blue filled triangle: vertices (170,20)-(128,92)-(212,92) → TOP-RIGHT
      - Green filled square:  bbox [82, 78, 152, 140]            → CENTRE
    """
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (224, 224), color=(235, 235, 235))
    draw = ImageDraw.Draw(img)
    draw.ellipse([20, 140, 100, 210], fill=(210, 60, 55))
    draw.polygon([(170, 20), (128, 92), (212, 92)], fill=(55, 90, 210))
    draw.rectangle([82, 78, 152, 140], fill=(55, 175, 75))
    return img


def _counting_image():
    """Synthesize image for the *counting* scenario (224 × 224).

    Layout: 7 filled circles in a loose 3×3 grid.
      - 3 red circles   (positions: top-left, top-right, centre)
      - 2 blue circles  (positions: top-centre, bottom-right)
      - 2 yellow circles (positions: middle-left, bottom-centre)
    """
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (224, 224), color=(245, 245, 230))
    draw = ImageDraw.Draw(img)
    circles = [
        (38,  38,  20, (210, 55, 55)),   # red   — top-left
        (112, 38,  20, (55, 90, 200)),   # blue  — top-centre
        (186, 38,  20, (210, 55, 55)),   # red   — top-right
        (38,  112, 20, (220, 200, 50)),  # yellow — middle-left
        (112, 112, 20, (210, 55, 55)),   # red   — centre
        (186, 112, 20, (55, 90, 200)),   # blue  — middle-right
        (112, 186, 20, (220, 200, 50)),  # yellow — bottom-centre
    ]
    for cx, cy, r, fill in circles:
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=fill)
    return img


def _binding_image():
    """Synthesize image for the *binding* scenario (224 × 224).

    Layout (white background):
      - Small blue circle:    bbox [20, 20, 80, 80]     → TOP-LEFT
      - Large red rectangle:  bbox [110, 110, 210, 210] → BOTTOM-RIGHT
    """
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (224, 224), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.ellipse([20, 20, 80, 80], fill=(55, 90, 210))
    draw.rectangle([110, 110, 210, 210], fill=(210, 60, 55))
    return img


# ---------------------------------------------------------------------------
# Case builders
# ---------------------------------------------------------------------------

def _spatial_cases(image):
    """Cases for the spatial-relationship failure scenario.

    Easy  (reliable PASS): existence, colour, shape-type.
    Hard  (likely FAIL):   left/right, above/below, quadrant placement.

    Ground-truth layout:
      red circle    → left-bottom   (centre ≈ 60, 175)
      blue triangle → top-right     (centre ≈ 170, 68)
      green square  → centre        (centre ≈ 117, 109)
    """
    from evalvitals.core.case import CaseBatch, FailureCase, Inputs

    return CaseBatch([
        # ── Easy ──────────────────────────────────────────────────────────────
        FailureCase(
            id="sp_exists_circle",
            inputs=Inputs(
                prompt="Is there a circle in this image? Answer yes or no.",
                image=image,
            ),
            expected={"all_of": ["yes"], "none_of": ["no"]},
        ),
        FailureCase(
            id="sp_list_shapes",
            inputs=Inputs(
                prompt=(
                    "List all the distinct shapes you can see in this image. "
                    "Answer with comma-separated shape names only."
                ),
                image=image,
            ),
            expected={"any_of": ["circle", "triangle", "square"]},
        ),
        FailureCase(
            id="sp_color_circle",
            inputs=Inputs(
                prompt="What color is the circular shape? Answer with one color name only.",
                image=image,
            ),
            expected={"any_of": ["red"]},
        ),
        # ── Hard: relational spatial queries (CV-Bench failure mode) ─────────
        # red circle (left) vs blue triangle (right):
        FailureCase(
            id="sp_left_right",
            inputs=Inputs(
                prompt=(
                    "Is the red circle to the LEFT of the blue triangle? "
                    "Answer yes or no."
                ),
                image=image,
            ),
            expected={"all_of": ["yes"], "none_of": ["no"]},
        ),
        # red circle (bottom) vs green square (centre):
        FailureCase(
            id="sp_above_below",
            inputs=Inputs(
                prompt=(
                    "Is the red circle BELOW the green square? "
                    "Answer yes or no."
                ),
                image=image,
            ),
            expected={"all_of": ["yes"], "none_of": ["no"]},
        ),
        # blue triangle occupies top-right quadrant:
        FailureCase(
            id="sp_quadrant",
            inputs=Inputs(
                prompt=(
                    "Which shape is located in the TOP-RIGHT area of the image? "
                    "Answer with just the shape name."
                ),
                image=image,
            ),
            expected={"any_of": ["triangle"]},
        ),
    ])


def _counting_cases(image):
    """Cases for the counting-hallucination scenario.

    Easy  (reliable PASS): presence, coarse magnitude comparisons.
    Hard  (likely FAIL):   exact total, per-colour subset, parity.

    Ground truth: 7 circles total — 3 red, 2 blue, 2 yellow.
    """
    from evalvitals.core.case import CaseBatch, FailureCase, Inputs

    return CaseBatch([
        # ── Easy ──────────────────────────────────────────────────────────────
        FailureCase(
            id="ct_any_circles",
            inputs=Inputs(
                prompt="Are there any circles in this image? Answer yes or no.",
                image=image,
            ),
            expected={"all_of": ["yes"], "none_of": ["no"]},
        ),
        FailureCase(
            id="ct_more_than_three",
            inputs=Inputs(
                prompt="Are there more than 3 circles in this image? Answer yes or no.",
                image=image,
            ),
            expected={"all_of": ["yes"], "none_of": ["no"]},
        ),
        FailureCase(
            id="ct_fewer_than_fifteen",
            inputs=Inputs(
                prompt="Are there fewer than 15 circles in this image? Answer yes or no.",
                image=image,
            ),
            expected={"all_of": ["yes"], "none_of": ["no"]},
        ),
        # ── Hard: exact and subset counts (POPE / TallyQA failure mode) ──────
        FailureCase(
            id="ct_exact_total",
            inputs=Inputs(
                prompt="How many circles are in this image? Answer with just the number.",
                image=image,
            ),
            expected={"any_of": ["7", "seven"]},
        ),
        FailureCase(
            id="ct_subset_red",
            inputs=Inputs(
                prompt="How many RED circles are in this image? Answer with just the number.",
                image=image,
            ),
            expected={"any_of": ["3", "three"]},
        ),
        FailureCase(
            id="ct_parity",
            inputs=Inputs(
                prompt=(
                    "Is the total number of circles in this image an odd number? "
                    "Answer yes or no."
                ),
                image=image,
            ),
            expected={"all_of": ["yes"], "none_of": ["no"]},
        ),
    ])


def _binding_cases(image):
    """Cases for the attribute-binding failure scenario.

    Easy  (reliable PASS): individual attribute / existence queries.
    Hard  (likely FAIL):   queries requiring attribute → referent binding.

    Ground truth: small blue circle (top-left), large red rectangle (bottom-right).
    """
    from evalvitals.core.case import CaseBatch, FailureCase, Inputs

    return CaseBatch([
        # ── Easy ──────────────────────────────────────────────────────────────
        FailureCase(
            id="bd_blue_exists",
            inputs=Inputs(
                prompt="Is there a blue shape in this image? Answer yes or no.",
                image=image,
            ),
            expected={"all_of": ["yes"], "none_of": ["no"]},
        ),
        FailureCase(
            id="bd_red_exists",
            inputs=Inputs(
                prompt="Is there a red shape in this image? Answer yes or no.",
                image=image,
            ),
            expected={"all_of": ["yes"], "none_of": ["no"]},
        ),
        FailureCase(
            id="bd_circle_exists",
            inputs=Inputs(
                prompt="Is the blue shape a circle? Answer yes or no.",
                image=image,
            ),
            expected={"all_of": ["yes"], "none_of": ["no"]},
        ),
        # ── Hard: attribute-to-referent binding (Winoground / ARO failure mode)
        FailureCase(
            id="bd_color_small",
            inputs=Inputs(
                prompt=(
                    "What color is the SMALLER shape in this image? "
                    "Answer with one color name only."
                ),
                image=image,
            ),
            expected={"any_of": ["blue"]},
        ),
        FailureCase(
            id="bd_shape_large",
            inputs=Inputs(
                prompt=(
                    "What shape is the LARGER object in this image? "
                    "Answer with one shape name only."
                ),
                image=image,
            ),
            expected={"any_of": ["rectangle", "square"]},
        ),
        FailureCase(
            id="bd_same_color",
            inputs=Inputs(
                prompt=(
                    "Is the small shape the same color as the large shape? "
                    "Answer yes or no."
                ),
                image=image,
            ),
            expected={"all_of": ["no"], "none_of": ["yes"]},
        ),
    ])


# ---------------------------------------------------------------------------
# Protocol builders
# ---------------------------------------------------------------------------

def _spatial_protocol():
    from evalvitals.eval_agent import ExperimentProtocol

    return ExperimentProtocol(
        description=(
            "We evaluate a VLM on spatial relationship understanding, reproducing "
            "the failure mode documented in CV-Bench (Tong et al., ICLR 2025). "
            "The image contains three distinctly coloured geometric shapes placed "
            "in different quadrants: a red circle (left-bottom), a blue triangle "
            "(top-right), and a green square (centre). The model correctly "
            "identifies shape existence and colours (easy, language-prior cases) "
            "but fails on relational queries — confusing which shape is left vs. "
            "right, above vs. below, or in which quadrant. We hypothesise that "
            "the visual attention mechanism does not reliably localise to the "
            "correct spatial region when processing relational queries, and that "
            "attention entropy is higher on spatial-question failures than on "
            "basic existence questions. Use attention analysis (per-layer weights, "
            "relative_attention to image regions, attention_rollout) to determine "
            "whether failures reflect a vision-encoding issue (early layers) or a "
            "cross-modal reasoning issue (late layers). Self-consistency across "
            "multiple samples will reveal whether spatial answers are randomly "
            "wrong (high variance) or systematically wrong in the same direction."
        ),
        task_domain="spatial visual question answering",
        success_criteria=(
            "Relational answers (left/right, above/below, quadrant) must match "
            "the ground-truth positions of objects in the image."
        ),
        target_modalities=frozenset({"text", "image"}),
    )


def _counting_protocol():
    from evalvitals.eval_agent import ExperimentProtocol

    return ExperimentProtocol(
        description=(
            "We evaluate a VLM on object counting, reproducing the hallucination "
            "patterns documented in POPE (Li et al., EMNLP 2023) and TallyQA "
            "(Acharya et al., AAAI 2019). The image contains 7 circles in three "
            "colours (3 red, 2 blue, 2 yellow). The model correctly handles coarse "
            "magnitude questions (any circles? more than 3?) but fails on exact "
            "counts — typically reporting 5 or 6 instead of 7 — consistent with "
            "the subitizing limit (~4 objects) documented in the VLM counting "
            "literature. Per the POPE hallucination benchmark, VLMs also over- or "
            "under-count per-category subsets. We hypothesise that (a) "
            "self_consistency is low for exact count answers (high variance across "
            "samples, indicating guessing), (b) token_entropy is elevated at the "
            "count-token position compared to binary questions, and (c) attention "
            "analysis shows diffuse, non-localised attention on counting failures "
            "rather than systematic per-circle tracking. Use attention analysis, "
            "token_entropy, and self_consistency to characterise the mechanism."
        ),
        task_domain="object counting visual question answering",
        success_criteria=(
            "Exact counts and per-colour subset counts must match the ground-truth "
            "object counts visible in the image (total=7, red=3, blue=2, yellow=2)."
        ),
        target_modalities=frozenset({"text", "image"}),
    )


def _binding_protocol():
    from evalvitals.eval_agent import ExperimentProtocol

    return ExperimentProtocol(
        description=(
            "We evaluate a VLM on attribute binding in compositional VQA, "
            "reproducing the failure mode documented in Winoground (Thrush et al., "
            "ACL 2022) and the ARO benchmark (Yuksekgonul et al., ICLR 2023). "
            "The image has two shapes with distinguishing (colour, size) pairs: a "
            "small blue circle (top-left) and a large red rectangle (bottom-right). "
            "The model correctly identifies that blue and red shapes exist (easy, "
            "global-feature queries) but fails when asked to bind an attribute to "
            "the correct referent — e.g., 'what color is the SMALL shape?' often "
            "returns 'red' instead of 'blue', and 'what shape is the LARGE object?' "
            "returns 'circle' instead of 'rectangle'. This is consistent with the "
            "ARO bag-of-words finding: VLMs encode attributes and objects "
            "independently and cannot reliably compose them. Use attention_rollout "
            "to check whether the model attends equally to both objects on binding "
            "questions (diffuse) vs. selectively to the queried referent "
            "(localised). relative_attention to image regions will reveal whether "
            "the correct spatial location of the queried object is reflected in the "
            "cross-attention pattern. self_consistency will measure variance."
        ),
        task_domain="compositional visual question answering",
        success_criteria=(
            "Attribute-binding answers must correctly assign colour to the small "
            "shape (blue) and shape-type to the large object (rectangle or square)."
        ),
        target_modalities=frozenset({"text", "image"}),
    )


# ---------------------------------------------------------------------------
# Smoke-test stubs
# ---------------------------------------------------------------------------

class _SmokeVLM:
    """Deterministic VLM stand-in for wiring tests.  No GPU / weights required."""

    def __init__(self) -> None:
        from evalvitals.core.capability import Capability

        self.capabilities = frozenset({Capability.GENERATE, Capability.ATTENTION})
        self.modalities = frozenset({"text", "image"})

    def __repr__(self) -> str:
        return "SmokeVLM()"

    def generate(self, inputs, **kwargs) -> str:  # noqa: C901
        p = str(getattr(inputs, "prompt", inputs)).lower()
        # ── spatial ─────────────────────────────────────────────────────────
        if "is there a circle" in p:
            return "Yes."
        if "list all the distinct shapes" in p:
            return "circle, triangle, square"
        if "color is the circular shape" in p:
            return "red"
        if "red circle to the left" in p:
            return "No, the red circle is to the right."       # wrong → FAIL
        if "red circle below" in p:
            return "No, the red circle is above the square."   # wrong → FAIL
        if "which shape is located in the top-right" in p:
            return "The circle."                               # wrong → FAIL
        # ── counting ────────────────────────────────────────────────────────
        if "are there any circles" in p:
            return "Yes, there are circles in the image."
        if "more than 3 circles" in p:
            return "Yes."
        if "fewer than 15 circles" in p:
            return "Yes."
        if "how many circles are in this image" in p:
            return "5"                                         # wrong → FAIL
        if "how many red circles" in p:
            return "2"                                         # wrong → FAIL
        if "total number of circles" in p and "odd" in p:
            return "No, the count appears to be even."        # wrong → FAIL
        # ── binding ─────────────────────────────────────────────────────────
        if "is there a blue shape" in p:
            return "Yes."
        if "is there a red shape" in p:
            return "Yes."
        if "blue shape a circle" in p:
            return "Yes."
        if "color is the smaller shape" in p:
            return "Red."                                      # wrong → FAIL
        if "shape is the larger object" in p:
            return "Circle."                                   # wrong → FAIL
        if "small shape the same color as the large" in p:
            return "Yes."                                      # wrong → FAIL
        return "Unknown."

    def forward(self, inputs, capture, spec=None):
        raise NotImplementedError("SmokeVLM only supports generate().")


class _SmokeProbe:
    """M1 stand-in: marks FAIL cases with a synthetic attention signal."""

    last_schema = None

    def probe(self, model, data, **kwargs):
        from evalvitals.core.case import Label
        from evalvitals.core.result import Result
        from evalvitals.eval_agent import ProbingSchema

        fail_ids = [case.id for case in data if case.label == Label.FAIL]
        self.last_schema = ProbingSchema(
            selected_analyzers=["attention"],
            rationale="Smoke probe: synthetic attention entropy signal on FAIL cases.",
            protocol=kwargs.get("protocol"),
        )
        findings = {
            "mean_entropy": 0.42,
            "per_case": [
                {"sample_id": cid, "attention_entropy": 2.1, "attention_signal": True}
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


_SMOKE_HYPOTHESES = {
    "spatial": (
        "The model fails relational spatial queries because attention does not "
        "localise to the correct region when processing left/right or above/below "
        "comparisons.",
        "spatial_attention_failure",
    ),
    "counting": (
        "The model undercounts objects beyond the subitizing limit because attention "
        "diffuses across the image rather than tracking each instance individually.",
        "counting_attention_diffusion",
    ),
    "binding": (
        "The model treats attributes as global bag-of-words features and fails to "
        "bind colour or shape to the correct referent.",
        "attribute_binding_failure",
    ),
}


class _SmokeDiagnosis:
    """M3 stand-in: emits one scenario-specific hypothesis."""

    def __init__(self, scenario: str) -> None:
        self._scenario = scenario

    def diagnose(self, analysis, prior_cycles=None):
        from evalvitals.eval_agent import DiagnosisResult, Hypothesis

        stmt, mode = _SMOKE_HYPOTHESES[self._scenario]
        h = Hypothesis(
            statement=stmt,
            target_model=analysis.model_name,
            predicted_failure_mode=mode,
        )
        return DiagnosisResult(
            model_name=analysis.model_name,
            hypotheses=[h],
            findings_summary={name: r.findings for name, r in analysis.raw_results.items()},
            raw_judge_output=f"HYPOTHESIS: {stmt}\nFAILURE_MODE: {mode}",
        )


def _run_smoke_test(args, scenario: str) -> None:
    from evalvitals.eval_agent import (
        CaseDiscoveryAgent,
        HypothesisTester,
        RunContext,
        StatsAnalysisAgent,
        StatsToolAgent,
        SurgeryAgent,
        VLDiagnoseLoop,
    )

    _image_fns = {"spatial": _spatial_image, "counting": _counting_image, "binding": _binding_image}
    _cases_fns = {"spatial": _spatial_cases, "counting": _counting_cases, "binding": _binding_cases}
    _proto_fns = {"spatial": _spatial_protocol, "counting": _counting_protocol, "binding": _binding_protocol}

    model = _SmokeVLM()
    image = _image_fns[scenario]()
    protocol = _proto_fns[scenario]()
    candidate_cases = _cases_fns[scenario](image)

    discovery = CaseDiscoveryAgent(
        scorer=_score_case,
        include_unknown=False,
    ).discover(model, candidate_cases, protocol=protocol)
    cases = discovery.cases

    print(f"\nSmoke test [{scenario}]:")
    print(
        f"  discovered {len(cases)} labeled cases "
        f"(PASS={discovery.n_pass}, FAIL={discovery.n_fail}, UNKNOWN={discovery.n_unknown})"
    )
    if not discovery.has_m5_groups:
        raise SystemExit(f"Smoke test [{scenario}] requires both PASS and FAIL cases.")

    run_dir = Path(args.run_dir) / scenario if args.run_dir else _OUTPUTS_DIR / scenario
    ctx = RunContext(run_dir, verbose=True, config={"smoke_test": True, "scenario": scenario})

    loop = VLDiagnoseLoop(
        model=model,
        protocol=protocol,
        probe_agent=_SmokeProbe(),
        stats_agent=StatsAnalysisAgent(stats_tool_agent=StatsToolAgent(max_tools=3)),
        diagnosis_agent=_SmokeDiagnosis(scenario),
        hypothesis_tester=HypothesisTester(min_effect=0.05),
        surgery_agent=SurgeryAgent(),
        max_cycles=1,
        run_logger=ctx.logger,
    )
    report = loop.run(cases)
    ctx.write_diagnose_report(report, cases)
    ctx.finalize()

    print(f"\nSmoke test [{scenario}] result:")
    print(f"  stopped_by={report.stopped_by}  cycles={report.cycles}")
    print(f"  verified={len(report.verified_hypotheses)}")
    if report.stopped_by != "criteria_met" or not report.verified_hypotheses:
        raise SystemExit(f"Smoke test [{scenario}] failed: no verified hypothesis.")

    fix = loop.run_m4(report, cases)
    if fix is None or fix.status.value != "supported":
        raise SystemExit(f"Smoke test [{scenario}] failed: M4 did not support hypothesis.")

    print("  m4_status=supported")
    print(f"Smoke test [{scenario}] passed.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="VLM research topics: paper-grounded failure scenarios for VLDiagnoseLoop"
    )
    parser.add_argument(
        "--scenario", choices=_SCENARIOS, default="spatial",
        help="Failure scenario to run (default: spatial).",
    )
    parser.add_argument("--model", default="qwen3-vl-4b-instruct")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument(
        "--judge-model", default="Gemini 3.1 Pro (Low)",
        help="agy model for M1–M5 judge.",
    )
    parser.add_argument("--max-cycles", type=int, default=2)
    parser.add_argument("--max-analyzers", type=int, default=2)
    parser.add_argument(
        "--smoke-test", action="store_true",
        help="Fast wiring test — no GPU, no Qwen, no agy required.",
    )
    parser.add_argument(
        "--allow-codegen", action="store_true",
        help="M1/M2 tier(b): generate bespoke stats tools in a sandbox.",
    )
    parser.add_argument(
        "--analysis-only", action="store_true",
        help="Run M1+M2 only (skip M3/M5/M4).",
    )
    parser.add_argument(
        "--run-dir", default=None,
        help=(
            "Output root.  Outputs land in <run-dir>/<scenario>/. "
            "Defaults to <project-root>/runs/<scenario>/."
        ),
    )
    args = parser.parse_args()

    scenario = args.scenario

    if args.smoke_test:
        _run_smoke_test(args, scenario)
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

    _image_fns = {"spatial": _spatial_image, "counting": _counting_image, "binding": _binding_image}
    _cases_fns = {"spatial": _spatial_cases, "counting": _counting_cases, "binding": _binding_cases}
    _proto_fns = {"spatial": _spatial_protocol, "counting": _counting_protocol, "binding": _binding_protocol}

    run_dir = (Path(args.run_dir) if args.run_dir else _OUTPUTS_DIR) / scenario
    ctx = RunContext(
        run_dir, verbose=True,
        config={
            "model": args.model,
            "judge_model": args.judge_model,
            "scenario": scenario,
            "max_cycles": args.max_cycles,
            "max_analyzers": args.max_analyzers,
            "analysis_only": args.analysis_only,
        },
    )

    # ── Load model ────────────────────────────────────────────────────────────
    print(f"\nLoading {args.model!r} on {args.device} ({args.dtype}) …  [scenario={scenario}]")
    model = evalvitals.load(
        args.model,
        backend="hf_local",
        device=args.device,
        dtype=args.dtype,
        want=["attention"],
    )
    print(f"  capabilities : {sorted(str(c.name) for c in model.capabilities)}")
    print(f"  modalities   : {sorted(model.modalities)}")

    # ── Judge ─────────────────────────────────────────────────────────────────
    try:
        judge = AgyModel(model=args.judge_model)
        print(f"\n  judge : antigravity CLI  model={args.judge_model or 'session default'}  [M1–M5]")
    except RuntimeError as _agy_err:
        import warnings as _w
        _w.warn(
            f"agy not available ({_agy_err}). Falling back to loaded model as judge.",
            stacklevel=2,
        )
        judge = model
        print(f"\n  judge : {args.model} (agy unavailable — using evaluated model as fallback)")

    # ── Protocol + cases ──────────────────────────────────────────────────────
    protocol = _proto_fns[scenario]()
    print(f"\nExperimentProtocol [{scenario}]:")
    print(f"  task_domain : {protocol.task_domain}")
    print(f"  description : {protocol.description[:90]}...")

    print("\nPreparing image and cases …")
    image = _image_fns[scenario]()
    candidate_cases = _cases_fns[scenario](image)
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

    # ── Agents ────────────────────────────────────────────────────────────────
    _m1_codegen = args.allow_codegen and not args.analysis_only
    probe_agent = ProbeAgent(
        judge=judge,
        max_analyzers=args.max_analyzers,
        allow_codegen=_m1_codegen,
        codegen_config=(
            CliAgentConfig(provider="antigravity", timeout_sec=120, model=args.judge_model)
            if _m1_codegen else None
        ),
    )
    stats_agent = StatsAnalysisAgent(
        judge=None if args.analysis_only else judge,
        figure_dir=str(ctx.figures_dir),
        allow_codegen=args.allow_codegen and not args.analysis_only,
        codegen_config=(
            CliAgentConfig(provider="antigravity", timeout_sec=120, model=args.judge_model)
            if args.allow_codegen and not args.analysis_only else None
        ),
    )
    diagnosis_agent = None if args.analysis_only else DiagnosisAgent(judge=judge)
    hypothesis_tester = None if args.analysis_only else HypothesisTester(judge=judge, min_effect=0.05)
    surgery_agent = None
    if not args.analysis_only:
        writer_cfg = ExperimentWriterConfig(
            cli_agent=CliAgentConfig(
                provider="antigravity", timeout_sec=120, model=args.judge_model
            ),
            exec_fix_timeout_sec=60,
        )
        surgery_agent = SurgeryAgent(judge=judge, writer_config=writer_cfg, run_context=ctx)

    # ── Loop ──────────────────────────────────────────────────────────────────
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
        analysis_only=args.analysis_only,
    )

    print(f"\n{'='*64}")
    print(f"VLDiagnoseLoop  model={args.model}  scenario={scenario}  max_cycles={args.max_cycles}")
    print(f"Output directory: {run_dir.resolve()}")
    print(f"{'='*64}")

    report = loop.run(cases)
    ctx.write_diagnose_report(report, cases, discovery=discovery_rows)

    print(f"\n{'='*64}")
    print(f"LOOP RESULT  stopped_by={report.stopped_by}  cycles={report.cycles}")
    print(f"  total hypotheses   : {len(getattr(report, 'all_hypotheses', []))}")
    print(f"  verified hypotheses: {len(getattr(report, 'verified_hypotheses', []))}")
    for vr in getattr(report, "verified_hypotheses", []):
        print(f"    [{vr.status.value}] {vr.hypothesis.statement}")
        print(
            f"           effect={vr.effect_size}  confidence={vr.confidence:.2f}"
            f"  protocol_ok={vr.is_consistent_with_protocol}"
        )
        print(f"           {vr.verdict}")

    if surgery_agent is not None:
        print(f"\n{'='*64}")
        print("M4  Fix proposal (post-loop)")
        print(f"{'='*64}")
        if getattr(report, "verified_hypotheses", []):
            fix = loop.run_m4(report, cases)
            if fix is not None:
                print(f"  hypothesis : {fix.hypothesis.statement}")
                print(f"  status     : {fix.status.value}  fixed={fix.fixed}")
                ev = fix.evidence or {}
                for k, v in list(ev.items())[:6]:
                    print(f"  {k:20s}: {v}")
            else:
                print("  SurgeryAgent returned None")
        else:
            print("  No verified hypotheses — skipping M4.")

    ctx.finalize()
    print(f"\n  Full guide -> {ctx.root / 'README.txt'}")
    print("Done.")


if __name__ == "__main__":
    main()
