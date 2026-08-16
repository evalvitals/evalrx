"""VLDiagnoseLoop on Qwen3-VL-4B: protocol-guided VL failure diagnosis.

Pipeline:

    ExperimentProtocol  ← user's NL description of what to investigate
         │
    M1  ProbeAgent           protocol-guided analyzer selection + execute
    M2  StatsAnalysisAgent   select stats tools (evalvitals.stats) + run +
                             e-BH FDR-correct + LLM-written evidence chain
    M3  DiagnosisAgent       Qwen as judge ("AI scientist" hypothesis gen)
    M5  HypothesisTester     statistical test + protocol consistency check
         │
    loop exits when M5 finds a verified, protocol-consistent hypothesis,
    or after --max-cycles cycles
         │
    M4  SurgeryAgent         agy/codex writes + runs targeted fix script
                             (called separately AFTER the loop)

Outputs written to --run-dir (default: ./outputs/), see manifest.json + the
auto-generated README.txt for the full index:
    run_log.jsonl          ← one JSON line per M1/M2/M3/M5 event
    artifacts/              ← per-cycle analyzer artifacts (.npy / .json)

Usage (via Docker — preferred):
    docker compose up

Usage (direct):
    python run.py
    python run.py --smoke-test     # fast local wiring test, no Qwen/GPU/agy
    python run.py --model qwen2.5-vl-7b-instruct --device cuda:0
    python run.py --analysis-only   # M1+M2 only, skip M3/M5/M4
    python run.py --max-cycles 3 --max-analyzers 3
"""

from __future__ import annotations

import argparse
import io
import re
import textwrap
import urllib.request
from pathlib import Path

# A stable, public sample photo (real objects/colours/layout for the VQA task).
# NOTE: the old Wikimedia thumbnail URL stopped working — it now 403s on the
# default urllib User-Agent, 400s on disallowed thumbnail sizes, and the file
# itself was removed (404). This raw GitHub asset needs no special headers.
_SAMPLE_URL = "https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg"
_OUTPUTS_DIR = Path(__file__).parent / "outputs"


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def _get_image(*, download: bool = False):
    from PIL import Image
    if not download:
        return _synthetic_image()
    try:
        # Wikimedia rejects the default urllib User-Agent with HTTP 403,
        # so send a descriptive UA as their policy requires.
        req = urllib.request.Request(
            _SAMPLE_URL,
            headers={"User-Agent": "evalvitals-example/1.0 (https://example.com; contact@example.com)"},
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = resp.read()
        img = Image.open(io.BytesIO(data)).convert("RGB")
        print(f"  Downloaded image: {img.size} px")
        return img
    except Exception as exc:
        print(f"  Could not download image ({exc}); generating synthetic fallback")
        return _synthetic_image()


def _synthetic_image():
    from PIL import Image, ImageDraw
    img = Image.new("RGB", (224, 224), color=(200, 220, 240))
    draw = ImageDraw.Draw(img)
    draw.rectangle([20, 20, 100, 100], fill=(220, 80, 60))   # red box left
    draw.rectangle([124, 20, 204, 100], fill=(60, 160, 80))  # green box right
    draw.rectangle([60, 130, 164, 190], fill=(80, 80, 200))  # blue box centre-bottom
    try:
        draw.text((10, 200), "synthetic test image", fill=(0, 0, 0))
    except Exception:
        pass
    return img


# ---------------------------------------------------------------------------
# Cases + robust scorer
# ---------------------------------------------------------------------------
#
# M5 needs BOTH passing and failing cases (a control group + a failure group) to
# run its fail-rate tests.  The default substring heuristic in CaseDiscoveryAgent
# is fragile on yes/no answers ("no" matches "snow"/"not"), and the old prompt
# set was all hard questions, so runs came out all-FAIL.  We fix both here:
#
#   - a word-boundary scorer (``_score_case``) so yes/no/colour terms match cleanly;
#   - a balanced prompt set: salient-feature questions a VLM reliably PASSES
#     (red present? list colours?) + precise-detail questions it reliably FAILS
#     (exact RGB hex, verbatim tiny caption).  This guarantees both M5 groups.


def _contains(term: str, text: str) -> bool:
    """Match *term* in *text*: word-boundary for plain alphanumerics (so "no"
    does not match "snow"/"not"), substring for terms with punctuation (hex)."""
    term = term.lower().strip()
    if not term:
        return False
    if re.fullmatch(r"[a-z0-9]+", term):
        return re.search(rf"\b{re.escape(term)}\b", text) is not None
    return term in text


def _score_case(case, observed):
    """Word-boundary-aware scorer for the dict/str ``expected`` rubrics."""
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


def _build_candidate_cases(image):
    """Balanced human-prior prompts: easy (reliably PASS) + hard (reliably FAIL).

    Labels are assigned after model execution by :func:`_score_case`.
    """
    from evalvitals.core.case import CaseBatch, FailureCase, Inputs

    return CaseBatch([
        # ── Easy: salient features a VLM reliably gets right → PASS (control) ──
        FailureCase(
            id="q_has_red",
            inputs=Inputs(
                prompt="Is there a red shape in this image? Answer yes or no.",
                image=image,
            ),
            expected={"all_of": ["yes"], "none_of": ["no"]},
        ),
        FailureCase(
            id="q_colors",
            inputs=Inputs(
                prompt="List the colors that appear in this image.",
                image=image,
            ),
            expected={"any_of": ["red", "green", "blue"]},
        ),
        FailureCase(
            id="q_count",
            inputs=Inputs(
                prompt="How many colored rectangles are in this image? "
                       "Answer with just the number.",
                image=image,
            ),
            expected={"any_of": ["3", "three"]},
        ),
        # ── Hard: precise pixel-level details a VLM cannot know → FAIL ──
        # (asks for the exact source values, which the model can only guess at).
        FailureCase(
            id="q_rgb",
            inputs=Inputs(
                prompt="What is the exact RGB hex code of the left rectangle? "
                       "Answer with the hex code only.",
                image=image,
            ),
            expected={"any_of": ["dc503c", "#dc503c"]},  # source fill is (220,80,60)
        ),
        FailureCase(
            id="q_coords",
            inputs=Inputs(
                prompt="What are the exact pixel coordinates (x1, y1, x2, y2) of "
                       "the blue rectangle? Answer with the four numbers.",
                image=image,
            ),
            expected={"all_of": ["60", "130", "164", "190"]},  # exact source box
        ),
        FailureCase(
            id="q_width",
            inputs=Inputs(
                prompt="Exactly how many pixels wide is the green rectangle? "
                       "Answer with one number.",
                image=image,
            ),
            expected={"any_of": ["80"]},  # green box spans x=124..204 → 80 px
        ),
    ])


# ---------------------------------------------------------------------------
# Smoke-test path
# ---------------------------------------------------------------------------

class _SmokeVLM:
    """Tiny deterministic VLM stand-in for testing this example's wiring."""

    def __init__(self) -> None:
        from evalvitals.core.capability import Capability

        self.capabilities = frozenset({Capability.GENERATE, Capability.ATTENTION})
        self.modalities = frozenset({"text", "image"})

    def __repr__(self) -> str:
        return "SmokeVLM()"

    def generate(self, inputs, **kwargs) -> str:
        prompt = str(getattr(inputs, "prompt", inputs)).lower()
        # Easy questions (salient features): answered correctly → PASS.
        if "is there a red shape" in prompt:
            return "Yes, there is a red shape."
        if "list the colors" in prompt:
            return "Red, green, and blue."
        if "how many colored rectangles" in prompt:
            return "3"
        # Hard questions (precise pixel-level details): the model can only
        # guess, so it gets these wrong → FAIL (by design, for M5 contrast).
        if "exact rgb hex code" in prompt:
            return "ff0000"
        if "exact pixel coordinates" in prompt:
            return "10, 20, 30, 40"
        if "pixels wide is the green rectangle" in prompt:
            return "42"
        return "Unknown."

    def forward(self, inputs, capture, spec=None):
        raise NotImplementedError("SmokeVLM only supports generate().")


class _SmokeProbe:
    """M1 probe that emits per-case signals on discovered failures."""

    last_schema = None

    def probe(self, model, data, **kwargs):
        from evalvitals.core.case import Label
        from evalvitals.core.result import Result
        from evalvitals.eval_agent import ProbingSchema

        fail_ids = [case.id for case in data if case.label == Label.FAIL]
        self.last_schema = ProbingSchema(
            selected_analyzers=["attention"],
            rationale="Smoke probe marks discovered failures as attention signals.",
            protocol=kwargs.get("protocol"),
        )
        findings = {
            "mean_entropy": 0.2,
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
    """M3 diagnosis stand-in with one protocol-consistent hypothesis."""

    def diagnose(self, analysis, prior_cycles=None):
        from evalvitals.eval_agent import DiagnosisResult, Hypothesis

        h = Hypothesis(
            statement=(
                "The model counts incorrectly when the attention diagnostic "
                "signal appears."
            ),
            target_model=analysis.model_name,
            predicted_failure_mode="attention",
        )
        return DiagnosisResult(
            model_name=analysis.model_name,
            hypotheses=[h],
            findings_summary={name: r.findings for name, r in analysis.raw_results.items()},
            raw_judge_output=(
                "HYPOTHESIS: The model counts incorrectly when the attention "
                "diagnostic signal appears.\nFAILURE_MODE: attention"
            ),
        )


def _build_protocol():
    from evalvitals.eval_agent import ExperimentProtocol

    return ExperimentProtocol(
        description=(
            "We evaluate a vision-language model on basic image understanding: "
            "counting objects, naming their colours, and describing their spatial "
            "arrangement. The model frequently gives wrong answers — it counts "
            "incorrectly, misnames colours, and confuses left/right positions. "
            "We want to know whether the model is actually using the image or "
            "just guessing from language patterns."
        ),
        task_domain="visual question answering",
        success_criteria=(
            "Object counts, colour names, and position descriptions must match "
            "what is visible in the image."
        ),
        target_modalities=frozenset({"text", "image"}),
    )


# Probe order: cheap/fast models first; quotas are per-model and reset on
# independent clocks, so whichever responds first is "the available version".
_JUDGE_CANDIDATES = (
    "Gemini 3.1 Pro (Low)",
    "Claude Sonnet 4.6 (Thinking)",
    "GPT-OSS 120B (Medium)",
    "Gemini 3.5 Flash (Low)",
    "Claude Opus 4.6 (Thinking)",
)


def _pick_agy_model() -> str:
    """Return the first agy model that actually answers a probe prompt.

    A quota-exhausted model returns an empty response (exit 0), so a tiny
    generation probe is the only reliable availability check.  Raises
    RuntimeError when every candidate is dead so the caller's existing
    fallback (loaded model as judge) takes over.
    """
    import warnings as _w

    from evalvitals.eval_agent import AgyModel

    for name in _JUDGE_CANDIDATES:
        try:
            probe = AgyModel(model=name, timeout_sec=60)
            with _w.catch_warnings():
                _w.simplefilter("ignore")  # quota warnings are expected here
                out = probe.generate("Reply with exactly the word OK")
            if out.strip():
                print(f"  judge probe: {name!r} responded — selected")
                return name
            print(f"  judge probe: {name!r} empty (likely quota-exhausted)")
        except RuntimeError as exc:
            print(f"  judge probe: {name!r} failed ({str(exc)[:80]})")
    raise RuntimeError(
        "no agy model responded to the availability probe "
        f"(tried {len(_JUDGE_CANDIDATES)} candidates — quotas likely exhausted)"
    )


def _build_protocol_med():
    from evalvitals.eval_agent import ExperimentProtocol

    return ExperimentProtocol(
        description=(
            "We evaluate a general vision-language model on radiology VQA "
            "(VQA-RAD): identification questions (imaging modality, plane, "
            "organ) and closed yes/no finding-presence questions (e.g. 'is "
            "there evidence of a pneumothorax?'). The model handles "
            "identification well but is unreliable on finding presence — its "
            "yes/no answers often contradict the radiologist gold label. We "
            "want to know whether presence answers are grounded in the scan, "
            "and whether the errors are hallucinated findings (answering yes "
            "to absent findings) or missed findings (answering no to present "
            "findings)."
        ),
        task_domain="medical visual question answering",
        success_criteria=(
            "Presence answers must match the radiologist gold label: 'yes' "
            "only when the finding is actually visible in the image, 'no' "
            "otherwise. Identification answers must name the correct "
            "modality/plane/organ."
        ),
        failure_patterns=(
            "Failures concentrate on finding-presence questions, while "
            "identification questions are mostly answered correctly."
        ),
        target_modalities=frozenset({"text", "image"}),
    )


def _build_medical_cases(args):
    """Load the VQA-RAD diagnosis mix (easy control + presence yes/no)."""
    from evalvitals.datasets import VQARADDataset

    ds = VQARADDataset(
        split="train",
        n_easy=args.n_easy,
        n_presence=args.n_presence,
        seed=0,
    )
    cases = ds.load()
    n_easy = sum(1 for c in cases if c.metadata.get("category") == "easy")
    n_pres = sum(1 for c in cases if c.metadata.get("category") == "presence")
    print(f"  VQA-RAD cases: {len(cases)} (easy={n_easy}, presence={n_pres})")
    return cases


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
    image = _synthetic_image()
    protocol = _build_protocol()
    discovery = CaseDiscoveryAgent(include_unknown=False).discover(
        model, _build_candidate_cases(image), protocol=protocol
    )
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
    print(f"  stopped_by={report.stopped_by} cycles={report.cycles}")
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
    parser = argparse.ArgumentParser(
        description="VLDiagnoseLoop on Qwen VL + image (new M1→M2→M3→M5+M4 pipeline)"
    )
    parser.add_argument("--model", default="qwen3-vl-4b-instruct")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument(
        "--judge-model", default="auto",
        help="agy model for the M1–M5 judge. 'auto' (default) probes a list of "
             "candidates at startup and picks the first that responds — agy "
             "quotas are per-model and exhaust independently. Pass an explicit "
             "name from `agy models` to skip probing; empty = session default.",
    )
    parser.add_argument("--max-cycles", type=int, default=2)
    parser.add_argument("--max-analyzers", type=int, default=2)
    parser.add_argument(
        "--analyzers", default="",
        help="Comma-separated analyzer names to pin for M1 (e.g. "
             "'pope,relative_attention,prompt_contrast'). Bypasses the LLM "
             "selection for reproducible experiments; empty = LLM-guided.",
    )
    parser.add_argument(
        "--depth", choices=["observational", "intervention"], default="observational",
        help="M5 stopping depth (P4): 'observational' stops on any supported "
             "hypothesis; 'intervention' keeps cycling until a hypothesis is "
             "verified by intervention-grade evidence (paired prompt contrast).",
    )
    parser.add_argument(
        "--smoke-test", action="store_true",
        help="Run a fast local wiring test without loading Qwen, GPU, or agy.",
    )
    parser.add_argument(
        "--scenario", choices=["synthetic", "vqa-rad"], default="synthetic",
        help="synthetic: labeled toy image (default). vqa-rad: radiology VQA "
             "from HuggingFace flaviagiammarino/vqa-rad — easy identification "
             "questions as the PASS control group + yes/no finding-presence "
             "questions where presence hallucination concentrates failures.",
    )
    parser.add_argument(
        "--n-easy", type=int, default=6,
        help="vqa-rad: number of easy identification questions (control group).",
    )
    parser.add_argument(
        "--n-presence", type=int, default=12,
        help="vqa-rad: number of yes/no presence questions (balanced yes/no gold).",
    )
    parser.add_argument(
        "--download-image", action="store_true",
        help="Use the demo Wikimedia image instead of the synthetic labeled image.",
    )
    parser.add_argument(
        "--allow-codegen", action="store_true",
        help="M2 tier(b): let the agent write+run a bespoke stats tool in a "
             "sandbox when no built-in tool fits (uses antigravity to write code).",
    )
    parser.add_argument(
        "--analysis-only", action="store_true",
        help="Run M1+M2 only (skip M3/M5/M4)",
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

    # ── Load model ────────────────────────────────────────────────────────────
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

    # ── Judge: agy CLI preferred; falls back to the loaded model ─────────────
    # agy requires no API key — it uses the user's existing OAuth token.
    # If the binary is not properly mounted, the loaded VLM acts as judge
    # (a warning is printed and the rationale will reflect this).
    try:
        judge_model = args.judge_model
        if judge_model == "auto":
            judge_model = _pick_agy_model()
        # 240s: M3 prompts now carry ~20 statistical verdicts + image attachments,
        # which routinely exceed agy's 120s default.
        judge = AgyModel(model=judge_model, timeout_sec=240)
        print(f"\n  judge : antigravity CLI ({judge._binary})  "
              f"model={judge_model or 'session default'}  [M1–M5, no API key]")
    except RuntimeError as _agy_err:
        import warnings as _w
        _w.warn(
            f"agy not available ({_agy_err}). "
            "Falling back to the loaded model as judge. "
            "To use agy: export AGY_PATH=$(which agy) before docker compose up.",
            stacklevel=2,
        )
        judge = model
        print(f"\n  judge : {args.model} (agy unavailable — using evaluated model as fallback)")

    run_dir = Path(args.run_dir)
    ctx = RunContext(
        run_dir, verbose=True,
        config={
            "model": args.model,
            "judge_model": judge_model,
            "scenario": args.scenario,
            "max_cycles": args.max_cycles,
            "max_analyzers": args.max_analyzers,
            "analysis_only": args.analysis_only,
        },
    )

    # ── Protocol + candidate cases (per scenario) ─────────────────────────────
    if args.scenario == "vqa-rad":
        protocol = _build_protocol_med()
        print("\nLoading VQA-RAD cases …")
        candidate_cases = _build_medical_cases(args)
    else:
        protocol = _build_protocol()
        print("\nPreparing image …")
        image = _get_image(download=args.download_image)
        candidate_cases = _build_candidate_cases(image)
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
    if discovery.errors:
        print(f"  discovery errors: {len(discovery.errors)}")
    if not discovery.has_m5_groups:
        print(
            "  WARNING: M5 needs both PASS and FAIL cases for fail-rate tests; "
            "this run may stop without verified hypotheses."
        )

    # ── Experiment protocol (the human prior) ─────────────────────────────────
    print("\nExperimentProtocol:")
    print(f"  task_domain : {protocol.task_domain}")
    print(f"  description : {protocol.description[:80]}...")

    # ── M1: ProbeAgent — agy selects analyzers from the protocol ─────────────
    # With --allow-codegen, M1 also generates a bespoke black-box probe (run in a
    # sandbox over the model's outputs) when no catalog analyzer fits the failure.
    _m1_codegen = args.allow_codegen and not args.analysis_only
    if args.analyzers.strip():
        # Pinned analyzer list — deterministic M1 for reproducible experiments.
        from evalvitals.eval_agent import StrategyProbe

        pinned = [a.strip() for a in args.analyzers.split(",") if a.strip()]
        print(f"  M1 pinned analyzers: {pinned}")
        probe_agent = ProbeAgent(
            probe=StrategyProbe(priority_override={k: pinned for k in ("vlm", "agent", "llm")}),
            judge=None,  # bypass LLM selection
            max_analyzers=len(pinned),
        )
    else:
        probe_agent = ProbeAgent(
            judge=judge,
            max_analyzers=args.max_analyzers,
            allow_codegen=_m1_codegen,
            codegen_config=(
                CliAgentConfig(provider="antigravity", timeout_sec=120, model=args.judge_model) if _m1_codegen else None
            ),
        )

    # ── M2: StatsAnalysisAgent — selects stats tools + agy writes narrative ──
    # M2 now runs a statistical-tool layer (signal/label association, McNemar +
    # e-value, Friedman, single-rate e-value, rank corr) selected from the
    # catalog, e-BH FDR-corrects across them, and (with figure_dir) saves a
    # forest plot of effect sizes. The judge writes a conclusion grounded in
    # those verdicts. Falls back to threshold rules when cases are unlabeled.
    stats_agent = StatsAnalysisAgent(
        judge=None if args.analysis_only else judge,
        figure_dir=str(ctx.figures_dir),
        # pope (5) + relative_attention (3) + prompt_contrast (5) per-case
        # signal keys — don't silently truncate any of them.
        max_signal_tools=16,
        allow_codegen=args.allow_codegen and not args.analysis_only,
        codegen_config=(
            CliAgentConfig(provider="antigravity", timeout_sec=120, model=args.judge_model)
            if args.allow_codegen and not args.analysis_only
            else None
        ),
    )

    # ── M3: DiagnosisAgent — agy proposes hypotheses ─────────────────────────
    diagnosis_agent = None
    if not args.analysis_only:
        diagnosis_agent = DiagnosisAgent(judge=judge)

    # ── M5: HypothesisTester — agy checks protocol consistency ───────────────
    hypothesis_tester = None
    if not args.analysis_only:
        hypothesis_tester = HypothesisTester(
            judge=judge, min_effect=0.05, min_evidence_grade=args.depth,
        )

    # ── M4: SurgeryAgent — agy writes and runs the fix script ────────────────
    surgery_agent = None
    if not args.analysis_only:
        writer_cfg = ExperimentWriterConfig(
            cli_agent=CliAgentConfig(provider="antigravity", timeout_sec=120, model=args.judge_model),
            exec_fix_timeout_sec=60,
        )
        surgery_agent = SurgeryAgent(judge=judge, writer_config=writer_cfg, run_context=ctx)

    # ── Run directory + verbose logger ────────────────────────────────────────
    print(f"\nOutput directory: {run_dir.resolve()}")
    print("  run_log.jsonl   ← one JSON line per M1/M2/M3/M5 event")
    print("  artifacts/      ← per-cycle analyzer artifacts (.npy / .json)")

    # ── VLDiagnoseLoop (M1→M2→M3→M5) ─────────────────────────────────────────
    loop = VLDiagnoseLoop(
        model=model,
        protocol=protocol,
        probe_agent=probe_agent,
        stats_agent=stats_agent,
        diagnosis_agent=diagnosis_agent,
        hypothesis_tester=hypothesis_tester,
        surgery_agent=surgery_agent,   # stored but NOT called inside run()
        max_cycles=args.max_cycles,
        run_logger=ctx.logger,
        analysis_only=args.analysis_only,
    )

    print(f"\n{'='*64}")
    print(f"VLDiagnoseLoop  model={args.model}  max_cycles={args.max_cycles}")
    print(f"{'='*64}")

    report = loop.run(cases)
    ctx.write_diagnose_report(report, cases, discovery=discovery_rows)

    # ── Print verified hypotheses ─────────────────────────────────────────────
    print(f"\n{'='*64}")
    print(f"LOOP RESULT  stopped_by={report.stopped_by}  cycles={report.cycles}")
    print(f"{'='*64}")
    print(f"  total hypotheses proposed : {len(report.all_hypotheses)}")
    print(f"  verified (protocol-consistent + statistically supported): "
          f"{len(report.verified_hypotheses)}")
    for vr in report.verified_hypotheses:
        print(f"    [{vr.status.value}] {vr.hypothesis.statement}")
        print(f"           effect={vr.effect_size}  confidence={vr.confidence:.2f}"
              f"  protocol_ok={vr.is_consistent_with_protocol}")
        print(f"           {vr.verdict}")

    # ── M4: post-loop fix proposal ────────────────────────────────────────────
    if surgery_agent is not None:
        print(f"\n{'='*64}")
        print("M4  Fix proposal (post-loop)")
        print(f"{'='*64}")
        if report.verified_hypotheses:
            fix = loop.run_m4(report, cases)
            if fix is not None:
                print(f"  hypothesis : {fix.hypothesis.statement}")
                print(f"  status     : {fix.status.value}  fixed={fix.fixed}")
                ev = fix.evidence or {}
                for k, v in list(ev.items())[:6]:
                    print(f"  {k:20s}: {v}")
            else:
                print("  SurgeryAgent returned None (no verified hypotheses to act on)")
        else:
            print("  No verified hypotheses — skipping M4.")

    ctx.finalize()
    print(f"\n  Full guide -> {ctx.root / 'README.txt'}")
    print("\nDone.")


if __name__ == "__main__":
    main()
