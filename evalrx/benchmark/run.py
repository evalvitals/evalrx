"""examples/benchmark entry point: one (modality, model size, dataset) cell per invocation.

    python -m _common.run --modality vlm --model qwen3.5-2b --dataset chartqa
        # judge/agent default: claude / claude-opus-5 / high

Data lands in ``<data-dir>/<dataset>/`` (frozen once, shared by every family of
the modality), outputs in ``<run-dir>/<model>/<dataset>[.<tag>]/``.
"""

from __future__ import annotations

import argparse

from . import tasks as T  # noqa: E402
from .models import BACKENDS, MODALITIES, SIZES, default_backend, matrix_text, resolve  # noqa: E402


def build_parser(p: argparse.ArgumentParser | None = None) -> argparse.ArgumentParser:
    if p is None:
        p = argparse.ArgumentParser(prog="evalrx run", description="Evaluate and diagnose a model: Stage 0 -> M1..M4 -> M5 -> fix")
    p.add_argument("--modality", choices=MODALITIES, help="which dataset family / input slot the cell uses")
    p.add_argument("--model", help=f"size key ({', '.join(SIZES)}) or a registered spec key")
    p.add_argument("--dataset", default=None, help="task name (default per modality: "
                   + ", ".join(f"{m}={n}" for m, n in T.DEFAULT_TASK.items()) + ")")
    p.add_argument("--backend", choices=list(BACKENDS), default=None,
                   help="endpoint = OpenAI-compatible server at --base-url (vLLM in practice; "
                        "black-box, fix ladder clamped to L2; the default for every modality); "
                        "hf_local = in-process transformers (white-box + paper methods, opens "
                        "L3a/L3b); jax_local = in-process JAX through the gemma library (Gemma "
                        "E2B/E4B, white-box reads, fix ladder clamped to L2 for now); gemini = "
                        "Google Gen AI API through google-genai (forced for the gemini family)")
    p.add_argument("--concurrency", type=int, default=1,
                   help="Cases generated at once during baseline discovery. Honoured only for "
                        "--backend endpoint (a local backend shares one GPU and is not "
                        "thread-safe); a served model can only batch requests it has in hand.")
    p.add_argument("--base-url", default="http://host.docker.internal:8020/v1")
    p.add_argument("--api-key", default=None,
                   help="endpoint: bearer token (default EMPTY, what vllm serve expects); "
                        "gemini: default GEMINI_API_KEY from the environment")
    p.add_argument("--request-timeout", type=float, default=300.0,
                   help="per-request timeout in seconds for the API backends")
    p.add_argument("--request-retries", type=int, default=5,
                   help="retries with backoff on 429/5xx for --backend gemini")
    p.add_argument("--thinking-level", choices=["minimal", "low", "medium", "high"], default=None,
                   help="gemini: thinking_level for a 3.x model (default: the model's floor, i.e. "
                        "minimal, or low on 3.7-flash); on a 2.5 model it maps to a thinking_budget")
    p.add_argument("--thinking-budget", type=int, default=None,
                   help="gemini: explicit thinking_budget for a 2.5 model (0 = off; 2.5-pro floor 128)")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--run-dir", default="outputs")
    p.add_argument("--run-tag", default="", help="write to <run-dir>/<model>/<dataset>.<tag>/")
    p.add_argument("--limit", type=int, default=None, help="cases used (default: the task's; 0 = every manifest row)")
    p.add_argument("--download-limit", type=int, default=None,
                   help="rows to freeze when the manifest is missing (default: the task's; 0 = whole slice)")
    p.add_argument("--seed", type=int, default=None, help="sampling seed for a fresh manifest (default: the task's)")
    p.add_argument("--model-path", default=None,
                   help="Load the weights from this local directory instead of the spec's "
                        "hub id. For an air-gapped box, a git-cloned checkout, or pinning "
                        "an exact revision; everything else about the spec is unchanged.")
    p.add_argument("--device", default=None, help="tpu | cpu | cuda | cuda:0 | auto (default: auto for 2-GPU sizes, else cuda)")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--attn-impl", choices=["sdpa", "eager", "auto"], default=None,
                   help="default: the size's (sdpa; eager for the remote NemotronH code, which has "
                        "no SDPA dispatch). eager is also what attention capture needs")
    p.add_argument("--max-new-tokens", type=int, default=None, help="generation cap (default: the task's)")
    p.add_argument("--temperature", type=float, default=None,
                   help="sampling temperature for Stage 0 and every model call (default: 0.6 for llm "
                        "tasks — greedy Qwen3.5 loops to the cap on reasoning prompts — else 0 = greedy)")
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--fix-baseline-repeats", type=int, default=None,
                   help="baseline samples per case in the fix stage (default: 5 when sampling, 1 greedy)")
    p.add_argument("--enable-thinking", action="store_true",
                   help="turn the model's thinking mode ON for every call (default OFF on every model; "
                        "gemini: leave the API's default level instead of sending the floor)")
    p.add_argument("--judge-provider", choices=["agy", "claude", "codex", "gemini"], default="claude")
    p.add_argument("--judge-model", default=None,
                   help="default per provider: claude -> claude-opus-5, codex -> gpt-5.6-terra")
    p.add_argument("--judge-effort", default="high")
    p.add_argument("--coder-provider", choices=["gemini_cli"], default=None,
                   help="Override the coding agent independently of the judge (requires an installed, authenticated CLI)")
    p.add_argument("--coder-model", default=None, help="Model for the coding-agent override")
    p.add_argument("--fix-tier", choices=["L1", "L2", "L3a", "L3b"], default="L3b",
                   help="fix-search ceiling; L3b opens the pre-audited internals-write "
                        "primitives (VLM + hf_local only; models without a usable "
                        "executor skip the tier). The API backends (endpoint, gemini) "
                        "expose no internals: their ceiling is clamped to L2")
    p.add_argument("--allow-codegen", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument(
        "--auto-escalate",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "search L0, L1, L2, L3a, then L3b up to --fix-tier on EXPLORE; "
            "freeze one candidate for CONFIRM"
        ),
    )
    p.add_argument("--code-only", action="store_true",
                   help="restrict the fix pool to the coder-written pipeline (L2 unless its "
                        "source actually reads model internals, then L3a)")
    p.add_argument("--fix-code-file", default="",
                   help="frozen Python pipeline to validate with --code-only (skip code generation)")
    p.add_argument("--fix-candidate", default="",
                   help="pre-register and validate only this named fix candidate")
    p.add_argument("--baseline-spec", default="",
                   help="deploy a previous round's winning pipeline spec as THE baseline "
                        "(path to its result.json or a bare spec dict): Stage-0 and the fix "
                        "baseline arm run this pipeline; candidates run on the raw model and "
                        "replace it (winner-as-new-baseline for recursive rounds)")
    p.add_argument("--registered-repairs-only", action="store_true",
                   help="restrict discovery to structurally compatible registered repair methods; "
                        "the agent still selects the mechanism and no method name is pre-registered")
    p.add_argument(
        "--allow-adapted-paper-methods",
        action="store_true",
        help=(
            "admit registered paper-method executors whose runtime fidelity is explicitly "
            "architecture-adapted; reports retain the adapted fidelity designation"
        ),
    )
    p.add_argument("--explore", action=argparse.BooleanOptionalAction, default=True,
                   help="in-cycle free-form EDA between M1 and M2")
    p.add_argument("--max-cycles", type=int, default=1)
    # Legacy replay only; normal runs always default to the adaptive controller.
    p.add_argument("--diagnosis-mode", choices=["fixed", "agentic"], default="agentic",
                   help=argparse.SUPPRESS)
    p.add_argument("--max-actions", type=int, default=12,
                   help="maximum diagnostic controller decisions (default: 12)")
    p.add_argument("--held-out", action=argparse.BooleanOptionalAction, default=False,
                   help="use the task's held-out validation manifest (manifest_val.json, a "
                        "disjoint fresh sample of half the main slice): M4 verifies once there "
                        "(holdout re-probe, pinned analyzers) and the fix ladder searches and "
                        "selects there, while the frozen winner is still scored exactly once on "
                        "CONFIRM. Default off = the 50/50 explore/confirm design (M4 screens on "
                        "explore, CONFIRM reserved for the frozen repair); the validation set "
                        "is then not used at all")
    p.add_argument("--confirm-split", type=float, default=0.5,
                   help="fraction of the main batch reserved as CONFIRM (label-stratified, seed "
                        "20260818); the rest is EXPLORE where M1-M4 and the fix search run. "
                        "0.5 = the 50/50 default; 1/3 with --limit 192 = 128 explore / 64 confirm")
    p.add_argument("--confirm-split-by", choices=["case", "media"], default="case",
                   help="triage_active: what the explore/confirm draw is over. case = the loop's own "
                        "stratified split (a judge-driven run on the same batch sees the same EXPLORE); media = "
                        "whole media groups (cases on the same image, by content) go to one side, so "
                        "DISCOVERY and CONFIRM never share an image and the validation leaves nothing out")
    p.add_argument("--val-limit", type=int, default=0,
                   help="with --held-out: cases taken from the top of manifest_val.json "
                        "(0 = the whole validation manifest)")
    p.add_argument("--m1-selection", choices=["pinned", "judge"], default="pinned",
                   help="pinned = the task's static analyzer set; judge = catalog selection "
                        "(modality-gated on the MODEL, so avoid on multimodal specs for text tasks)")
    p.add_argument("--m1-policy", choices=["judge", "triage_active"], default="judge",
                   help="judge = the current diagnosis (decision judge over M1-M4, then M5/fix); "
                        "triage_active = triage controller: after Stage 0, run the triage battery "
                        "strictly on a matched EXPLORE sample, expand it and run route follow-ups "
                        "chosen by fixed rules, write the trace, cost ledger and triage report "
                        "under <run>/probing/, and stop. M2-M5 do not run; a judge is built only with "
                        "--triage-explanations judge; the controller never sees CONFIRM (only "
                        "--triage-validate reads it, after the stop)")
    p.add_argument("--triage-n-fail", type=int, default=8,
                   help="triage_active: FAIL cases in the sample (PASS controls are matched per gold answer)")
    p.add_argument("--triage-max-fail", type=int, default=32,
                   help="triage_active: the most FAIL cases the expansion rule may grow the sample to "
                        "(doubling from --triage-n-fail; a value at or below it disables expansion)")
    p.add_argument("--triage-max-actions", type=int, default=12,
                   help="triage_active: decisions after the battery (expansions + route follow-ups; the "
                        "scored scheduler's further chunks of a step it already took do not count)")
    p.add_argument("--triage-scheduler", choices=["rules", "scored", "random", "cheapest"], default="rules",
                   help="triage_active: rules = MVP-2 fixed rounds over the routes' layers; scored = MVP-3 "
                        "value per cost, U = q*[D+T+C-R]+/(c+eps) (report: triage_report.scored.*); random / "
                        "cheapest = controls over the scored candidates and stop rules that differ in the pick "
                        "only (a seeded uniform draw, or the lowest cost), never exploring")
    p.add_argument("--triage-budget-usd", type=float, default=0.0,
                   help="triage_active: USD cap on priced API models (0 = none), charged like "
                        "--triage-budget-sec")
    p.add_argument("--triage-chunk", type=int, default=8,
                   help="triage_active: cases per chunk of a reusable step (FAIL and PASS alternating; 0 = one "
                        "piece). The budget is checked before every chunk on the cost measured so far, and the "
                        "scored scheduler decides one chunk at a time")
    p.add_argument("--triage-explanations", choices=["none", "judge"], default="none",
                   help="triage_active with --triage-scheduler scored: judge = the --judge-provider/"
                        "--judge-model judge proposes competing explanations with predictions once the "
                        "sample has settled (and again when the evidence thins them), giving D (none = D "
                        "stays 0, no judge is built)")
    p.add_argument("--triage-weights", default="d=1,t=1,c=1,r=1",
                   help="triage_active with --triage-scheduler scored: the weights of D, T, C and R in U, "
                        "as k=v pairs (keys left out keep 1; an ablation sets one to 0)")
    p.add_argument("--triage-explore-share", type=float, default=0.1,
                   help="triage_active with --triage-scheduler scored: the share of the evidence spending "
                        "exploration may take (0 = none)")
    p.add_argument("--triage-explore-from", choices=["targets", "candidates"], default="targets",
                   help="triage_active with --triage-scheduler scored: what exploration draws from: targets = "
                        "inactive routes' next candidates and other routes' representatives (actions the value "
                        "never takes); candidates = the candidates other than the best (an ablation)")
    p.add_argument("--triage-validate", action=argparse.BooleanOptionalAction, default=False,
                   help="triage_active: freeze the issues the triage report supports and validate them once on "
                        "the CONFIRM cases whose media EXPLORE does not contain (<run>/validation/)")
    p.add_argument("--triage-validation-budget-sec", type=float, default=0.0,
                   help="triage_active with --triage-validate: wall-clock cap on the validation's evidence (0 = none)")
    p.add_argument("--triage-validation-source", default=None, metavar="MANIFEST",
                   help="triage_active with --triage-validate: a manifest of cases used only for DIAGNOSTIC_VALIDATION, "
                        "beside the CONFIRM cases: labelled by their own Stage 0 pass "
                        "(<run>/baseline_validation_source.json), never seen by the controller or the judge, and left "
                        "out where their media DISCOVERY contains (ChartQA: python -m evalrx.benchmark.tasks.chartqa "
                        "<dir> --exclude <pool>/manifest.json freezes the human rows the pool left)")
    p.add_argument("--triage-seed", type=int, default=0,
                   help="triage_active: rank seed of the nested sampler")
    p.add_argument("--triage-budget-sec", type=float, default=0.0,
                   help="triage_active: wall-clock cap (0 = none) on the evidence the run reads, each "
                        "observation at its first execution's cost, plus judge calls; a running "
                        "action is not interrupted, later ones are skipped")
    p.add_argument("--analyzer-max-cases", type=int, default=0, help="cap per analyzer (0 = every case)")
    p.add_argument("--m2-codegen", action=argparse.BooleanOptionalAction, default=None,
                   help="coder-written M2 statistics tools (default: on)")
    p.add_argument(
        "--fix-validation-cases",
        type=int,
        default=64,
        help=(
            "cap cases per candidate during EXPLORE selection (default: 64); "
            "the frozen winner still uses every untouched CONFIRM case"
        ),
    )
    p.add_argument("--fix-exec-timeout", type=int, default=2400)
    p.add_argument("--fix-repair-rounds", type=int, default=2,
                   help="feedback-driven coded-pipeline attempts (default: 2)")
    p.add_argument("--baseline-only", action="store_true",
                   help="download + load + Stage 0 only (no judge): the per-cell smoke check")
    p.add_argument("--skip-fix", action="store_true", help="stop after M1..M4")
    p.add_argument(
        "--skip-surgery", "--skip-m5", dest="skip_m5",
        action="store_true",
        help="skip the optional pre-fix surgery experiment; keep the full tiered fix search",
    )
    p.add_argument("--download-only", action="store_true")
    p.add_argument("--no-download", action="store_true")
    p.add_argument("--smoke-test", action="store_true", help="scorer/matrix checks, no data, no model")
    p.add_argument("--list", action="store_true", help="print the support matrix and exit")
    return p


def _smoke_test() -> None:
    from evalrx.specs import get_spec

    from .models import cells
    from .scoring import score_output

    assert score_output("exact_or_numeric", "Values: 40\nFinal answer: 42", ["42"], numeric_tolerance=0.05)
    assert score_output("exact_or_numeric", "Final answer: 6.8%", ["6.8"], numeric_tolerance=0.05)
    assert not score_output("exact_or_numeric", "106", ["100"], numeric_tolerance=0.05)
    assert score_output("multiple_choice_letter", "The answer is (B).", ["B"], choices=["A", "B", "C", "D"])
    assert not score_output("multiple_choice_letter", "A or maybe C", ["B"], choices=["A", "B", "C", "D"])
    assert score_output("yes_no", "Yes, there is.", ["Yes"]) and not score_output("yes_no", "no", ["Yes"])
    n = 0
    for modality, family, size in cells():
        spec = get_spec(size.specs[modality])
        assert spec.chat_template_kwargs.get("enable_thinking") is not True
        n += 1
    print(f"Smoke test passed: scorers OK, {n} matrix cells resolve to registered specs, thinking off.")


def main(argv: list[str] | None = None) -> int:
    return run_args(build_parser().parse_args(argv))


def run_args(args) -> int:
    if args.list:
        print(matrix_text())
        return 0
    if args.smoke_test:
        _smoke_test()
        return 0
    if not args.modality or not args.model:
        build_parser().error("--modality and --model are required (or --list / --smoke-test)")
    if args.max_actions < 1:
        raise ValueError("max-actions must be positive")
    if args.diagnosis_mode == "agentic" and args.held_out:
        raise ValueError("agentic diagnosis uses EXPLORE; --held-out is supported only in fixed mode")
    triage = getattr(args, "m1_policy", "judge") == "triage_active"
    if triage and args.held_out:
        raise ValueError("--m1-policy triage_active works on EXPLORE only; it has no --held-out role yet")
    if triage and args.triage_n_fail < 1:
        raise ValueError("--triage-n-fail must be positive")
    if triage and args.triage_max_actions < 0:
        raise ValueError("--triage-max-actions must be non-negative")
    if triage:
        triage_weights(args.triage_weights)
        if not 0.0 <= args.triage_explore_share < 1.0:
            raise ValueError("--triage-explore-share must be in [0, 1)")
    dataset = args.dataset or T.DEFAULT_TASK[args.modality]
    task = T.get(dataset)
    if task.modality != args.modality:
        raise SystemExit(f"dataset {dataset!r} is a {task.modality} task, not {args.modality}")
    args.backend = args.backend or default_backend(args.modality)
    resolved = resolve(args.model, args.modality, args.backend)
    from .runner import run

    return run(args, task, resolved)


def triage_weights(text: str) -> dict[str, float]:
    """``"d=1,t=1,c=0"`` -> ``{"d": 1.0, "t": 1.0, "c": 0.0, "r": 1.0}``: the scored scheduler's weights."""
    out = {"d": 1.0, "t": 1.0, "c": 1.0, "r": 1.0}
    for part in filter(None, (x.strip() for x in str(text).split(","))):
        key, sep, value = part.partition("=")
        key = key.strip().lower()
        if not sep or key not in out:
            raise ValueError(f"--triage-weights: {part!r} is not one of d=, t=, c=, r=")
        out[key] = float(value)
        if out[key] < 0:
            raise ValueError(f"--triage-weights: {key} must be non-negative")
    return out


if __name__ == "__main__":
    raise SystemExit(main())
