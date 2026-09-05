"""examples/benchmark entry point: one (modality, model size, dataset) cell per invocation.

    python -m _common.run --modality vlm --model qwen3.5-2b --dataset chartqa \
        --judge-provider codex --judge-model gpt-5.6-terra --judge-effort medium

Data lands in ``<data-dir>/<dataset>/`` (frozen once, shared by every family of
the modality), outputs in ``<run-dir>/<model>/<dataset>[.<tag>]/``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):  # `python run.py` from inside _common/ — re-root on examples/benchmark
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "_common"  # noqa: A001

from . import tasks as T  # noqa: E402
from .models import BACKENDS, MODALITIES, SIZES, default_backend, matrix_text, resolve  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="EvalRX benchmark cell: Stage 0 -> M1..M4 -> M5 -> fix")
    p.add_argument("--modality", choices=MODALITIES, help="which dataset family / input slot the cell uses")
    p.add_argument("--model", help=f"size key ({', '.join(SIZES)}) or a registered spec key")
    p.add_argument("--dataset", default=None, help="task name (default per modality: "
                   + ", ".join(f"{m}={n}" for m, n in T.DEFAULT_TASK.items()) + ")")
    p.add_argument("--backend", choices=list(BACKENDS), default=None,
                   help="hf_local = in-process transformers (white-box + paper methods; the default "
                        "for vlm/alm); endpoint = OpenAI-compatible server at --base-url (black-box; "
                        "the default for llm); gemini = Google Gen AI API through google-genai "
                        "(forced for the gemini family)")
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
    p.add_argument("--device", default=None, help="cuda | cuda:0 | auto (default: auto for 2-GPU sizes, else cuda)")
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
    p.add_argument("--judge-provider", choices=["agy", "claude", "codex"], default="codex")
    p.add_argument("--judge-model", default="gpt-5.6-terra")
    p.add_argument("--judge-effort", default="medium")
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
    p.add_argument("--m1-selection", choices=["pinned", "judge"], default="pinned",
                   help="pinned = the task's static analyzer set; judge = catalog selection "
                        "(modality-gated on the MODEL, so avoid on multimodal specs for text tasks)")
    p.add_argument("--analyzer-max-cases", type=int, default=0, help="cap per analyzer (0 = every case)")
    p.add_argument("--m2-codegen", action=argparse.BooleanOptionalAction, default=None,
                   help="coder-written M2 statistics tools (default: on for llm, off otherwise)")
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
        "--skip-m5",
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
    args = build_parser().parse_args(argv)
    if args.list:
        print(matrix_text())
        return 0
    if args.smoke_test:
        _smoke_test()
        return 0
    if not args.modality or not args.model:
        build_parser().error("--modality and --model are required (or --list / --smoke-test)")
    dataset = args.dataset or T.DEFAULT_TASK[args.modality]
    task = T.get(dataset)
    if task.modality != args.modality:
        raise SystemExit(f"dataset {dataset!r} is a {task.modality} task, not {args.modality}")
    args.backend = args.backend or default_backend(args.modality)
    resolved = resolve(args.model, args.modality, args.backend)
    from .runner import run

    return run(args, task, resolved)


if __name__ == "__main__":
    raise SystemExit(main())
