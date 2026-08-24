"""examples/benchmark entry point: one (modality, model size, dataset) cell per invocation.

    python -m _common.run --modality vlm --model qwen3.5-2b --dataset chartqa \
        --judge-provider claude --judge-model claude-opus-5 --judge-effort high

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
from .models import MODALITIES, SIZES, matrix_text, resolve  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="EvalVitals benchmark cell: Stage 0 -> M1..M5 -> M4 -> fix")
    p.add_argument("--modality", choices=MODALITIES, help="which dataset family / input slot the cell uses")
    p.add_argument("--model", help=f"size key ({', '.join(SIZES)}) or a registered spec key")
    p.add_argument("--dataset", default=None, help="task name (default per modality: "
                   + ", ".join(f"{m}={n}" for m, n in T.DEFAULT_TASK.items()) + ")")
    p.add_argument("--backend", choices=["hf_local", "endpoint"], default="hf_local",
                   help="hf_local = in-process transformers (default; white-box + paper methods); "
                        "endpoint = OpenAI-compatible server (black-box)")
    p.add_argument("--concurrency", type=int, default=1,
                   help="Cases generated at once during baseline discovery. Honoured only for "
                        "--backend endpoint (a local backend shares one GPU and is not "
                        "thread-safe); a served model can only batch requests it has in hand.")
    p.add_argument("--base-url", default="http://host.docker.internal:8020/v1")
    p.add_argument("--api-key", default="EMPTY")
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
                   help="turn the model's thinking mode ON for every call (default OFF on every model)")
    p.add_argument("--judge-provider", choices=["agy", "claude", "codex"], default="agy")
    p.add_argument("--judge-model", default="")
    p.add_argument("--judge-effort", default="high")
    p.add_argument("--fix-tier", choices=["L1", "L2", "L3a"], default="L3a")
    p.add_argument("--allow-codegen", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--auto-escalate", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--code-only", action="store_true",
                   help="restrict the fix pool to the coder-written L2 pipeline")
    p.add_argument("--explore", action=argparse.BooleanOptionalAction, default=True,
                   help="in-cycle free-form EDA between M1 and M2")
    p.add_argument("--max-cycles", type=int, default=1)
    p.add_argument("--m1-selection", choices=["pinned", "judge"], default="pinned",
                   help="pinned = the task's static analyzer set; judge = catalog selection "
                        "(modality-gated on the MODEL, so avoid on multimodal specs for text tasks)")
    p.add_argument("--analyzer-max-cases", type=int, default=0, help="cap per analyzer (0 = every case)")
    p.add_argument("--m2-codegen", action=argparse.BooleanOptionalAction, default=None,
                   help="coder-written M2 statistics tools (default: on for llm, off otherwise)")
    p.add_argument("--fix-validation-cases", type=int, default=256)
    p.add_argument("--fix-exec-timeout", type=int, default=2400)
    p.add_argument("--baseline-only", action="store_true",
                   help="download + load + Stage 0 only (no judge): the per-cell smoke check")
    p.add_argument("--skip-fix", action="store_true", help="stop after M1..M5")
    p.add_argument("--download-only", action="store_true")
    p.add_argument("--no-download", action="store_true")
    p.add_argument("--smoke-test", action="store_true", help="scorer/matrix checks, no data, no model")
    p.add_argument("--list", action="store_true", help="print the support matrix and exit")
    return p


def _smoke_test() -> None:
    from .scoring import score_output
    from .models import cells
    from evalvitals.specs import get_spec

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
    resolved = resolve(args.model, args.modality, args.backend)
    from .runner import run

    return run(args, task, resolved)


if __name__ == "__main__":
    raise SystemExit(main())
