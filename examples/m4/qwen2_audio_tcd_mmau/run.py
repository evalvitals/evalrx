#!/usr/bin/env python3
"""TCD on Qwen2-Audio-7B-Instruct x MMAU test-mini -- diagnose, propose, validate.

Runs the same frozen-data / diagnosis-selection-confirmation protocol as
``examples/m4/vlm_paper_benchmark/run_hf_autofix.py``, narrowed to one paper
method (TCD) on one model/benchmark pair, since that runner's multi-paper
generality (prompt-contract tables, --only-paper-candidate, IFCD checkpoints)
does not apply here. Run ``download_mmau.py`` first.

The baseline arm calls ``generate_tcd_baseline`` (forced greedy), not the
bare ``generate()`` -- Qwen2-Audio-7B-Instruct's own generation_config
defaults to do_sample=True, and TCD's candidate is greedy by construction
(Eq. 9), so a sampled baseline paired against it would not be a valid
comparison (see generate_tcd_baseline's docstring in hf_local.py).

Usage:
    python download_mmau.py --limit 140
    python run.py --model qwen2-audio-7b-instruct --limit 120
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any, Callable

from evalvitals.core.case import CaseBatch, FailureCase, Inputs, Label
from evalvitals.eval_agent.hypothesis import Hypothesis, hypothesis_to_dict
from evalvitals.eval_agent.stages.fix_agent import FixAgent
from evalvitals.models.backends.base import RuntimeConfig
from evalvitals.models.backends.hf_local import HFLocalModel
from evalvitals.specs import get_spec

HERE = Path(__file__).parent
DATA = HERE / "data"
OUT = HERE / "outputs"


def load_records(limit: int | None = None) -> list[dict[str, Any]]:
    manifest = DATA / "mmau_test_mini.jsonl"
    if not manifest.is_file():
        raise SystemExit(f"{manifest} does not exist -- run download_mmau.py first")
    records = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line]
    return records[:limit] if limit else records


def shuffled_rows(rows: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    shuffled = list(rows)
    random.Random(seed).shuffle(shuffled)
    return shuffled


def task_prompt(row: dict[str, Any]) -> str:
    """The common prompt used by every arm -- not itself counted as a fix."""
    options = "\n".join(row["choices"])
    return (
        f"{row['instruction']}\n\n{options}\n\n"
        "Listen to the audio and reply with only the option letter (A, B, C, or D)."
    )


def parsed_choice(output: str) -> str:
    marked = re.findall(r"(?:answer|choice|final)\s*[:=-]?\s*([A-D])\b", output.upper())
    letters = re.findall(r"\b([A-D])\b", output.upper())
    return marked[-1] if marked else (letters[-1] if letters else "")


def score(row: dict[str, Any], output: str) -> bool:
    return parsed_choice(output) == str(row["expected"]).strip().upper()


def score_case(case: FailureCase, output: str) -> bool:
    return score({"expected": case.expected}, output)


def row_inputs(row: dict[str, Any]) -> Inputs:
    return Inputs(prompt=task_prompt(row), audio=str(DATA / row["audio_path"]))


def evaluate(
    rows: list[dict[str, Any]], strategy: Callable[[dict[str, Any]], str]
) -> dict[str, Any]:
    cases = []
    for row in rows:
        output = strategy(row)
        cases.append({"id": row["id"], "output": output, "correct": score(row, output)})
    correct = sum(case["correct"] for case in cases)
    return {"n": len(cases), "correct": correct, "accuracy": correct / len(cases), "cases": cases}


def diagnostic_split(
    rows: list[dict[str, Any]], baseline: dict[str, Any], diagnosis_cases: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Mirrors run_autofix.py's diagnostic_split: keep the diagnosis partition
    informative (favor observed failures) without touching confirmation rows."""
    correct_by_id = {case["id"]: case["correct"] for case in baseline["cases"]}
    failures = [row for row in rows if not correct_by_id[row["id"]]]
    passes = [row for row in rows if correct_by_id[row["id"]]]
    n_fail = min(len(failures), max(1, diagnosis_cases // 2))
    if len(failures) > 1:
        n_fail = min(n_fail, len(failures) - 1)
    diagnosis = failures[:n_fail]
    diagnosis.extend(passes[: diagnosis_cases - len(diagnosis)])
    if len(diagnosis) < diagnosis_cases:
        diagnosis.extend(failures[n_fail:diagnosis_cases])
    diagnosis_ids = {row["id"] for row in diagnosis}
    selection = [row for row in rows if row["id"] not in diagnosis_ids]
    return diagnosis, selection


def make_cases(rows: list[dict[str, Any]], baseline: dict[str, Any]) -> CaseBatch:
    baseline_by_id = {case["id"]: case for case in baseline["cases"]}
    return CaseBatch(
        FailureCase(
            id=row["id"],
            inputs=row_inputs(row),
            expected=row["expected"],
            observed=baseline_by_id[row["id"]]["output"],
            label=Label.PASS if baseline_by_id[row["id"]]["correct"] else Label.FAIL,
            metadata={**row["metadata"], "task": row["task"]},
        )
        for row in rows
    )


# TCD's own framing (Section 1: "temporal smoothing bias... transient
# acoustic cues may be under-utilized in favor of temporally smooth context
# that is better supported by language priors"), not derived from this run's
# own failures -- matches run_hf_autofix.py's precedent of a hand-tuned
# hypothesis for a paper whose failure mechanism is already pinned down by
# the source paper, rather than free-text self-diagnosis on a 7B judge.
HYPOTHESIS_STATEMENT = (
    "The model may exhibit temporal smoothing bias: transient acoustic cues "
    "(brief onsets, short-duration events, fine timing detail) are "
    "under-weighted relative to temporally smooth context and language "
    "priors, causing errors specifically on temporally-localized or "
    "fine-grained acoustic-detail questions."
)


class _JudgeModel:
    """Gives FixAgent's L1/L2/L3a judge calls a longer decode budget than the
    short answer-letter scoring calls need, without loading a second copy of
    the weights.

    Mirrors ``vlm_paper_benchmark/run_hf_autofix.py``'s ``_JudgeModel``
    (same measured problem there: an unwrapped judge inherits whatever
    ``max_new_tokens`` scoring used -- 8 here -- nowhere near enough to
    return a parseable JSON proposal list).
    """

    def __init__(self, model: HFLocalModel, max_new_tokens: int = 900) -> None:
        self._model = model
        self._max_new_tokens = max_new_tokens

    def generate(self, inputs: object, **kwargs: object) -> str:
        kwargs.setdefault("max_new_tokens", self._max_new_tokens)
        return self._model.generate(inputs, **kwargs)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="qwen2-audio-7b-instruct")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, default=120)
    parser.add_argument("--diagnosis-cases", type=int, default=8)
    parser.add_argument("--selection-cases", type=int, default=32)
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--max-tier", default="L3a")
    parser.add_argument("--seed", type=int, default=20260814)
    parser.add_argument("--output-name", default="tcd_mmau")
    parser.add_argument(
        "--allow-adapted-paper-methods",
        action="store_true",
        help="also admit TCD on non-layer-matched audio specs (reported as adapted, never native)",
    )
    parser.add_argument(
        "--judge",
        action="store_true",
        help=(
            "wire a real judge (the model under test itself, wrapped with a longer decode "
            "budget) so FixAgent's L1/L2 tiers genuinely propose candidates from the diagnosis "
            "text, instead of judge=None (which makes _ask_judge() a no-op and leaves only the "
            "pre-registered L3a paper-default candidate)"
        ),
    )
    parser.add_argument(
        "--unrestricted",
        action="store_true",
        help=(
            "drop candidate_allowlist=['tcd_temporal_blur'] so every admissible candidate "
            "(paper defaults AND, with --judge, judge-proposed ones) competes on equal footing"
        ),
    )
    args = parser.parse_args()

    split = args.diagnosis_cases + args.selection_cases
    if args.diagnosis_cases < 1 or args.selection_cases < 8 or split >= args.limit:
        raise SystemExit("need diagnosis >=1, selection >=8 and a non-empty confirmation split")

    rows = shuffled_rows(load_records(), args.seed)
    if len(rows) < args.limit:
        raise SystemExit(f"only {len(rows)} rows available, fewer than --limit {args.limit}")
    rows = rows[: args.limit]
    probe_rows, confirm_rows = rows[:split], rows[split:]

    model = HFLocalModel(
        get_spec(args.model),
        RuntimeConfig(device=args.device, dtype="bfloat16", max_new_tokens=args.max_tokens),
    )
    model.load()
    print(f"paper_method_fidelity(tcd) = {model.paper_method_fidelity('tcd')!r}")

    def baseline_generate(row: dict[str, Any]) -> str:
        return model.generate_tcd_baseline(row_inputs(row), max_new_tokens=args.max_tokens)

    baseline_probe = evaluate(probe_rows, baseline_generate)
    diagnosis_rows, selection_rows = diagnostic_split(probe_rows, baseline_probe, args.diagnosis_cases)
    baseline_diagnosis = {
        "n": len(diagnosis_rows),
        "correct": sum(
            case["correct"] for case in baseline_probe["cases"]
            if case["id"] in {row["id"] for row in diagnosis_rows}
        ),
    }
    baseline_diagnosis["accuracy"] = baseline_diagnosis["correct"] / len(diagnosis_rows)
    baseline_selection = {
        "n": len(selection_rows),
        "cases": [
            case for case in baseline_probe["cases"]
            if case["id"] in {row["id"] for row in selection_rows}
        ],
    }
    baseline_selection["correct"] = sum(case["correct"] for case in baseline_selection["cases"])
    baseline_selection["accuracy"] = baseline_selection["correct"] / len(selection_rows)
    baseline_confirm = evaluate(confirm_rows, baseline_generate)

    hypothesis = Hypothesis(
        statement=HYPOTHESIS_STATEMENT,
        target_model=args.model,
        predicted_failure_mode="temporal_smoothing_bias",
        metadata={"fix_tier": args.max_tier},
    )

    agent = FixAgent(
        judge=_JudgeModel(model) if args.judge else None,
        max_tier=args.max_tier,
        score_fn=score_case,
        max_validation_cases=0,
        allow_adapted_paper_methods=args.allow_adapted_paper_methods,
        candidate_allowlist=None if args.unrestricted else ["tcd_temporal_blur"],
        # L2 coded pipelines call a coding backend (CLI agent or judge-as-coder);
        # this run tests the declarative judge tiers (L1/L2/L3a), not free codegen
        # against the model handle -- off regardless of --judge so a 7B judge's
        # unparseable Python doesn't silently eat the round with a wasted attempt
        # (run_hf_autofix.py measured exactly this failure mode on a 7B judge).
        allow_codegen=False,
    )
    selection = agent.propose_and_validate(
        model, make_cases(selection_rows, baseline_selection), [hypothesis]
    )

    confirmation: dict[str, Any] = {"skipped": "no selection candidate"}
    if selection.best is not None:
        validated = FixAgent(score_fn=score_case).validate_candidate(
            model, make_cases(confirm_rows, baseline_confirm), selection.best.candidate
        )
        confirmation = {
            "candidate": selection.best.candidate.name,
            "fixed": validated.fixed,
            "n_fixed": validated.n_fixed,
            "n_broken": validated.n_broken,
            "effect": validated.effect,
            "e_value": validated.e_value,
            "baseline_accuracy": validated.n_baseline_correct / validated.n_pairs,
            "candidate_accuracy": validated.n_candidate_correct / validated.n_pairs,
            "summary": validated.summary,
        }

    report = {
        "paper": "tcd",
        "backend": "hf_local",
        "model": args.model,
        "paper_method_fidelity": model.paper_method_fidelity("tcd"),
        "baseline_decoding": "generate_tcd_baseline (forced greedy)",
        "max_tier": args.max_tier,
        "allow_adapted_paper_methods": args.allow_adapted_paper_methods,
        "judge_enabled": args.judge,
        "candidate_allowlist": None if args.unrestricted else ["tcd_temporal_blur"],
        "splits": {
            "diagnosis": len(diagnosis_rows),
            "selection": len(selection_rows),
            "confirmation": len(confirm_rows),
            "shuffle_seed": args.seed,
        },
        "baseline": {
            "diagnosis": baseline_diagnosis,
            "selection": baseline_selection,
            "confirmation": baseline_confirm,
        },
        "hypothesis": hypothesis_to_dict(hypothesis),
        "auto_fix": {"selection": selection.to_dict(), "confirmation": confirmation},
    }
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{args.output_name}.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(path), "confirmation": confirmation}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
