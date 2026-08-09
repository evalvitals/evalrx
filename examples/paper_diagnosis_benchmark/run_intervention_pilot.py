#!/usr/bin/env python3
"""Evaluate paper-style and EvalVitals auto-fixes on frozen benchmark slices.

This deliberately runs a *small transfer pilot*, not a full-scale reproduction.
It separates diagnosis, candidate selection, and untouched confirmation rows.
The runner also preserves backend finish reasons, allowing a telemetry-grounded
runtime repair (for example a decode-length stop) without guessing from prose.
Results and raw model output stay in the gitignored ``outputs/`` directory.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Callable

from openai import OpenAI

from evalvitals.core.capability import Capability
from evalvitals.core.case import CaseBatch, FailureCase, Inputs, Label
from evalvitals.core.model import Model, Trace
from evalvitals.eval_agent.hypothesis import Hypothesis
from evalvitals.eval_agent.stages.fix_agent import FixAgent

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "intervention_pilot"
OUTPUT_DIR = ROOT / "outputs" / "intervention_pilot"
MODEL_ID = "gpt-qwen3-vl-8b"
BASE_URL = "http://127.0.0.1:8010/v1"


class EndpointModel(Model):
    """Minimal EvalVitals ``Model`` adapter for the local OpenAI-compatible server."""

    capabilities = frozenset({Capability.GENERATE})
    modalities = frozenset({"text"})

    def __init__(self, client: OpenAI, model_id: str, *, default_max_tokens: int = 256) -> None:
        self._client = client
        self._model_id = model_id
        self._default_max_tokens = default_max_tokens

    def generate_with_metadata(self, inputs: Any, **kwargs: Any) -> tuple[str, dict[str, Any]]:
        prompt = inputs.prompt if isinstance(inputs, Inputs) else str(inputs)
        temperature = float(kwargs.pop("temperature", 0.0))
        max_tokens = int(kwargs.pop("max_tokens", self._default_max_tokens))
        response = self._client.chat.completions.create(
            model=self._model_id,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )
        choice = response.choices[0]
        return choice.message.content or "", {
            "finish_reason": choice.finish_reason or "unknown",
            "generation_config": {"max_tokens": max_tokens, "temperature": temperature},
        }

    def generate(self, inputs: Any, **kwargs: Any) -> str:
        return self.generate_with_metadata(inputs, **kwargs)[0]

    def forward(self, inputs: Any, capture: set[Capability], spec: Any = None) -> Trace:
        raise NotImplementedError("the intervention pilot is generation-only")


def load_rows(name: str, limit: int) -> list[dict[str, str]]:
    path = DATA_DIR / name
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if len(rows) < limit:
        raise ValueError(f"{path} has {len(rows)} rows, fewer than requested {limit}")
    return rows[:limit]


def answer_key(row: dict[str, str], output: str) -> str:
    if row["dataset"] == "truthful_qa":
        text = output.strip().upper()
        explicit = re.findall(r"(?:FINAL\s+ANSWER|ANSWER|CHOICE)\s*[:=-]?\s*([A-D])\b", text)
        if explicit:
            return explicit[-1]
        letters = re.findall(r"\b([A-D])\b", text)
        return letters[-1] if letters else ""
    hashes = re.findall(r"####\s*(-?[0-9][0-9,]*(?:\.[0-9]+)?)", output)
    if hashes:
        return hashes[-1].replace(",", "")
    numbers = re.findall(r"-?[0-9][0-9,]*(?:\.[0-9]+)?", output.replace(",", ""))
    return numbers[-1] if numbers else ""


def score(row: dict[str, str], output: str) -> bool:
    return answer_key(row, output) == str(row["answer"])


def most_common(outputs: list[str], row: dict[str, str]) -> str:
    keys = [answer_key(row, output) for output in outputs]
    valid = [key for key in keys if key]
    if not valid:
        return outputs[0]
    winner = Counter(valid).most_common(1)[0][0]
    return next(output for output, key in zip(outputs, keys) if key == winner)


def paper_output(model: EndpointModel, paper_id: str, row: dict[str, str]) -> str:
    question = row["question"]
    if paper_id == "zero_shot_cot":
        return model.generate(f"{question}\n\nLet's think step by step. End with the final answer.")
    if paper_id == "self_consistency":
        samples = [
            model.generate(
                f"{question}\n\nSolve carefully step by step. End with the final answer.",
                temperature=0.7,
            )
            for _ in range(5)
        ]
        return most_common(samples, row)
    if paper_id == "least_to_most":
        decomposition = model.generate(
            "Break this problem into the smallest useful subproblems. "
            f"Do not answer yet.\n\n{question}"
        )
        return model.generate(
            f"Solve the original problem using this decomposition. End with the final answer.\n\n"
            f"Problem:\n{question}\n\nDecomposition:\n{decomposition}"
        )
    if paper_id == "self_refine":
        draft = model.generate(
            f"Solve the problem carefully. End with the final answer.\n\n{question}"
        )
        feedback = model.generate(
            "Check this attempted answer for arithmetic, reasoning, or "
            "instruction-following errors. "
            f"Give concise correction advice only.\n\nProblem:\n{question}\n\nAttempt:\n{draft}"
        )
        return model.generate(
            f"Produce a corrected final answer using the feedback. End with the final answer.\n\n"
            f"Problem:\n{question}\n\nAttempt:\n{draft}\n\nFeedback:\n{feedback}"
        )
    if paper_id == "cove":
        draft = model.generate(f"Answer the question.\n\n{question}")
        checks = model.generate(
            "List short, independent checks needed to verify this answer.\n\n"
            f"Question:\n{question}\n\nDraft:\n{draft}"
        )
        return model.generate(
            f"Answer the original question after applying the independent verification checks. "
            f"For multiple choice, reply with only the letter.\n\nQuestion:\n{question}\n\n"
            f"Draft:\n{draft}\n\nChecks:\n{checks}"
        )
    raise ValueError(f"unknown paper id {paper_id!r}")


def evaluate(
    rows: list[dict[str, str]], strategy: Callable[[dict[str, str]], str | tuple[str, dict[str, Any]]]
) -> dict[str, Any]:
    outcomes = []
    for row in rows:
        result = strategy(row)
        output, telemetry = result if isinstance(result, tuple) else (result, {})
        outcomes.append({
            "id": row["id"],
            "correct": score(row, output),
            "output": output,
            **telemetry,
        })
    correct = sum(item["correct"] for item in outcomes)
    return {
        "n": len(outcomes),
        "correct": correct,
        "accuracy": correct / len(outcomes),
        "cases": outcomes,
    }


def diagnose_failures(
    model: EndpointModel, baseline: dict[str, Any], rows: list[dict[str, str]]
) -> str:
    failures = []
    by_id = {row["id"]: row for row in rows}
    for item in baseline["cases"]:
        if not item["correct"]:
            row = by_id[item["id"]]
            failures.append({
                "question": row["question"][:500],
                "output": item["output"][:500],
                "finish_reason": item.get("finish_reason"),
            })
    if not failures:
        return (
            "No baseline failures were observed in this frozen slice; auto-fix is not applicable."
        )
    evidence = json.dumps(failures[:8], ensure_ascii=False)
    prompt = (
        "Diagnose the shared failure mechanism in these incorrect model responses. "
        "State one narrow, testable hypothesis without proposing a fix and without claiming "
        "facts not in the examples.\n\n" + evidence
    )
    return model.generate(prompt).strip()


def runtime_hypothesis(baseline: dict[str, Any]) -> Hypothesis | None:
    """Create an L0 hypothesis only from explicit, backend-provided telemetry."""
    length_failures = [
        item for item in baseline["cases"]
        if not item["correct"] and item.get("finish_reason") == "length"
    ]
    if not length_failures:
        return None
    caps = [
        item.get("generation_config", {}).get("max_tokens")
        for item in length_failures
        if item.get("generation_config", {}).get("max_tokens")
    ]
    if not caps:
        return None
    return Hypothesis(
        statement=(
            f"{len(length_failures)} baseline failures ended with finish_reason=length "
            f"at a configured max_tokens cap; test a bounded decode-budget increase."
        ),
        target_model=MODEL_ID,
        predicted_failure_mode="generation truncation at configured decode budget",
        metadata={"fix_tier": "L0", "evidence": "backend finish_reason telemetry"},
    )


def as_cases(
    rows: list[dict[str, str]], baseline: dict[str, Any], *, repair_max_tokens: int | None = None
) -> CaseBatch:
    """Adapt one frozen split plus its baseline telemetry to EvalVitals cases."""
    baseline_by_id = {item["id"]: item for item in baseline["cases"]}
    return CaseBatch(
        FailureCase(
            id=row["id"],
            inputs=Inputs(prompt=row["question"]),
            expected=row["answer"],
            observed=baseline_by_id[row["id"]]["output"],
            label=(
                Label.PASS if score(row, baseline_by_id[row["id"]]["output"]) else Label.FAIL
            ),
            metadata={
                "dataset": row["dataset"],
                "finish_reason": baseline_item.get("finish_reason"),
                "generation_config": baseline_item.get("generation_config", {}),
                "generation_policy": (
                    {"max_tokens_cap": repair_max_tokens} if repair_max_tokens else {}
                ),
                "output_key_pattern": (
                    r"####\s*(-?[0-9][0-9,]*(?:\.[0-9]+)?)"
                    if row["dataset"] == "openai/gsm8k"
                    else r"(?:FINAL\s+ANSWER|ANSWER|CHOICE)\s*[:=-]?\s*([A-D])\b"
                ),
            },
        )
        for row in rows
        for baseline_item in [baseline_by_id[row["id"]]]
    )
def score_case(case: FailureCase, output: str) -> bool:
    return (
        answer_key({"dataset": case.metadata["dataset"], "answer": case.expected}, output)
        == case.expected
    )


def auto_fix(
    model: EndpointModel,
    rows: list[dict[str, str]],
    baseline: dict[str, Any],
    hypothesis_text: str,
    *,
    allow_codegen: bool,
    max_judge_candidates: int,
    repair_max_tokens: int | None,
) -> tuple[Any, CaseBatch]:
    cases = as_cases(rows, baseline, repair_max_tokens=repair_max_tokens)
    hypothesis = runtime_hypothesis(baseline) or Hypothesis(
        statement=hypothesis_text,
        target_model=MODEL_ID,
        predicted_failure_mode="generation failures in the frozen benchmark slice",
    )

    max_tier = "L0" if hypothesis.metadata.get("fix_tier") == "L0" else "L2"
    agent = FixAgent(
        judge=model,
        max_tier=max_tier,
        score_fn=score_case,
        max_validation_cases=len(cases),
        max_repair_rounds=1,
        exec_timeout_sec=900,
        allow_codegen=allow_codegen,
        max_judge_candidates=max_judge_candidates,
    )
    return agent.propose_and_validate(model, cases, [hypothesis]), cases


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paper",
        choices=["zero_shot_cot", "self_consistency", "least_to_most", "self_refine", "cove"],
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=120,
        help="Total frozen cases, partitioned into diagnosis/selection/confirmation.",
    )
    parser.add_argument("--diagnosis-cases", type=int, default=24)
    parser.add_argument("--selection-cases", type=int, default=36)
    parser.add_argument(
        "--baseline-max-tokens",
        type=int,
        default=256,
        help="Recorded baseline decode cap; auto-fix may raise it only after length telemetry.",
    )
    parser.add_argument(
        "--repair-max-tokens",
        type=int,
        default=512,
        help="Maximum completion budget approved by the deployment policy for an L0 repair.",
    )
    parser.add_argument("--out", type=Path, default=OUTPUT_DIR)
    parser.add_argument(
        "--allow-codegen",
        action="store_true",
        help="Also evaluate a free-form L2 code candidate (slow; sandbox timeout is 900 seconds).",
    )
    parser.add_argument(
        "--auto-candidates",
        type=int,
        default=1,
        help="Candidates per auto-fix tier (default: 1 for a bounded screening run).",
    )
    args = parser.parse_args()

    source = "truthfulqa_mc1.jsonl" if args.paper == "cove" else "gsm8k.jsonl"
    rows = load_rows(source, args.limit)
    split_end = args.diagnosis_cases + args.selection_cases
    if args.diagnosis_cases < 1 or args.selection_cases < 8 or split_end >= len(rows):
        raise SystemExit(
            "need >=1 diagnosis row, >=8 selection rows, and at least one untouched "
            "confirmation row; increase --limit or reduce split sizes"
        )
    diagnosis_rows = rows[:args.diagnosis_cases]
    selection_rows = rows[args.diagnosis_cases:split_end]
    confirm_rows = rows[split_end:]
    model = EndpointModel(
        OpenAI(base_url=BASE_URL, api_key="EMPTY", timeout=900),
        MODEL_ID,
        default_max_tokens=args.baseline_max_tokens,
    )

    baseline_diagnosis = evaluate(
        diagnosis_rows, lambda row: model.generate_with_metadata(row["question"])
    )
    diagnosis = diagnose_failures(model, baseline_diagnosis, diagnosis_rows)
    baseline_selection = evaluate(
        selection_rows, lambda row: model.generate_with_metadata(row["question"])
    )
    paper = evaluate(confirm_rows, lambda row: paper_output(model, args.paper, row))
    baseline_confirm = evaluate(
        confirm_rows, lambda row: model.generate_with_metadata(row["question"])
    )

    if baseline_selection["correct"] < baseline_selection["n"]:
        selection_outcome, _ = auto_fix(
            model,
            selection_rows,
            baseline_selection,
            diagnosis,
            allow_codegen=args.allow_codegen,
            max_judge_candidates=args.auto_candidates,
            repair_max_tokens=args.repair_max_tokens,
        )
        automated: dict[str, Any] = {"selection": selection_outcome.to_dict()}
        if selection_outcome.best is not None:
            confirm_agent = FixAgent(score_fn=score_case, max_validation_cases=0)
            confirmation = confirm_agent.validate_candidate(
                model,
                as_cases(
                    confirm_rows,
                    baseline_confirm,
                    repair_max_tokens=args.repair_max_tokens,
                ),
                selection_outcome.best.candidate,
            )
            automated["confirmation"] = {
                "candidate": selection_outcome.best.candidate.name,
                "tier": selection_outcome.best.candidate.tier.label,
                "payload": selection_outcome.best.candidate.payload,
                "n_pairs": confirmation.n_pairs,
                "n_fixed": confirmation.n_fixed,
                "n_broken": confirmation.n_broken,
                "effect": confirmation.effect,
                "e_value": confirmation.e_value,
                "fixed": confirmation.fixed,
                "verdict": confirmation.verdict,
                "summary": confirmation.summary,
                "baseline_accuracy": baseline_confirm["accuracy"],
                "candidate_accuracy": (
                    (baseline_confirm["correct"] + confirmation.n_fixed - confirmation.n_broken)
                    / baseline_confirm["n"]
                    if confirmation.n_pairs == baseline_confirm["n"] else None
                ),
                "accuracy_delta": (
                    (confirmation.n_fixed - confirmation.n_broken) / baseline_confirm["n"]
                    if confirmation.n_pairs == baseline_confirm["n"] else None
                ),
            }
        else:
            automated["confirmation"] = {"skipped": "no candidate selected on selection split"}
    else:
        automated = {
            "selection": {"fixed": False, "skipped": "no baseline failures"},
            "confirmation": {"skipped": "no candidate selected"},
        }
    report = {
        "paper_id": args.paper,
        "model": MODEL_ID,
        "n": len(rows),
        "splits": {
            "diagnosis": [row["id"] for row in diagnosis_rows],
            "selection": [row["id"] for row in selection_rows],
            "confirmation": [row["id"] for row in confirm_rows],
        },
        "baseline": {
            "diagnosis": baseline_diagnosis,
            "selection": baseline_selection,
            "confirmation": baseline_confirm,
        },
        "paper_method": paper,
        "auto_diagnosis": diagnosis,
        "auto_fix": automated,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    out_path = args.out / f"{args.paper}.json"
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        json.dumps(
            {
                "paper": args.paper,
                "baseline_confirmation": baseline_confirm["accuracy"],
                "paper_method": paper["accuracy"],
                "auto_fixed_confirmation": automated["confirmation"].get("fixed", False),
                "output": str(out_path),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
