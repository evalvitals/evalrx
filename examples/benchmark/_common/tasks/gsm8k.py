"""GSM8K test split (Cobbe et al. 2021) — a seeded 450-of-1,319 sample.

HF ``openai/gsm8k`` (config ``main``, split ``test``, 1,319 grade-school math
word problems). The gold is the number after the ``####`` marker of the
reference solution (commas and dollar signs stripped); the model may reason
freely and is graded on the ``Answer:`` line by the benchmark's
``exact_or_numeric`` rule with zero tolerance, whose numeric path already
equates ``1,234`` / ``$1234`` / ``1234.0``. The 450-row slice is a
``random.Random(seed)`` sample of the test indices kept in test-file order;
``limit=0`` freezes the whole split.
"""

from __future__ import annotations

import random
from pathlib import Path

from .base import Task, _protocol, write_manifest
from .llm import PINNED_M1

HF_REPO = "openai/gsm8k"
EXPECTED_TEST_ROWS = 1_319
INSTRUCTION = ("Solve the problem. Put the final answer on its own last line "
               "as 'Answer: <answer>'.")


def _load_test_rows() -> list[dict]:
    from datasets import load_dataset

    rows = load_dataset(HF_REPO, "main", split="test")
    if rows.num_rows != EXPECTED_TEST_ROWS:
        raise SystemExit(
            f"{HF_REPO} main/test has {rows.num_rows} rows, expected {EXPECTED_TEST_ROWS}"
        )
    return list(rows)


def _gold(answer: str, source_index: int) -> tuple[str, int]:
    """``(canonical numeric gold, reasoning steps)`` from a GSM8K solution:
    the number after ``####``, commas/dollar signs stripped."""
    if "####" not in answer:
        raise SystemExit(f"row {source_index}: no '####' marker in the reference solution")
    rationale, tail = answer.rsplit("####", 1)
    canonical = tail.strip().replace(",", "").replace("$", "").strip()
    try:
        float(canonical)
    except ValueError:
        raise SystemExit(f"row {source_index}: gold {tail.strip()!r} is not numeric") from None
    return canonical, len([line for line in rationale.splitlines() if line.strip()])


def download(out_dir: Path, limit: int = 450, seed: int = 0) -> dict:
    """Freeze ``limit`` seeded-sampled rows (0 = the whole test split) in
    test-file order."""
    out_dir = Path(out_dir)
    all_rows = _load_test_rows()
    n = min(limit, len(all_rows)) if limit and limit > 0 else len(all_rows)
    if n < len(all_rows):
        picked = sorted(random.Random(seed).sample(range(len(all_rows)), n))
    else:
        picked = list(range(len(all_rows)))
    rows = []
    for source_index in picked:
        row = all_rows[source_index]
        question = str(row.get("question", "")).strip()
        if not question:
            raise SystemExit(f"row {source_index}: empty question")
        gold, n_steps = _gold(str(row.get("answer", "")), source_index)
        rows.append({
            "id": f"gsm8k-test-{source_index:04d}",
            "dataset": HF_REPO, "subset": "main/test",
            "source_index": source_index, "sample_seed": seed,
            "image": None, "audio": None,
            "prompt": f"{question}\n\n{INSTRUCTION}",
            "answers": [gold],
            "task": "exact_or_numeric", "numeric_tolerance": 0.0,
            "metadata": {"n_reasoning_steps": n_steps},
        })
    write_manifest(out_dir / "manifest.json", rows)
    return {"kept": len(rows), "test_rows": len(all_rows),
            "manifest": str(out_dir / "manifest.json")}


def protocol(model_label: str):
    return _protocol(
        description=(
            f"A text-only LLM ({model_label}) solves grade-school multi-step math word "
            "problems from the GSM8K test split (a seeded 450-of-1,319 sample). Each "
            "problem takes two to eight arithmetic steps over small quantities stated in "
            "the text; the model may reason before committing to a final 'Answer:' line. "
            "Failure cases are items whose final number does not equal the reference "
            "answer; the batch also contains, as controls, items answered correctly."
        ),
        task_domain="grade-school multi-step arithmetic word problems (GSM8K)",
        success_criteria=(
            "The number on the final answer line must equal the reference answer exactly "
            "(zero tolerance after normalising commas, currency signs and trailing "
            "zeros); the reasoning chain itself is not graded"
        ),
        failure_patterns=(
            "single arithmetic slips mid-chain that propagate to the final number; "
            "misread quantities, units or time spans; percent/fraction base errors; "
            "off-by-one on inclusive counts; answering an intermediate quantity instead "
            "of the asked one; a correct value lost to a malformed final line"
        ),
        target_modalities=frozenset({"text"}),
    )


TASK = Task(
    name="gsm8k", modality="llm", kind="exact_or_numeric", title="GSM8K/test",
    download=download, protocol=protocol,
    pinned_m1=PINNED_M1,
    default_limit=450, default_seed=0, max_new_tokens=1024,
    short_answer=False,
    source="openai/gsm8k (main, test split)",
)
