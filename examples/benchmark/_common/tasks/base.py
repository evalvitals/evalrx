"""The task contract every dataset module fulfils + the shared manifest/case glue.

A task = one frozen dataset slice with its scorer, protocol and pinned M1 set.
``download()`` materialises ``<data-dir>/<task>/manifest.json`` (plus images/
or audio/) — the SAME manifest protocol for every modality::

    {"id", "prompt", "image": "images/..|null", "audio": "audio/..|null",
     "answers": [...], "task": <kind>, "numeric_tolerance", "choices", "metadata", ...}

so ``build_cases`` / ``score_case`` are modality-blind and a model that runs
three modalities runs them through one code path.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from ..scoring import score_output


@dataclass(frozen=True)
class Task:
    name: str                                  # registry key; also data/<name>/
    modality: str                              # vlm | llm | alm
    kind: str                                  # exact_or_numeric | multiple_choice_letter | yes_no | llm_graded | short_answer_em
    title: str                                 # human label used in protocols/logs
    download: Callable[..., dict]              # download(out_dir: Path, limit: int, seed: int) -> summary
    protocol: Callable[[str], Any]             # protocol(model_label) -> ExperimentProtocol
    pinned_m1: tuple                           # static M1 analyzer set (ProbeAgent(judge=None))
    default_limit: int = 256                   # rows frozen AND used when --limit is absent
    default_seed: int = 0
    max_new_tokens: int = 64
    output_contract: dict | None = None        # copied into protocol + case metadata when set
    short_answer: bool = True                  # terse EOS answers: mark finish_reason="stop"
    source: str = ""                           # citation / hub id for the README matrix


def manifest_path(data_dir: Path, task: Task) -> Path:
    return Path(data_dir) / task.name / "manifest.json"


def write_manifest(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def load_rows(path: Path, limit: int = 0) -> list[dict]:
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"empty or invalid manifest: {path}")
    return rows[:limit] if limit and limit > 0 else rows


def _resolve_media(base: Path, rel: str | None) -> str | None:
    if not rel:
        return None
    path = (base / rel).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"missing media file {path}")
    return str(path)


def build_cases(task: Task, manifest: Path, limit: int = 0):
    """``(CaseBatch, rows)`` — unlabeled candidates for the discovery pass."""
    from evalrx.core.case import CaseBatch, FailureCase, Inputs, Provenance, Source

    manifest = Path(manifest)
    rows = load_rows(manifest, limit)
    cases = []
    for row in rows:
        answers = row["answers"] if isinstance(row["answers"], list) else [row["answers"]]
        metadata = dict(row.get("metadata") or {})
        metadata.update({
            "dataset": task.name,
            "task": task.kind,
            "gold": row.get("gold", answers if len(answers) > 1 else answers[0]),
            "numeric_tolerance": float(row.get("numeric_tolerance", 0.0)),
        })
        if row.get("choices"):
            metadata["choices"] = list(row["choices"])
        if task.output_contract:
            metadata["output_contract"] = dict(task.output_contract)
        cases.append(FailureCase(
            id=str(row["id"]),
            inputs=Inputs(
                prompt=str(row["prompt"]),
                image=_resolve_media(manifest.parent, row.get("image")),
                audio=_resolve_media(manifest.parent, row.get("audio")),
            ),
            # exact_or_numeric keeps the alias list (ChartQA labels are lists);
            # the letter / yes-no / graded kinds expect one string.
            expected=answers if task.kind == "exact_or_numeric" else answers[0],
            metadata=metadata,
            provenance=Provenance(source=Source.DATASET, metadata={"manifest": str(manifest)}),
        ))
    return CaseBatch(cases), rows


def score_case(case: Any, output: str) -> bool:
    """The one grader for discovery, M1 probes and FixAgent — reads the case's own task."""
    meta = getattr(case, "metadata", None) or {}
    gold = meta.get("gold", getattr(case, "expected", None))
    return score_output(
        str(meta.get("task", "exact_or_numeric")), str(output if output is not None else ""), gold,
        numeric_tolerance=float(meta.get("numeric_tolerance", 0.0)),
        choices=meta.get("choices"), dataset=str(meta.get("dataset", "")),
    )


def label_case(case: Any, output: str):
    from evalrx.core.case import Label

    return Label.PASS if score_case(case, output) else Label.FAIL


def _protocol(**kwargs):
    from evalrx.eval_agent import ExperimentProtocol

    return ExperimentProtocol(**kwargs)
