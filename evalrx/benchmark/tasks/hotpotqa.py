"""HotpotQA, the GEPA split (arXiv:2507.19457) — the 300-question test pool.

GEPA's appendix fixes a 150/300/300 train/val/test split of HF
``hotpotqa/hotpot_qa`` (config ``fullwiki``, split ``train``, 90,447 rows):
slice the split IN ORDER into a test pool (first 40 %), a val pool (middle
40 %) and a train pool (last 20 %), then sample each pool with a fresh
``random.Random(); rng.seed(1)`` (``gepa-artifact`` ``benchmarks/benchmark.py``).
``download`` re-runs that recipe bit-for-bit — sampling indices instead of row
objects, which picks the identical positions — and by default freezes the TEST
pool's 300 questions (the same items behind GEPA Table 1, Qwen3-8B 42.33 EM);
``split`` accepts a single name or a ``+``-joined list. ``val_limit`` > 0 also
freezes ``manifest_val.json`` from the TRAIN pool's 150 questions — a disjoint
pool of the same published recipe — the optional held-out validation set.

The protocol here is NOT GEPA's: their program answers via two-hop ColBERTv2
retrieval over a hosted Wikipedia-2017 index that is not reproducible locally.
We hand the model HotpotQA's own 10 candidate paragraphs (2 gold + 8
distractors, the dataset's distractor setting, formatted exactly like
``examples/dataset_selection`` ``band_locate._hotpot_context``) so the task
stays multi-hop READING over the same items, and grade with ``short_answer_em`` —
the same SQuAD normalisation as GEPA's ``dspy.evaluate.answer_exact_match``. Published GEPA numbers are
therefore a distribution anchor for these items, not a per-item comparison.
Everything is drawn from the 2018 TRAIN split: a contamination risk worth
naming in any writeup.
"""

from __future__ import annotations

import random
from pathlib import Path

from .base import Task, _protocol, write_manifest
from .llm import PINNED_M1

HF_REPO = "hotpotqa/hotpot_qa"
EXPECTED_TRAIN_ROWS = 90_447
SIZES = {"train": 150, "val": 300, "test": 300}
GEPA_SEED = 1
_CONTEXT_CHAR_CAP = 14_000
INSTRUCTION = ("Solve the problem. Put the final answer on its own last line "
               "as 'Answer: <answer>'.")


def _load_train_rows() -> list[dict]:
    """``fullwiki/train`` in the hub's canonical row order (the split logic
    depends on nothing but this order)."""
    from datasets import load_dataset

    rows = load_dataset(HF_REPO, "fullwiki", split="train")
    if rows.num_rows != EXPECTED_TRAIN_ROWS:
        raise SystemExit(
            f"{HF_REPO} fullwiki/train has {rows.num_rows} rows, expected "
            f"{EXPECTED_TRAIN_ROWS}; the GEPA split is defined on that revision "
            "and cannot be reconstructed from a different one"
        )
    return list(rows)


def _pool_bounds(split: str, n: int) -> tuple[int, int]:
    """GEPA ``create_splits``: test = first 40 %, val = middle 40 %, train = last 20 %."""
    cuts = {"test": (0, int(0.4 * n)), "val": (int(0.4 * n), int(0.8 * n)),
            "train": (int(0.8 * n), n)}
    return cuts[split]


def _gepa_indices(pool_len: int, size: int, seed: int) -> list[int]:
    """GEPA ``trim_dataset`` on index space: ``rng.sample`` picks positions from
    ``(len, k, seed)`` alone, so sampling ``range(pool_len)`` yields exactly the
    rows that sampling the row list would, in the same order; a pool smaller
    than ``size`` is kept whole in pool order (``trim_dataset`` returns it as is)."""
    if size >= pool_len:
        return list(range(pool_len))
    return random.Random(seed).sample(range(pool_len), size)


def _context(row: dict) -> str:
    """The 10 candidate paragraphs (2 gold + 8 distractors). ``context`` is a
    dict of PARALLEL lists (``title`` / ``sentences``), so zipping is the only
    correct read; same formatting + cap as ``band_locate._hotpot_context``."""
    ctx = row.get("context")
    if not isinstance(ctx, dict):
        return ""
    paras = [f"{title}: {''.join(sents).strip()}"
             for title, sents in zip(ctx.get("title") or [], ctx.get("sentences") or [])
             if str(title).strip()]
    return "\n\n".join(paras)[:_CONTEXT_CHAR_CAP]


def _pool_rows(all_rows: list[dict], part: str, seed: int) -> tuple[list[dict], list[int]]:
    """One GEPA pool's frozen rows, in GEPA sample order, + its pool bounds."""
    start, end = _pool_bounds(part, len(all_rows))
    picked = _gepa_indices(end - start, SIZES[part], seed)
    rows = []
    for rank, pool_index in enumerate(picked):
        source_index = start + pool_index
        row = all_rows[source_index]
        context = _context(row)
        question = str(row.get("question", "")).strip()
        answer = str(row.get("answer", "")).strip()
        if not (context and question and answer):
            raise SystemExit(f"row {source_index} is not gradable (empty context/question/answer)")
        supporting = row.get("supporting_facts") or {}
        rows.append({
            "id": f"hotpotqa-gepa-{part}-{source_index:05d}",
            "dataset": HF_REPO, "subset": f"fullwiki/train (GEPA {part} pool)",
            "source_index": source_index, "sample_rank": rank, "sample_seed": seed,
            "image": None, "audio": None,
            "prompt": f"Context:\n{context}\n\nQuestion: {question}\n\n{INSTRUCTION}",
            "answers": [answer],
            "task": "short_answer_em", "numeric_tolerance": 0.0,
            "metadata": {"hotpot_id": str(row.get("id", "")), "type": str(row.get("type", "")),
                         "level": str(row.get("level", "")), "gepa_split": part,
                         "supporting_titles": sorted(set(supporting.get("title") or []))},
        })
    return rows, [start, end]


def download(out_dir: Path, limit: int = 300, seed: int = GEPA_SEED,
             split: str = "test", val_limit: int = 0) -> dict:
    """Freeze one or more GEPA splits, each in GEPA sample order.

    ``split`` is a single name or a ``+``-joined list (default ``test``: the
    300 test questions behind GEPA Table 1). ``limit`` > 0 keeps the first
    ``limit`` rows of the concatenated order (a deterministic prefix); 0 keeps
    everything. ``seed`` = 1 is the published split — anything else is a
    same-recipe robustness variant, not GEPA's set. ``val_limit`` > 0 also
    freezes ``manifest_val.json`` from the GEPA TRAIN pool (150 questions,
    disjoint from the test pool by construction).
    """
    parts = [p.strip() for p in split.split("+") if p.strip()]
    for part in parts:
        if part not in SIZES:
            raise ValueError(f"unknown GEPA split {part!r}; expected one of {tuple(SIZES)}")
    out_dir = Path(out_dir)
    all_rows = _load_train_rows()
    rows, pools = [], {}
    for part in parts:
        part_rows, pools[part] = _pool_rows(all_rows, part, seed)
        rows.extend(part_rows)
    if limit and limit > 0:
        rows = rows[:limit]
    write_manifest(out_dir / "manifest.json", rows)
    summary = {"kept": len(rows), "split": split, "train_rows": len(all_rows),
               "pool": pools, "manifest": str(out_dir / "manifest.json")}
    if val_limit and val_limit > 0:
        if "train" in parts:
            raise ValueError("val_limit uses the GEPA train pool, which the main "
                             "manifest already includes — the sets would overlap")
        val_rows, val_pool = _pool_rows(all_rows, "train", seed)
        val_rows = val_rows[:val_limit]
        write_manifest(out_dir / "manifest_val.json", val_rows)
        summary.update(kept_val=len(val_rows), val_pool=val_pool,
                       manifest_val=str(out_dir / "manifest_val.json"))
    return summary


def protocol(model_label: str):
    return _protocol(
        description=(
            f"A text-only LLM ({model_label}) answers the 300 questions of the GEPA test "
            "split of HotpotQA (fullwiki/train, seed-1 sample) — the same items as GEPA's "
            "Table 1. Each prompt carries the dataset's own 10 "
            "candidate Wikipedia "
            "paragraphs (2 gold + 8 distractors); answering needs a two-hop combination "
            "of two of them (bridge questions) or a comparison of two entities "
            "(comparison questions, often with a yes/no answer). Failure cases are items "
            "answered incorrectly under normalised exact match; the batch also contains, "
            "as controls, items the model answered correctly."
        ),
        task_domain="multi-hop reading comprehension (HotpotQA, distractor setting)",
        success_criteria=(
            "The final answer must equal the gold answer after SQuAD-style normalisation "
            "(case, punctuation, articles); no partial credit, no LLM judge. Verbose "
            "answers that contain the gold span plus extra words score 0."
        ),
        failure_patterns=(
            "failed second hops (the bridge entity is found but its property is not); "
            "answers copied from a distractor paragraph about a similar entity; "
            "unsupported recall that ignores the provided context; comparison questions "
            "flipped to the wrong side; correct content lost to the exact-match surface "
            "(extra words, wrong granularity such as a full date for a year)"
        ),
        target_modalities=frozenset({"text"}),
        metadata={
            "gepa_reference": ("GEPA (arXiv:2507.19457) reports Qwen3-8B 42.33 EM on these "
                               "same 300 items under its two-hop ColBERTv2 retrieval "
                               "protocol; with the 10 in-prompt paragraphs used here that "
                               "number anchors the distribution, not per-item difficulty"),
            "contamination_note": "all items come from HotpotQA's public 2018 TRAIN split",
        },
    )


TASK = Task(
    name="hotpotqa_gepa", modality="llm", kind="short_answer_em", title="HotpotQA/GEPA-test",
    download=download, protocol=protocol,
    pinned_m1=PINNED_M1,
    default_limit=300, val_limit=150, default_seed=GEPA_SEED, max_new_tokens=512,
    short_answer=False,
    source="hotpotqa/hotpot_qa fullwiki/train, GEPA test split seed 1 (val: GEPA train pool), arXiv:2507.19457",
)
