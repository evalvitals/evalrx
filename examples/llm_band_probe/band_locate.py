"""Locate the usable difficulty band for a model across candidate datasets.

The 2026-08 survey's headline finding was that a benchmark is only usable for
diagnosis when the model under test lands in roughly the 30-70% accuracy band:
a saturated set has no FAIL mass and a floored one has no PASS mass, so M2 has
nothing to contrast either way.  Which band a dataset falls in is a property of
the *pair*, not of the dataset, so it has to be measured against the actual
model rather than read off a leaderboard.

This script samples each candidate config, runs it through an OpenAI-compatible
endpoint, grades with the same numeric-aware grader the probes use, and reports
accuracy with a Wilson interval so "saturated" and "floor" are claims with error
bars rather than point estimates.

    python band_locate.py --out results.json --n 60 --concurrency 16
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import requests

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from evalvitals.analyzers.reasoning._text import (  # noqa: E402
    answer_equal,
    extract_answer,
    normalize_answer,
)

ROWS_API = "https://datasets-server.huggingface.co/rows"
BASE_URL = os.environ.get("BAND_BASE_URL", "http://127.0.0.1:8020/v1")
MODEL_ID = os.environ.get("BAND_MODEL_ID", "qwen3.5-9b")

_MC_LETTERS = "ABCDEFGHIJ"


# ----------------------------------------------------------------------
# Candidate specs
# ----------------------------------------------------------------------
@dataclass
class Spec:
    """One candidate dataset slice to measure."""

    name: str
    chapter: str
    dataset: str
    config: str = "default"
    split: str = "test"
    question_field: str = "question"
    answer_field: str = "answer"
    # row -> (prompt, gold) ; None drops the row
    adapter: Optional[Callable[[dict], Optional[tuple]]] = None
    instruction: str = (
        "Solve the problem. Put the final answer on its own last line as "
        "'Answer: <answer>'."
    )
    # Gold shapes differ enough that a single grader would silently under-report
    # (list-wrapped LaTeX, answer aliases, full grids) -> allow an override.
    grader: Optional[Callable[[str, Any], bool]] = None
    #: Feed the grader the RAW generation instead of the extracted answer.
    #: extract_answer is structurally single-line (the tag regex does not span
    #: newlines), so a multi-line gold — a puzzle grid — graded on the extracted
    #: span can only ever see its first row and scores 0 for every model.
    grades_raw_output: bool = False
    max_tokens: int = 8192
    note: str = ""
    meta: dict = field(default_factory=dict)


def _text(row: dict, *keys: str) -> str:
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            return str(value)
    return ""


def _mc_prompt(question: str, options: list[str]) -> str:
    lines = [question, ""]
    lines += [f"{_MC_LETTERS[i]}. {opt}" for i, opt in enumerate(options)]
    return "\n".join(lines)


# -- per-dataset adapters (field names verified against /first-rows) -----
def _adapter_plain(q: str, a: str) -> Callable[[dict], Optional[tuple]]:
    def _fn(row: dict) -> Optional[tuple]:
        question, gold = _text(row, q), _text(row, a)
        return (question, gold) if question and gold else None

    return _fn


def _adapter_imo(row: dict) -> Optional[tuple]:
    # IMO-AnswerBench ships field names WITH SPACES; only integer answers are
    # cheaply gradable, the rest need a judge.
    question, gold = _text(row, "question", "Question"), _text(row, "answer", "Answer")
    if not question or not gold or not re.fullmatch(r"-?\d+", gold.strip()):
        return None
    return question, gold.strip()


def _adapter_gsm_symbolic(row: dict) -> Optional[tuple]:
    question, gold = _text(row, "question"), _text(row, "answer")
    if not question or not gold:
        return None
    # GSM8K-style gold: "reasoning #### 42"
    return question, gold.split("####")[-1].strip()


def _adapter_lcb_execution(row: dict) -> Optional[tuple]:
    code, inp, out = _text(row, "code"), _text(row, "input"), _text(row, "output")
    if not code or not out:
        return None
    prompt = (
        "You are given a Python function and a call to it. Predict the return "
        f"value exactly.\n\n{code}\n\nCall:\n{inp}\n\n"
        "Give only the returned value on the last line as 'Answer: <value>'."
    )
    return prompt, out.strip()


def _adapter_cruxeval(row: dict) -> Optional[tuple]:
    code, inp, out = _text(row, "code"), _text(row, "input"), _text(row, "output")
    if not code or not out:
        return None
    prompt = (
        "Predict the output of this Python function for the given input.\n\n"
        f"{code}\n\nInput: {inp}\n\n"
        "Give only the output value on the last line as 'Answer: <value>'."
    )
    return prompt, out.strip()


def _adapter_mc(question_keys: tuple, options_key: str, answer_key: str):
    def _fn(row: dict) -> Optional[tuple]:
        question = _text(row, *question_keys)
        options = row.get(options_key)
        gold = _text(row, answer_key)
        if not question or not isinstance(options, (list, tuple)) or not gold:
            return None
        if gold in _MC_LETTERS:
            letter = gold
        elif gold in options:
            letter = _MC_LETTERS[list(options).index(gold)]
        else:
            return None
        return _mc_prompt(question, list(options)), letter

    return _fn


def _adapter_musr(row: dict) -> Optional[tuple]:
    # MuSR ships `choices` as the STRING repr of a list, not a list
    import ast

    narrative, question = _text(row, "narrative"), _text(row, "question")
    raw, gold = row.get("choices"), _text(row, "answer_choice")
    if isinstance(raw, str):
        try:
            raw = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            return None
    if not narrative or not isinstance(raw, (list, tuple)) or gold not in raw:
        return None
    letter = _MC_LETTERS[list(raw).index(gold)]
    return _mc_prompt(f"{narrative}\n\n{question}", list(raw)), letter


def _adapter_olympiadbench(row: dict) -> Optional[tuple]:
    # final_answer is a LIST of LaTeX strings; take the first and strip the $
    question = _text(row, "question")
    gold = row.get("final_answer")
    if isinstance(gold, (list, tuple)):
        gold = gold[0] if gold else ""
    gold = str(gold or "").strip().strip("$").strip()
    if not question or not gold or row.get("image_1"):
        return None  # image-bearing items are not a text-reasoning measurement
    return question, gold


def _adapter_musique(row: dict) -> Optional[tuple]:
    question, gold = _text(row, "question"), _text(row, "answer")
    if not question or not gold:
        return None
    paragraphs = row.get("paragraphs") or []
    context = "\n\n".join(
        f"{p.get('title', '')}: {p.get('paragraph_text', '')}"
        for p in paragraphs
        if isinstance(p, dict)
    )[:12000]
    aliases = [a for a in (row.get("answer_aliases") or []) if str(a).strip()]
    prompt = f"Context:\n{context}\n\nQuestion: {question}" if context else question
    return prompt, [gold, *aliases]


def _grade_aliases(prediction: str, gold: Any) -> bool:
    """Any listed alias counts — MuSiQue golds are surface forms of one entity."""
    golds = gold if isinstance(gold, (list, tuple)) else [gold]
    return any(answer_equal(prediction, g) for g in golds)


def _grade_zebra(prediction: str, gold: Any) -> bool:
    """Full-grid exact match, with each cell BOUND to its house.

    A set of bare ``attr=value`` cells would pass a model that found every value
    but assigned them to the wrong houses — which is the whole puzzle. Cells are
    therefore keyed by house number on both sides.
    """
    want = _zebra_cells(str(gold))
    got = _zebra_cells(str(prediction))
    return bool(want) and want <= got


_HOUSE_HEADER = re.compile(r"house\s*[:#]?\s*(\d+)\s*[:\-]?", re.IGNORECASE)


def _zebra_cells(text: str) -> set:
    """Parse ``House 1: Name=Arnold, Color=white`` blocks into (house, attr, value).

    Split on the house headers rather than matching a greedy body per header: a
    greedy ``(.*)`` stops at the newline, so an answer written on ONE line loses
    every house after the first — and no split of that already-truncated body
    can recover them.
    """
    cells: set = set()
    parts = _HOUSE_HEADER.split(str(text))
    # split() yields [prefix, house1, body1, house2, body2, ...]
    for house, body in zip(parts[1::2], parts[2::2]):
        for cell in re.split(r"[;,|\n]", body):
            if "=" not in cell:
                continue
            attr, _, value = cell.partition("=")
            attr, value = normalize_answer(attr), normalize_answer(value)
            if attr and value:
                cells.add((house, attr, value))
    return cells


def _adapter_zebra(row: dict) -> Optional[tuple]:
    puzzle = _text(row, "puzzle")
    solution = row.get("solution")
    if not puzzle or not isinstance(solution, dict):
        return None
    header = solution.get("header") or []
    rows = solution.get("rows") or []
    if not header or not rows:
        return None
    # canonical gold in the exact shape the instruction asks the model for
    lines = []
    for r in rows:
        house = str(r[0]).strip() if r else ""
        cells = [
            f"{header[i]}={cell}"
            for i, cell in enumerate(r)
            if 0 < i < len(header)
        ]
        lines.append(f"House {house}: " + ", ".join(cells))
    return puzzle, "\n".join(lines)


def _adapter_folio(row: dict) -> Optional[tuple]:
    premises = _text(row, "premises")
    conclusion = _text(row, "conclusion")
    gold = _text(row, "label")
    if not premises or not conclusion or not gold:
        return None
    prompt = (
        f"Premises:\n{premises}\n\nConclusion: {conclusion}\n\n"
        "Is the conclusion True, False, or Uncertain given ONLY the premises?"
    )
    return prompt, gold


def _adapter_bbh(row: dict) -> Optional[tuple]:
    return _adapter_plain("input", "target")(row)


SPECS: list[Spec] = [
    # ── ch1 math: expected to span saturated → floor ────────────────────
    Spec("math500", "ch1-math", "HuggingFaceH4/MATH-500", split="test",
         max_tokens=20480, adapter=_adapter_plain("problem", "answer"), note="saturation check"),
    Spec("gsm_symbolic_main", "ch1-math", "apple/GSM-Symbolic", config="main",
         split="test", adapter=_adapter_gsm_symbolic, max_tokens=4096,
         note="paired-axis anchor (main/p1/p2)"),
    Spec("gsm_symbolic_p2", "ch1-math", "apple/GSM-Symbolic", config="p2",
         split="test", adapter=_adapter_gsm_symbolic, max_tokens=4096,
         note="hardest rung of the same axis"),
    Spec("aime_2026", "ch1-math", "MathArena/aime_2026", split="train",
         max_tokens=20480, adapter=_adapter_plain("problem", "answer"), note="frontier-saturated"),
    Spec("hmmt_feb_2026", "ch1-math", "MathArena/hmmt_feb_2026", split="train",
         adapter=_adapter_plain("problem", "answer"), max_tokens=20480),
    Spec("matharena_apex", "ch1-math", "MathArena/apex-shortlist", split="train",
         max_tokens=20480, adapter=_adapter_plain("problem", "answer"), note="floor check"),
    Spec("arxivmath", "ch1-math", "MathArena/arxivmath", split="train",
         adapter=_adapter_plain("problem", "answer"), max_tokens=20480,
         note="contamination-resistant by construction"),
    Spec("beyond_aime", "ch1-math", "ByteDance-Seed/BeyondAIME", split="test",
         adapter=_adapter_plain("problem", "answer"), max_tokens=20480),
    Spec("olympiadbench_math", "ch1-math", "Hothan/OlympiadBench",
         config="OE_TO_maths_en_COMP", split="train", adapter=_adapter_olympiadbench,
         max_tokens=20480),
    # ── ch2 code: the sandbox-free path ─────────────────────────────────
    Spec("lcb_execution", "ch2-code", "livecodebench/execution-v2", split="test",
         adapter=_adapter_lcb_execution, max_tokens=20480,
         note="exact string, NO sandbox"),
    Spec("cruxeval_output", "ch2-code", "cruxeval-org/cruxeval", split="test",
         adapter=_adapter_cruxeval, max_tokens=20480, note="exact string, no sandbox"),
    # ── ch3 puzzles ─────────────────────────────────────────────────────
    Spec("zebralogic", "ch3-puzzle", "WildEval/ZebraLogic", config="grid_mode",
         split="test", adapter=_adapter_zebra, grader=_grade_zebra,
         grades_raw_output=True, max_tokens=20480,
         instruction=(
             "Solve the puzzle. After your reasoning, output the full solution "
             "as one line per house in exactly this form:\n"
             "House 1: Name=..., Color=...\nHouse 2: Name=..., Color=...\n"
             "Use the attribute names from the puzzle. Prefix the block with "
             "'Answer:' on its own line."
         ),
         note="25 grid sizes x exactly 40; full-grid metric"),
    Spec("enigmata_eval", "ch3-puzzle", "BytedTsinghua-SIA/Enigmata-Eval",
         split="train", adapter=_adapter_plain("prompt", "answer"), max_tokens=20480),
    # ── ch4 atomic reasoning ────────────────────────────────────────────
    Spec("bbh_navigate", "ch4-basic", "lukaemon/bbh", config="navigate", split="test",
         adapter=_adapter_bbh, max_tokens=4096),
    Spec("bbh_tracking7", "ch4-basic", "lukaemon/bbh",
         config="tracking_shuffled_objects_seven_objects", split="test",
         adapter=_adapter_bbh, max_tokens=4096),
    Spec("bbh_word_sorting", "ch4-basic", "lukaemon/bbh", config="word_sorting",
         split="test", adapter=_adapter_bbh, max_tokens=4096),
    Spec("bbeh", "ch4-basic", "BBEH/bbeh", split="train",
         adapter=_adapter_plain("input", "target"), max_tokens=20480),
    Spec("musique", "ch4-basic", "bdsaglam/musique", split="validation",
         adapter=_adapter_musique, grader=_grade_aliases, max_tokens=4096,
         note="2/3/4-hop x answerable"),
    Spec("bamboogle", "ch4-basic", "chiayewken/bamboogle", split="test",
         adapter=_adapter_plain("Question", "Answer"), max_tokens=20480),
    Spec("folio", "ch4-basic", "tasksource/folio", split="validation",
         adapter=_adapter_folio, max_tokens=20480),
    Spec("musr_murder", "ch4-basic", "TAUR-Lab/MuSR", split="murder_mysteries",
         adapter=_adapter_musr, max_tokens=20480),
    Spec("mmlu_pro", "ch4-basic", "TIGER-Lab/MMLU-Pro", split="test",
         adapter=_adapter_mc(("question",), "options", "answer"), max_tokens=20480),
]


# ----------------------------------------------------------------------
# HF rows API
# ----------------------------------------------------------------------
def fetch_rows(spec: Spec, want: int, seed: int = 0, timeout: int = 60,
               n_windows: int = 12) -> list[dict]:
    """Pull rows through the datasets-server from windows spread over the split.

    Many of these splits are ordered by subset, task, or difficulty, so a head
    sample measures one slice while claiming to measure the set. Shuffling the
    order of 100-row pages does NOT fix that on its own: stopping as soon as
    enough rows are collected then takes them all from whichever one or two
    pages came first, which is still a single contiguous block.

    So: draw ``n_windows`` offsets spaced across the whole split (jittered, not
    page-aligned) and take an equal share from each. The result is a stratified
    cluster sample — better than a head, but still clustered, which is why the
    interval this feeds is called approximate.
    """
    params = {
        "dataset": spec.dataset,
        "config": spec.config,
        "split": spec.split,
        "offset": 0,
        "length": 1,
    }
    meta = requests.get(ROWS_API, params=params, timeout=timeout)
    meta.raise_for_status()
    total = meta.json().get("num_rows_total", 0)
    if not total:
        return []

    rng = random.Random(seed)
    target = want * 3  # over-fetch: the adapter drops ungradable rows
    windows = max(1, min(n_windows, math.ceil(total / 10)))
    per_window = min(100, max(1, math.ceil(target / windows)))
    stride = total / windows

    offsets = []
    for i in range(windows):
        base = int(i * stride)
        span = max(1, int(stride) - per_window)
        offsets.append(min(max(0, base + rng.randrange(span)), max(0, total - 1)))
    rng.shuffle(offsets)

    rows: list[dict] = []
    seen_offsets: set[int] = set()
    for offset in offsets:
        if offset in seen_offsets:
            continue
        seen_offsets.add(offset)
        params["offset"] = offset
        params["length"] = min(per_window, total - offset)
        try:
            resp = requests.get(ROWS_API, params=params, timeout=timeout)
            resp.raise_for_status()
        except requests.RequestException:
            continue
        rows.extend(r["row"] for r in resp.json().get("rows", []))
        if len(rows) >= target:
            break
    rng.shuffle(rows)
    return rows


# ----------------------------------------------------------------------
# Endpoint
# ----------------------------------------------------------------------
#: Thinking models split the chain into `reasoning_content`. Probes that audit
#: the CHAIN (arith_audit, step values) see nothing without it, so it is folded
#: in on request; answer extraction still reads the tail, which is `content`.
INCLUDE_REASONING = os.environ.get("BAND_INCLUDE_REASONING", "0") == "1"


def generate(prompt: str, max_tokens: int, temperature: float = 0.0,
             retries: int = 3) -> str:
    payload = {
        "model": MODEL_ID,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    for attempt in range(retries):
        try:
            resp = requests.post(
                f"{BASE_URL}/chat/completions", json=payload, timeout=900
            )
            resp.raise_for_status()
            message = resp.json()["choices"][0]["message"]
            content = message.get("content") or ""
            reasoning = message.get("reasoning_content") or ""
            if INCLUDE_REASONING and reasoning:
                return f"{reasoning}\n\n{content}" if content else reasoning
            # an empty content with a populated chain means the budget ran out
            # mid-thought; returning "" would report that as a refusal
            return content or reasoning
        except Exception:
            if attempt == retries - 1:
                return ""
            time.sleep(2 * (attempt + 1))
    return ""


def wilson(k: int, n: int, z: float = 1.96) -> tuple:
    """Wilson score interval — the point estimate alone cannot support
    'saturated' or 'floored' at these sample sizes."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (round(max(0.0, centre - half), 4), round(min(1.0, centre + half), 4))


#: Above this share of tag-less outputs the accuracy is a BUDGET measurement,
#: not a capability one, and the band cannot be claimed either way.
TRUNCATION_ALARM = 0.10


def band_of(acc: float, lo: float, hi: float, no_tag_rate: float = 0.0) -> str:
    """Classify against the 30-70% usable band, using the interval not the point.

    A high tag-less rate short-circuits the classification: the model never
    reached its answer on those items, so what was measured is the token budget.
    Reporting "USABLE" there would send the whole arc after a mechanism that
    does not exist.
    """
    if no_tag_rate > TRUNCATION_ALARM:
        return "budget_limited"
    if lo >= 0.70:
        return "saturated"
    if hi <= 0.30:
        return "floor"
    if 0.30 <= acc <= 0.70:
        return "USABLE"
    return "marginal"


# ----------------------------------------------------------------------
def run_spec(spec: Spec, n: int, concurrency: int, temperature: float) -> dict:
    started = time.time()
    try:
        raw = fetch_rows(spec, n)
    except Exception as exc:
        return {"name": spec.name, "error": f"fetch failed: {exc}"}
    adapter = spec.adapter or _adapter_plain(spec.question_field, spec.answer_field)

    items: list[tuple] = []
    for row in raw:
        try:
            pair = adapter(row)
        except Exception:
            pair = None
        if pair:
            items.append(pair)
        if len(items) >= n:
            break
    if not items:
        return {"name": spec.name, "error": "no gradable rows after adaptation"}

    def _one(item):
        question, gold = item
        prompt = f"{question}\n\n{spec.instruction}"
        output = generate(prompt, spec.max_tokens, temperature)
        predicted = extract_answer(output)
        grade = spec.grader or answer_equal
        graded_text = output if spec.grades_raw_output else predicted
        from evalvitals.analyzers.reasoning._text import has_answer_tag

        return {
            "correct": int(bool(grade(graded_text, gold))),
            "empty": int(not output.strip()),
            "no_answer_tag": int(not has_answer_tag(output)),
            "chars": len(output),
            "gold": str(gold)[:60],
            "predicted": str(predicted)[:60],
        }

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        graded = list(pool.map(_one, items))

    n_graded = len(graded)
    k = sum(g["correct"] for g in graded)
    acc = k / n_graded
    lo, hi = wilson(k, n_graded)
    no_tag_rate = sum(g["no_answer_tag"] for g in graded) / n_graded
    return {
        "name": spec.name,
        "chapter": spec.chapter,
        "dataset": spec.dataset,
        "config": spec.config,
        "split": spec.split,
        "n": n_graded,
        "n_correct": k,
        "accuracy": round(acc, 4),
        "ci95": [lo, hi],
        "band": band_of(acc, lo, hi, no_tag_rate),
        "empty_rate": round(sum(g["empty"] for g in graded) / n_graded, 4),
        # thinking models overrun the budget before the tag; without this column a
        # truncation-limited score reads as a capability score
        "no_answer_tag_rate": round(no_tag_rate, 4),
        "mean_output_chars": round(
            sum(g["chars"] for g in graded) / n_graded, 1
        ),
        "seconds": round(time.time() - started, 1),
        "note": spec.note,
        "samples": graded[:5],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="band_results.json")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--only", default="", help="comma-separated spec names")
    args = ap.parse_args()

    wanted = {s.strip() for s in args.only.split(",") if s.strip()}
    specs = [s for s in SPECS if not wanted or s.name in wanted]

    results: list[dict] = []
    out_path = args.out
    for spec in specs:
        print(f"[{spec.name}] running...", flush=True)
        result = run_spec(spec, args.n, args.concurrency, args.temperature)
        results.append(result)
        if "error" in result:
            print(f"  ERROR {result['error']}", flush=True)
        else:
            print(
                f"  acc={result['accuracy']:.3f} "
                f"CI={result['ci95']} band={result['band']} "
                f"n={result['n']} {result['seconds']}s",
                flush=True,
            )
        with open(out_path, "w") as fh:
            json.dump({"model": MODEL_ID, "results": results}, fh, indent=2)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
