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
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Optional

import requests

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from evalrx.analyzers.reasoning._text import (  # noqa: E402
    answer_equal,
    extract_answer,
    normalize_answer,
)
from evalrx.benchmark.tasks.llm.fetch import fetch_rows  # noqa: E402
from evalrx.benchmark.tasks.llm.specs import (  # noqa: E402
    ANSWER_MARK,
    MC_LETTERS,
    Spec,
    bbh_adapter,
    field_text,
    grade_latex,
    mc_adapter,
    mc_prompt,
    plain_adapter,
)
from evalrx.benchmark.tasks.llm.specs import SPECS as TASK_SPECS  # noqa: E402

BASE_URL = os.environ.get("BAND_BASE_URL", "http://127.0.0.1:8020/v1")
MODEL_ID = os.environ.get("BAND_MODEL_ID", "qwen3.5-9b")
#: Thinking mode for every request this module sends. OFF by default: each call
#: carries ``chat_template_kwargs={"enable_thinking": False}``. It is sent
#: explicitly because the Qwen3.5 checkpoints disagree on the template default
#: when the kwarg is absent (Qwen3.5-2B: off, Qwen3.5-9B: on). The band numbers
#: recorded in datasets.py were measured WITH thinking; set
#: ``BAND_ENABLE_THINKING=1`` (or ``generate(..., enable_thinking=True)``) to
#: reproduce that mode.
ENABLE_THINKING = os.environ.get("BAND_ENABLE_THINKING", "0") == "1"


# ----------------------------------------------------------------------
# Candidate specs. The nine the benchmark runs (TASK_SPECS) and their shared
# adapters/graders live in evalrx.benchmark.tasks.llm.specs; the rest of the
# survey is defined here.
# ----------------------------------------------------------------------
def _adapter_imo(row: dict) -> Optional[tuple]:
    # IMO-AnswerBench ships field names WITH SPACES; only integer answers are
    # cheaply gradable, the rest need a judge.
    question, gold = field_text(row, "question", "Question"), field_text(row, "answer", "Answer")
    if not question or not gold or not re.fullmatch(r"-?\d+", gold.strip()):
        return None
    return question, gold.strip()


def _adapter_gsm_symbolic(row: dict) -> Optional[tuple]:
    question, gold = field_text(row, "question"), field_text(row, "answer")
    if not question or not gold:
        return None
    # GSM8K-style gold: "reasoning #### 42"
    return question, gold.split("####")[-1].strip()


def _adapter_lcb_execution(row: dict) -> Optional[tuple]:
    code, inp, out = field_text(row, "code"), field_text(row, "input"), field_text(row, "output")
    if not code or not out:
        return None
    prompt = (
        "You are given a Python function and a call to it. Predict the return "
        f"value exactly.\n\n{code}\n\nCall:\n{inp}\n\n"
        "Give only the returned value on the last line as 'Answer: <value>'."
    )
    return prompt, out.strip()


def _adapter_polymath(row: dict) -> Optional[tuple]:
    # golds are LaTeX wrapped in $...$ ("$\\frac{\\pi}{3}$"); the delimiters are
    # notation, not answer
    question, gold = field_text(row, "question"), field_text(row, "answer")
    gold = gold.strip().strip("$").strip()
    if not question or not gold:
        return None
    return question, gold


def _adapter_musr(row: dict) -> Optional[tuple]:
    # MuSR ships `choices` as the STRING repr of a list, not a list
    import ast

    narrative, question = field_text(row, "narrative"), field_text(row, "question")
    raw, gold = row.get("choices"), field_text(row, "answer_choice")
    if isinstance(raw, str):
        try:
            raw = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            return None
    if not narrative or not isinstance(raw, (list, tuple)) or gold not in raw:
        return None
    letter = MC_LETTERS[list(raw).index(gold)]
    return mc_prompt(f"{narrative}\n\n{question}", list(raw)), letter


def _adapter_olympiadbench(row: dict) -> Optional[tuple]:
    # final_answer is a LIST of LaTeX strings; take the first and strip the $
    question = field_text(row, "question")
    gold = row.get("final_answer")
    if isinstance(gold, (list, tuple)):
        gold = gold[0] if gold else ""
    gold = str(gold or "").strip().strip("$").strip()
    if not question or not gold or row.get("image_1"):
        return None  # image-bearing items are not a text-reasoning measurement
    return question, gold


def _adapter_musique(row: dict) -> Optional[tuple]:
    question, gold = field_text(row, "question"), field_text(row, "answer")
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


def _hotpot_context(row: dict) -> str:
    """The 10 candidate paragraphs, 2 gold + 8 distractors — HotpotQA's own setup.

    ``context`` is a dict of PARALLEL lists (``title``, ``sentences``), not a list
    of paragraph objects, so zipping is the only correct read; indexing it like
    MuSiQue's ``paragraphs`` silently yields nothing.
    """
    ctx = row.get("context")
    if not isinstance(ctx, dict):
        return ""
    titles = ctx.get("title") or []
    bodies = ctx.get("sentences") or []
    paras = [
        f"{title}: {''.join(sents).strip()}"
        for title, sents in zip(titles, bodies)
        if str(title).strip()
    ]
    return "\n\n".join(paras)[:14000]


def _adapter_hotpot(row: dict) -> Optional[tuple]:
    """Open-book: the model is handed the paragraphs and must combine two of them."""
    question, gold = field_text(row, "question"), field_text(row, "answer")
    if not question or not gold:
        return None
    context = _hotpot_context(row)
    if not context:
        return None
    return f"Context:\n{context}\n\nQuestion: {question}", gold


def _adapter_hotpot_closed(row: dict) -> Optional[tuple]:
    """Closed-book: same questions, no paragraphs — comparable to bamboogle.

    Kept as a separate spec rather than a flag because the two settings are
    different tasks: one measures multi-hop READING, the other multi-hop RECALL,
    and pooling them would average a reading score with a memory score.
    """
    question, gold = field_text(row, "question"), field_text(row, "answer")
    if not question or not gold:
        return None
    return question, gold


def _answer_region(text: str) -> str:
    """Everything after the LAST 'Answer:' marker — the model's stated answer.

    Scanning the whole generation instead would grade the CHAIN. A 2*2 puzzle
    has two possible assignments and the chain enumerates both, so a subset
    match against the full text passes whatever the model finally concluded —
    the score would approach 100% without measuring anything. No marker means
    no stated answer, which is a failure and not a licence to go looking.
    """
    marks = list(ANSWER_MARK.finditer(str(text or "")))
    return text[marks[-1].end():] if marks else ""


def _grade_zebra(prediction: str, gold: Any) -> bool:
    """Full-grid exact match, with each cell BOUND to its house.

    A set of bare ``attr=value`` cells would pass a model that found every value
    but assigned them to the wrong houses — which is the whole puzzle. Cells are
    therefore keyed by house number on both sides.
    """
    want = _zebra_cells(str(gold))
    got = _zebra_cells(_answer_region(str(prediction)))
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
            # The attribute is a COLUMN LABEL, not an answer: "Car Model",
            # "CarModel" and "car  model" name the same column, and a model
            # penalised for the spacing would be scored on typography.
            attr = re.sub(r"[^a-z0-9]", "", normalize_answer(attr))
            value = normalize_answer(value)
            if attr and value:
                cells.add((house, attr, value))
    return cells


def _adapter_zebra(row: dict) -> Optional[tuple]:
    puzzle = field_text(row, "puzzle")
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
    # The format order is built from THIS row's header, so the spec carries no
    # fixed example to contradict it.
    return f"{puzzle}\n\n{_zebra_instruction(header)}", "\n".join(lines)


#: Built per row, because a FIXED example is what broke the earlier sweep.
#:
#: The old instruction showed "House 1: Name=..., Color=..." and then said "use
#: the attribute names from the puzzle". On any puzzle without a Color
#: attribute those two orders contradict, and the model does not pick one — it
#: oscillates. A 2*2 item (solved correctly inside the chain within a few
#: hundred tokens) then spent the remaining ~8000 repeating "The attribute
#: names are `Name` and `Car models`. Wait, I'll check if I can just use `Car`."
#: until the budget ran out, and was scored a truncated failure.
#:
#: Naming the exact attributes removes the contradiction. It is a formatting
#: aid, not a hint: the header is the column NAMES, never the values.
def _zebra_instruction(header: list) -> str:
    attrs = [str(h) for h in header[1:]]
    shape = ", ".join(f"{a}=<value>" for a in attrs)
    return (
        "Solve the puzzle. After your reasoning, write 'Answer:' on its own "
        "line, then one line per house in exactly this form:\n"
        f"House 1: {shape}\n"
        f"House 2: {shape}\n"
        f"Use these attribute names verbatim: {', '.join(attrs)}. "
        "Write nothing after the last house line."
    )


def _row_in(field: str, allowed) -> Callable[[dict], bool]:
    """Keep rows whose ``field`` is in ``allowed`` — one rung of a ladder."""
    allowed = frozenset(str(a) for a in allowed)
    return lambda row: str(row.get(field, "")) in allowed


def _zebra_logspace(size: str) -> float:
    """log10 of the assignment space for an ``H*A`` ZebraLogic grid.

    A grid with H houses and A attributes has ``(H!)^A`` complete assignments,
    which spans 0.6 to 17.1 in log10 across the 25 sizes in the split. That
    range is the whole reason the pooled score is meaningless: 2*2 and 6*6 are
    not the same benchmark, and averaging them describes neither.
    """
    houses, attrs = (int(part) for part in size.split("*"))
    return attrs * math.log10(math.factorial(houses))


#: The split is exactly 40 items for each of the 25 sizes (verified against the
#: full split, not sampled), so ranking by search space and cutting into fifths
#: gives five rungs of 200 rows each — enough for n=50 per rung with room over.
_ZEBRA_SIZES = [f"{h}*{a}" for h in range(2, 7) for a in range(2, 7)]
ZEBRA_TIERS: list[tuple[str, tuple[str, ...]]] = [
    (f"t{i // 5 + 1}", tuple(sorted(_ZEBRA_SIZES, key=_zebra_logspace)[i:i + 5]))
    for i in range(0, len(_ZEBRA_SIZES), 5)
]


# ── Enigmata ─────────────────────────────────────────────────────────────────
# Measured over the FULL 4758-row split, not a sample: 36 tasks in 7 types, and
# the gold format varies by task far more than the pooled spec assumed.
#
#  * 4 tasks (binario, campsite, star_battle, zebra_logic) ship MULTI-LINE grid
#    golds. extract_answer is single-line, so those 550 rows could not score
#    above 0 no matter what the model wrote — the pooled 0.12 was partly this.
#  * ~20 tasks ship a JSON matrix ("[[3, 1, 2], [7, 8, 6]]"), which the default
#    substring grader marks wrong on a stray space.
#  * The rest are short strings ("NO", "False", "Qg7#") the default grader
#    handles correctly.
#
# So: grade the matrices structurally, keep the short-string tasks on the
# default grader, and leave the 4 grid tasks out entirely rather than let a
# guaranteed 0 depress a rung.

#: JSON-matrix golds — compared as parsed structure, not as text.
_ENIGMATA_STRUCTURAL = frozenset({
    "big_bench_symbolic", "car_painting", "eight_puzzle", "fifteen_puzzle",
    "hamiltonian_path", "hitori", "kakurasu", "light_up", "magic_square",
    "minesweeper", "nine_puzzle", "sixteen_puzzle", "skyscraper", "slant",
    "sudoku", "sudoku2", "sum_skyscraper", "symbolic_hard", "tic_tac_toe",
    "twiddle",
})
#: Short scalar golds the default grader already reads correctly.
_ENIGMATA_SHORT = frozenset({
    "FOLIO", "checkmate_in_one", "crypto_KKA", "crypto_KPA",
    "hamiltonian_cycle", "knights_and_knaves", "natural_language_navigation",
})
#: Excluded, with the reason, so this is a decision rather than an oversight:
#: multi-line grid golds (binario/campsite/star_battle/zebra_logic) and
#: free-prose golds no cheap grader can judge ("The answer is: (10-5/15)*9-2 =
#: 85", "cannot form 24", a crossword JSON dict).
_ENIGMATA_UNGRADED = frozenset({
    "binario", "campsite", "star_battle", "zebra_logic",
    "countdown", "game24", "maze", "stack_permutation", "full_crosswords",
})
_ENIGMATA_GRADABLE = _ENIGMATA_STRUCTURAL | _ENIGMATA_SHORT


def _freeze(obj: Any) -> Any:
    if isinstance(obj, (list, tuple)):
        return tuple(_freeze(x) for x in obj)
    if isinstance(obj, str):
        return obj.strip().lower()
    return obj


def _last_bracket_span(text: str, window: int = 6000) -> str:
    """The last balanced ``[...]`` group, searched from the end.

    The answer matrix is at the tail; the chain above it is full of unbalanced
    brackets. Only the tail ``window`` is scanned so a 40k-token generation
    cannot turn this into a quadratic scan.
    """
    text = text[-window:]
    end = text.rfind("]")
    while end != -1:
        depth = 0
        for i in range(end, -1, -1):
            if text[i] == "]":
                depth += 1
            elif text[i] == "[":
                depth -= 1
                if depth == 0:
                    return text[i:end + 1]
        end = text.rfind("]", 0, end)
    return ""


def _structural_key(value: Any) -> Optional[tuple]:
    """Parse a bracketed answer into a canonical nested tuple.

    ``ast.literal_eval`` only builds literals — model output never executes.
    """
    import ast

    span = _last_bracket_span(str(value or ""))
    if not span or len(span) > 20000:
        return None
    try:
        return _freeze(ast.literal_eval(span))
    except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
        return None


#: Enigmata's own response-format section asks for the answer inside a fence.
_FENCE = re.compile(r"```[a-zA-Z0-9_+-]*\s*\n(.*?)```", re.DOTALL)


def _last_fence(text: str) -> str:
    blocks = _FENCE.findall(str(text or ""))
    return blocks[-1].strip() if blocks else ""


def _grid_key(text: str) -> Optional[tuple]:
    """Parse a whitespace-separated grid into the same shape a matrix parses to.

    Enigmata asks for "3 4 1 2\\n1 2 3 4"; its gold is "[[3, 4, 1, 2], [1, 2,
    3, 4]]". Those are the SAME answer, and a grader that only reads brackets
    scores every correct fenced grid as wrong — which is most of the split.
    """
    rows = []
    for line in str(text or "").strip().splitlines():
        cells = line.split()
        if not cells:
            continue
        row = []
        for cell in cells:
            try:
                row.append(int(cell))
            except ValueError:
                row.append(cell.strip().lower())
        rows.append(tuple(row))
    return tuple(rows) if rows else None


def _grade_structural(prediction: Any, gold: Any) -> bool:
    """Structure-aware equality across the three shapes one answer arrives in.

    ``[[1, 2]]`` == ``[[1,2]]`` == a fenced ``1 2`` grid. Gold is always the
    bracketed form; the model writes whichever the dataset asked for.
    """
    raw = str(prediction or "")
    want = _structural_key(gold)
    fence = _last_fence(raw)

    if want is None:
        # A scalar gold ("NO", "No valid solution."). Check the fence first —
        # the dataset told the model to put its answer there.
        return bool(
            (fence and answer_equal(fence, gold))
            or answer_equal(extract_answer(raw), gold)
        )

    # The fence is where the dataset told the model to put the answer. The raw
    # fallback is for models that ignored that, and is limited to the TAIL so it
    # cannot mine a discarded working grid out of the middle of the chain.
    for candidate in (fence, raw[-1200:]):
        if not candidate:
            continue
        # brackets win when present: a flat path "[0, 2, 3, 1, 0]" read as a
        # grid would come back as a tuple of string fragments
        got = _structural_key(candidate) if "[" in candidate else _grid_key(candidate)
        if got is not None and got == want:
            return True
        if got is None and "[" in candidate:
            # a fence holding a grid that also mentions a bracket elsewhere
            if _grid_key(candidate) == want:
                return True
    return False


def _enigmata_difficulty(row: dict) -> str:
    meta = row.get("meta")
    if isinstance(meta, str):
        try:
            meta = json.loads(meta)
        except (ValueError, TypeError):
            return ""
    return str((meta or {}).get("difficulty", "")) if isinstance(meta, dict) else ""


#: Tasks whose gold is ONE valid certificate among many. Any Hamiltonian cycle
#: is a correct answer and it may start at any vertex and run in either
#: direction, so exact match against the stored one marks correct answers wrong.
#: Measured over the full split: 47% of `hamiltonian_cycle` golds are a specific
#: cycle rather than "NO", and every `hamiltonian_path` gold is a path.
#: Only their DECISION instances are uniquely gradable, so only those are kept.
_ENIGMATA_CERTIFICATE = frozenset({"hamiltonian_cycle", "hamiltonian_path"})
_DECISIONS = frozenset({"NO", "YES"})


def _enigmata_filter(difficulty: Optional[str] = None,
                     tasks=None) -> Callable[[dict], bool]:
    allowed = frozenset(tasks) if tasks is not None else _ENIGMATA_GRADABLE

    def _fn(row: dict) -> bool:
        task = str(row.get("task_name", ""))
        if task not in allowed:
            return False
        if (task in _ENIGMATA_CERTIFICATE
                and str(row.get("answer", "")).strip().upper() not in _DECISIONS):
            return False
        return difficulty is None or _enigmata_difficulty(row) == difficulty

    return _fn


def _adapter_folio(row: dict) -> Optional[tuple]:
    premises = field_text(row, "premises")
    conclusion = field_text(row, "conclusion")
    gold = field_text(row, "label")
    if not premises or not conclusion or not gold:
        return None
    prompt = (
        f"Premises:\n{premises}\n\nConclusion: {conclusion}\n\n"
        "Is the conclusion True, False, or Uncertain given ONLY the premises?"
    )
    return prompt, gold


# ── LiveBench-Math ────────────────────────────────────────────────────────────
# GEPA's own class for this benchmark is `LiveBenchMathBench`, which is where the
# name "MathBench" in circulation comes from -- it is NOT OpenCompass MathBench.
#
# 368 items over three tasks (verified over the full split, not sampled):
# AMPS_Hard 150 / math_comp 146 / olympiad 72. 182 are still live; the rest carry
# a `livebench_removal_date` and are retired from the rolling benchmark.
#
# Every item ships its OWN answer-format instruction inside `turns[0]` -- boxed
# LaTeX for AMPS_Hard, a bare 3-digit string for math_comp, a comma-separated
# expression ordering for olympiad. So append_instruction is off: adding a fourth
# format order is the same defect that made the model loop on ZebraLogic.
_LB_SEQUENCE = re.compile(r"[\d,\s]+")


def _adapter_livebench_math(row: dict) -> Optional[tuple]:
    turns = row.get("turns")
    if isinstance(turns, (list, tuple)):
        # an EMPTY list must not fall through to field_text(), which stringifies it
        # to "[]" — a non-empty string that would be sent to the model verbatim
        prompt = str(turns[0]) if turns else ""
    else:
        prompt = field_text(row, "turns")
    gold = field_text(row, "ground_truth")
    if not prompt or not gold:
        return None
    return prompt, gold.strip()


def _grade_livebench_math(prediction: Any, gold: Any) -> bool:
    """Three answer shapes in one benchmark, so three comparisons.

    The olympiad task answers with an ORDERING of expression identifiers
    ("1,6,7,2,3,4,5"); order carries the whole answer, so it is compared as a
    sequence rather than as a set or a number. Everything else falls through to
    the numeric-then-normalised-LaTeX grader, which already prefers \\boxed{}.
    """
    want = str(gold).strip()
    if "," in want and _LB_SEQUENCE.fullmatch(want):
        got = extract_answer(prediction)
        seq = [p.strip() for p in str(got).split(",")]
        return [p for p in seq if p] == [p.strip() for p in want.split(",") if p.strip()]
    return grade_latex(prediction, gold)


def _still_live(row: dict) -> bool:
    """Drop questions LiveBench has retired from the rolling release."""
    return not str(row.get("livebench_removal_date") or "").strip()


def _adapter_imo_answerbench(row: dict) -> Optional[tuple]:
    # field names carry a CAPITAL and a SPACE: "Problem", "Short Answer"
    question, gold = field_text(row, "Problem"), field_text(row, "Short Answer")
    if not question or not gold:
        return None
    return question, gold.strip()


SPECS: list[Spec] = [
    # ── ch1 math: expected to span saturated → floor ────────────────────
    Spec("math500", "ch1-math", "HuggingFaceH4/MATH-500", split="test",
         max_tokens=20480, adapter=plain_adapter("problem", "answer"), note="saturation check"),
    Spec("gsm_symbolic_main", "ch1-math", "apple/GSM-Symbolic", config="main",
         split="test", adapter=_adapter_gsm_symbolic, max_tokens=4096,
         note="paired-axis anchor (main/p1/p2)"),
    Spec("gsm_symbolic_p2", "ch1-math", "apple/GSM-Symbolic", config="p2",
         split="test", adapter=_adapter_gsm_symbolic, max_tokens=4096,
         note="hardest rung of the same axis"),
    Spec("aime_2026", "ch1-math", "MathArena/aime_2026", split="train",
         max_tokens=20480, adapter=plain_adapter("problem", "answer"), note="frontier-saturated"),
    Spec("hmmt_feb_2026", "ch1-math", "MathArena/hmmt_feb_2026", split="train",
         adapter=plain_adapter("problem", "answer"), max_tokens=20480),
    Spec("matharena_apex", "ch1-math", "MathArena/apex-shortlist", split="train",
         max_tokens=20480, adapter=plain_adapter("problem", "answer"), note="floor check"),
    Spec("arxivmath", "ch1-math", "MathArena/arxivmath", split="train",
         adapter=plain_adapter("problem", "answer"), max_tokens=20480,
         note="contamination-resistant by construction"),
    Spec("beyond_aime", "ch1-math", "ByteDance-Seed/BeyondAIME", split="test",
         adapter=plain_adapter("problem", "answer"), max_tokens=20480),
    Spec("olympiadbench_math", "ch1-math", "Hothan/OlympiadBench",
         config="OE_TO_maths_en_COMP", split="train", adapter=_adapter_olympiadbench,
         max_tokens=20480),
    # PolyMATH is TWO paired axes in one repo: config = language (18), split =
    # difficulty (low/medium/high/top), 125 problems each, and the same index in
    # two languages is the same problem with the same gold. The `low` rung is
    # GSM8K-level and omitted as pre-saturated.
    Spec("polymath_en_medium", "ch1-math", "Qwen/PolyMath", grader=grade_latex, config="en",
         split="medium", adapter=_adapter_polymath, max_tokens=20480,
         note="paired axis: same 125 problems as every other language; "
              "only 47% of golds are plain numbers, the rest are matched on "
              "normalised LaTeX form and algebraic rewrites will miss"),
    Spec("polymath_en_high", "ch1-math", "Qwen/PolyMath", grader=grade_latex, config="en",
         split="high", adapter=_adapter_polymath, max_tokens=20480,
         note="difficulty ladder rung 3 of 4 (AIME-level); 68% numeric golds"),
    Spec("polymath_en_top", "ch1-math", "Qwen/PolyMath", grader=grade_latex, config="en",
         split="top", adapter=_adapter_polymath, max_tokens=20480,
         note="difficulty ladder rung 4 of 4 (olympiad-level); only 31% "
              "numeric golds — a low score here is partly the grader"),
    Spec("polymath_zh_medium", "ch1-math", "Qwen/PolyMath", grader=grade_latex, config="zh",
         split="medium", adapter=_adapter_polymath, max_tokens=20480,
         note="cross-language arm of polymath_en_medium — identical golds"),
    # ── ch2 code: the sandbox-free path ─────────────────────────────────
    Spec("lcb_execution", "ch2-code", "livecodebench/execution-v2", split="test",
         adapter=_adapter_lcb_execution, max_tokens=20480,
         note="exact string, NO sandbox"),
    TASK_SPECS["cruxeval_output"],
    # ── ch3 puzzles ─────────────────────────────────────────────────────
    # Kept for the record: this is the spec that produced the pooled 0.22.
    # It is SUPERSEDED by the zebra_t* rungs below — see their note.
    Spec("zebralogic", "ch3-puzzle", "WildEval/ZebraLogic", config="grid_mode",
         split="test", adapter=_adapter_zebra, grader=_grade_zebra,
         grades_raw_output=True, max_tokens=20480, append_instruction=False,
         note="POOLED over 25 grid sizes — superseded by zebra_t1..t5"),
    Spec("enigmata_eval", "ch3-puzzle", "BytedTsinghua-SIA/Enigmata-Eval",
         split="train", adapter=plain_adapter("prompt", "answer"), max_tokens=20480,
         note="POOLED over all tasks — superseded by enigmata_* slices"),
    # ── ch4 atomic reasoning ────────────────────────────────────────────
    Spec("bbh_navigate", "ch4-basic", "lukaemon/bbh", config="navigate", split="test",
         adapter=bbh_adapter, max_tokens=4096),
    TASK_SPECS["bbh_tracking7"],
    TASK_SPECS["bbh_word_sorting"],
    Spec("bbh_object_counting", "ch4-basic", "lukaemon/bbh",
         config="object_counting", split="test", adapter=bbh_adapter,
         max_tokens=4096,
         note="250 items; TextGrad's BBH pair with word_sorting. Gold is a bare "
              "integer ('8'), so answer_equal's numeric path grades it exactly "
              "and there is no option-letter surface to mis-extract — unlike "
              "bbh_tracking7, whose '(X)' golds were mis-graded until the "
              "_PLACEHOLDER fix. The other four BBH slices measured 0.980 "
              "saturated / 0.720 marginal / 0.600 / 0.540, so this one's band "
              "is genuinely unknown"),
    Spec("bbeh", "ch4-basic", "BBEH/bbeh", split="train",
         adapter=plain_adapter("input", "target"), max_tokens=20480),
    Spec("musique", "ch4-basic", "bdsaglam/musique", split="validation",
         adapter=_adapter_musique, grader=_grade_aliases, max_tokens=4096,
         note="2/3/4-hop x answerable"),
    TASK_SPECS["bamboogle"],
    Spec("hotpotqa_gepa", "ch4-basic", "hotpotqa/hotpot_qa",
         config="fullwiki", split="train", adapter=_adapter_hotpot,
         max_tokens=20480,
         note="Open-book (10 paragraphs, 2 gold + 8 distractors) — HotpotQA's own "
              "distractor protocol. GEPA reports Qwen3-8B 42.33 on a seed-1 300-item "
              "sample of this same split, but its retrieval setup is not reproducible "
              "here, so that anchor is a reference for the distribution, NOT for these "
              "items. Sampled from TRAIN to match GEPA's source, which is also a "
              "contamination risk worth naming"),
    Spec("hotpotqa_gepa_closed", "ch4-basic", "hotpotqa/hotpot_qa",
         config="fullwiki", split="train", adapter=_adapter_hotpot_closed,
         max_tokens=20480,
         note="Closed-book control: same questions, no paragraphs. Isolates whether a "
              "band position comes from multi-hop READING or multi-hop RECALL"),
    Spec("folio", "ch4-basic", "tasksource/folio", split="validation",
         adapter=_adapter_folio, max_tokens=20480),
    Spec("musr_murder", "ch4-basic", "TAUR-Lab/MuSR", split="murder_mysteries",
         adapter=_adapter_musr, max_tokens=20480),
    Spec("supergpqa", "ch4-basic", "m-a-p/SuperGPQA", split="train",
         adapter=mc_adapter(("question",), "options", "answer_letter"),
         max_tokens=20480,
         note="285 disciplines; carries difficulty/discipline/is_calculation "
              "fields for slicing"),
    # SuperGPQA is in-band (0.520) but 26,529 items. These are its own named
    # subdivisions under 1000, NOT a random sub-sample -- but the pooled 0.520 is
    # 67% Engineering+Science, and the small disciplines are humanities skewed
    # heavily toward `easy`, so the band has to be re-measured per slice rather
    # than inherited. Medicine/hard is the only small slice on the hard end.
    TASK_SPECS["supergpqa_medicine_hard"],
    TASK_SPECS["supergpqa_law"],
    TASK_SPECS["supergpqa_economics"],
    Spec("mmlu_pro", "ch4-basic", "TIGER-Lab/MMLU-Pro", split="test",
         adapter=mc_adapter(("question",), "options", "answer"), max_tokens=20480),
    # ── added after the 2026-08-14 screening ────────────────────────────
    Spec("livebench_math", "ch1-math", "livebench/math", split="test",
         adapter=_adapter_livebench_math, grader=_grade_livebench_math,
         append_instruction=False, max_tokens=40960,
         note="368 items: AMPS_Hard 150 / math_comp 146 / olympiad 72. "
              "GEPA reports Qwen3-8B baseline 48.70 on its 126-item random "
              "third of this same set; the split's task mix (52/50/24) is "
              "proportionally identical to the full set, so the full 368 is "
              "comparable to that anchor at 3x the sample"),
    Spec("livebench_math_live", "ch1-math", "livebench/math", split="test",
         adapter=_adapter_livebench_math, grader=_grade_livebench_math,
         append_instruction=False, max_tokens=40960,
         row_filter=_still_live, n_windows=12, fetch_multiplier=4,
         note="the 182 questions LiveBench has NOT retired — the actual live "
              "rolling benchmark; contamination-limited by construction"),
    Spec("olymmath_en_hard", "ch1-math", "RUC-AIBOX/OlymMATH", config="en-hard",
         split="test", adapter=plain_adapter("problem", "answer"),
         max_tokens=40960,
         note="ACL 2026 long; purpose-built to resist saturation. n=100 means "
              "a ~+/-10pp interval — read the band as guidance, not a test"),
    TASK_SPECS["minervamath"],
    Spec("imo_answerbench", "ch1-math", "Hwilner/imo-answerbench", split="train",
         adapter=_adapter_imo_answerbench, grader=grade_latex, max_tokens=40960,
         note="EMNLP 2025 main (DeepMind). Every accuracy anchor the screening "
              "produced for this set was found to be fabricated, so the band is "
              "genuinely unknown — this run is the first real evidence"),
    TASK_SPECS["bbh_causal_judgement"],
    Spec("bbeh_boardgame_qa", "ch4-basic", "jgyasu/bbeh", config="boardgame_qa",
         split="train", adapter=plain_adapter("input", "target"),
         max_tokens=40960,
         note="BBEH (2025) rule-deduction; far less post-training contamination "
              "than 2022-era BBH"),
]


# ── ch3, sliced ───────────────────────────────────────────────────────────────
# The pooled specs above measured a MIXTURE and reported its mean, which is why
# ch3 came back with nothing usable: a 0.22 that is really "t1 saturated, t5
# floored" sits in no band at all. Both datasets carry the ladder as a column,
# so the rungs are scanned rather than averaged. The budget also doubles — the
# pooled runs were >10% tag-less, so part of that 0.22 was the token limit.
SPECS += [
    Spec(f"zebra_{tier}", "ch3-puzzle", "WildEval/ZebraLogic", config="grid_mode",
         split="test", adapter=_adapter_zebra, grader=_grade_zebra,
         grades_raw_output=True, max_tokens=40960,
         # the format order is per-row, built by the adapter from this puzzle's
         # own attribute names
         append_instruction=False,
         row_filter=_row_in("size", sizes),
         # 1000-row split / 100-row pages: 10 windows sweep all of it, and each
         # rung is 200 rows scattered across it, so nothing narrower will do.
         n_windows=10, fetch_multiplier=4,
         note=(f"grid sizes {'/'.join(sizes)} — "
               f"log10 search space "
               f"{min(_zebra_logspace(s) for s in sizes):.1f}"
               f"-{max(_zebra_logspace(s) for s in sizes):.1f}; "
               "200 rows available"))
    for tier, sizes in ZEBRA_TIERS
]

#: Enigmata's own declared ladder. fetch_multiplier is deliberately larger than
#: the rung can supply so the early-stop never fires: the split is ORDERED BY
#: TASK, so stopping after a few windows would sample a handful of tasks and
#: call it the rung.
SPECS += [
    Spec(f"enigmata_{level}", "ch3-puzzle", "BytedTsinghua-SIA/Enigmata-Eval",
         split="train", adapter=plain_adapter("prompt", "answer"),
         grader=_grade_structural, grades_raw_output=True, max_tokens=40960,
         # Enigmata ships its own "Response Format" section
         append_instruction=False,
         row_filter=_enigmata_filter(difficulty=level),
         n_windows=48, fetch_multiplier=20,
         note=f"declared difficulty={level}, 15 gradable tasks, ~700 rows")
    for level in ("easy", "medium", "hard")
] + [
    Spec("enigmata_short", "ch3-puzzle", "BytedTsinghua-SIA/Enigmata-Eval",
         split="train", adapter=plain_adapter("prompt", "answer"),
         grader=_grade_structural, grades_raw_output=True, max_tokens=40960,
         append_instruction=False,
         row_filter=_enigmata_filter(tasks=_ENIGMATA_SHORT),
         n_windows=48, fetch_multiplier=20,
         note="scalar-gold tasks only (NO/False/Qg7#) — the rung with the "
              "least grader risk, so a floor here is the model, not the parser"),
]


# ----------------------------------------------------------------------
# Endpoint
# ----------------------------------------------------------------------
#: Thinking models split the chain into `reasoning_content`. Probes that audit
#: the CHAIN (arith_audit, step values) see nothing without it, so it is folded
#: in on request; answer extraction still reads the tail, which is `content`.
INCLUDE_REASONING = os.environ.get("BAND_INCLUDE_REASONING", "0") == "1"


#: Sampling for the sweep. NOT greedy, deliberately.
#:
#: Measured on one ZebraLogic 2*2 item, which the model solves correctly within
#: a few hundred tokens either way:
#:
#:   T=0.0                    16384 tok, finish_reason=length  (never stopped)
#:   T=0.7 top_p=0.95          7602 tok, finish_reason=stop
#:   T=1.0 top_p=0.95 top_k=20 1352 tok, finish_reason=stop
#:
#: Under greedy decoding the model finishes reasoning and then loops on
#: self-verification ("I will ensure the answer format is exact." repeated) to
#: the token cap. Scoring that as a failed puzzle measures the decoding
#: configuration. These are Qwen's documented thinking-mode settings; they are
#: kept with thinking off (ENABLE_THINKING) too -- harmless there, and still
#: what protects a run that turns thinking back on.
SAMPLING = {"temperature": 0.6, "top_p": 0.95, "top_k": 20}


def generate(prompt: str, max_tokens: int, temperature: float = 0.0,
             retries: int = 3, sampling: Optional[dict] = None,
             enable_thinking: Optional[bool] = None) -> tuple:
    """Return ``(text, finish_reason)``.

    ``enable_thinking`` (default: the module's ``ENABLE_THINKING``, off) is sent
    on every request as ``chat_template_kwargs`` so the chat template renders the
    same way on every checkpoint.

    The finish reason matters as much as the text. ``no_answer_tag_rate`` was
    only ever a PROXY for "the budget ran out before the answer", and it is a
    bad one for datasets that do not use an answer tag — there it reads 100%
    truncation on a run with none. ``finish_reason == "length"`` is the
    endpoint stating it directly. ``"error"`` is kept distinct from a truncated
    generation so a dead request is never counted as a wrong answer.
    """
    payload = {
        "model": MODEL_ID,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "chat_template_kwargs": {
            "enable_thinking": bool(
                ENABLE_THINKING if enable_thinking is None else enable_thinking
            )
        },
    }
    payload.update(sampling or {})
    last_exc = "unknown"
    for attempt in range(retries):
        try:
            # Some of these tasks generate 70k+ characters and queue behind a
            # full batch; a 30-minute ceiling turned ~14% of one Enigmata slice
            # into client-side timeouts that were then scored as wrong answers.
            resp = requests.post(
                f"{BASE_URL}/chat/completions", json=payload, timeout=3600
            )
            resp.raise_for_status()
            choice = resp.json()["choices"][0]
            message = choice["message"]
            reason = choice.get("finish_reason") or "unknown"
            content = message.get("content") or ""
            reasoning = message.get("reasoning_content") or ""
            if INCLUDE_REASONING and reasoning:
                text = f"{reasoning}\n\n{content}" if content else reasoning
            else:
                # an empty content with a populated chain means the budget ran
                # out mid-thought; returning "" would report that as a refusal
                text = content or reasoning
            return text, reason
        except Exception as exc:
            # Record WHICH failure. An error_rate with no cause is a number
            # you cannot act on: a read timeout, a refused connection and a
            # malformed body all need different fixes.
            last_exc = type(exc).__name__
            if attempt == retries - 1:
                return "", f"error:{last_exc}"
            time.sleep(2 * (attempt + 1))
    return "", f"error:{last_exc}"


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

    The veto is deliberately unconditional, and it is NOT redundant with
    ``budget_bracket`` below.  The bracket answers "could the accuracy leave the
    band", but the band is a proxy for the thing M2 actually needs: PASS and FAIL
    pools that both carry capability signal.  At a 36% tag-less rate more than a
    third of the FAIL labels are budget artefacts, and a paired test over that
    pool attributes truncation to capability no matter how comfortably the point
    estimate sits mid-band.  Label quality and band position are two conditions,
    not one.
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


def budget_bracket(acc: float, budget_signal: float) -> tuple:
    """Bound where the accuracy could land if every unfinished item were resolved.

    An unfinished item is one the model never got to answer, so resolving it can
    only ADD correct answers: the budget moves accuracy upward only, making the
    measured value a floor and ``acc + budget_signal`` its ceiling.  This does
    not overturn a ``budget_limited`` verdict -- see ``band_of`` -- but it is what
    makes such a verdict actionable, by separating the datasets a bigger budget
    would rescue from the ones it would not.  A bracket that stays inside the
    band means re-measuring is worth the GPU hours; one that spans 0.18-0.86 means
    the run would tell you nothing you do not already know.
    """
    return (round(acc, 4), round(min(1.0, acc + max(0.0, budget_signal)), 4))


def bracket_in_band(acc: float, budget_signal: float) -> bool:
    """True when no budget could carry this dataset out of the 30-70% band."""
    lo, hi = budget_bracket(acc, budget_signal)
    return lo >= 0.30 and hi <= 0.70


# ----------------------------------------------------------------------
def run_spec(spec: Spec, n: int, concurrency: int, temperature: float,
             sampling: Optional[dict] = None) -> dict:
    started = time.time()
    fetch_rows.last_failed_windows = 0
    try:
        raw = fetch_rows(spec, n)
    except Exception as exc:
        return {"name": spec.name, "error": f"fetch failed: {exc}"}
    failed_windows = getattr(fetch_rows, "last_failed_windows", 0)
    adapter = spec.adapter or plain_adapter(spec.question_field, spec.answer_field)

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
        prompt = (
            f"{question}\n\n{spec.instruction}" if spec.append_instruction
            else question
        )
        output, reason = generate(prompt, spec.max_tokens, temperature,
                                  sampling=sampling)
        predicted = extract_answer(output)
        grade = spec.grader or answer_equal
        graded_text = output if spec.grades_raw_output else predicted
        from evalrx.analyzers.reasoning._text import has_answer_tag

        return {
            "correct": int(bool(grade(graded_text, gold))),
            "empty": int(not output.strip()),
            "no_answer_tag": int(not has_answer_tag(output)),
            "truncated": int(reason == "length"),
            "errored": int(reason.startswith("error")),
            "finish_reason": reason,
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
    truncated_rate = sum(g["truncated"] for g in graded) / n_graded
    error_rate = sum(g["errored"] for g in graded) / n_graded
    import collections as _c
    error_kinds = dict(_c.Counter(
        g["finish_reason"] for g in graded if g["errored"]
    ))
    # A spec with no answer tag in its protocol reads 100% tag-less on a run
    # with no truncation at all, so the tag proxy only counts when the spec
    # actually asks for a tag. finish_reason applies either way.
    budget_signal = max(
        truncated_rate, no_tag_rate if spec.append_instruction else 0.0
    )
    return {
        "name": spec.name,
        "chapter": spec.chapter,
        "dataset": spec.dataset,
        "config": spec.config,
        "split": spec.split,
        "n": n_graded,
        "n_requested": n,
        # A sliced spec can run out of rows (ZebraLogic has exactly 40 per grid
        # size). Without this column an n=11 slice and an n=50 one print the
        # same way and the wider interval reads as a real difference.
        "undersampled": bool(n_graded < n),
        "failed_windows": failed_windows,
        "n_correct": k,
        "accuracy": round(acc, 4),
        "ci95": [lo, hi],
        "band": band_of(acc, lo, hi, budget_signal),
        #: where accuracy could land once every unfinished item is resolved. Read
        #: it only to triage a budget_limited row: a bracket inside the band means
        #: a bigger budget settles the question, a bracket spanning it means the
        #: re-run is a coin flip and the GPU hours are better spent elsewhere.
        "budget_bracket": list(budget_bracket(acc, budget_signal)),
        "bracket_in_band": bracket_in_band(acc, budget_signal),
        "empty_rate": round(sum(g["empty"] for g in graded) / n_graded, 4),
        # thinking models overrun the budget before the tag; without this column a
        # truncation-limited score reads as a capability score
        "no_answer_tag_rate": round(no_tag_rate, 4),
        #: the endpoint's own finish_reason — the truth the tag rate approximates
        "truncated_rate": round(truncated_rate, 4),
        #: dead requests, kept separate so they are never read as wrong answers
        "error_rate": round(error_rate, 4),
        "error_kinds": error_kinds,
        "mean_output_chars": round(
            sum(g["chars"] for g in graded) / n_graded, 1
        ),
        "seconds": round(time.time() - started, 1),
        "max_tokens": spec.max_tokens,
        "sampling": {"temperature": temperature, **(sampling or {})},
        "note": spec.note,
        "samples": graded[:5],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="band_results.json")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--temperature", type=float, default=SAMPLING["temperature"])
    ap.add_argument("--greedy", action="store_true",
                    help="temperature 0 with no top_p/top_k. A thinking model loops to the token cap under greedy decoding; see SAMPLING.")
    ap.add_argument("--only", default="", help="comma-separated spec names")
    ap.add_argument("--max-tokens", type=int, default=0,
                    help="override every spec's budget. Use to settle a "
                         "budget_limited verdict without forking the spec "
                         "table; the value used is recorded per result.")
    args = ap.parse_args()

    wanted = [s.strip() for s in args.only.split(",") if s.strip()]
    by_name = {s.name: s for s in SPECS}
    unknown = [w for w in wanted if w not in by_name]
    if unknown:
        raise SystemExit(f"unknown spec(s): {', '.join(unknown)}")
    # Honour the ORDER given: results are written after every spec, so putting
    # the rungs that bracket the ladder first makes a long sweep readable — and
    # abortable — before it finishes.
    specs = [by_name[w] for w in wanted] if wanted else list(SPECS)
    if args.max_tokens:
        import dataclasses as _dc
        specs = [_dc.replace(s, max_tokens=args.max_tokens) for s in specs]

    sampling = None if args.greedy else {
        k: v for k, v in SAMPLING.items() if k != "temperature"
    }
    if args.greedy:
        args.temperature = 0.0
    print(f"sampling: temperature={args.temperature} {sampling or '(greedy)'}")

    results: list[dict] = []
    out_path = args.out
    for spec in specs:
        print(f"[{spec.name}] running...", flush=True)
        result = run_spec(spec, args.n, args.concurrency, args.temperature,
                          sampling=sampling)
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
