"""Contract checks for the LLM band-probe harness.

The graders are where silent wrongness lives in this harness: a grader that
under-reports bins a usable dataset as 'floor' on a number that measured the
grader, not the model. ZebraLogic already shipped that bug once (a single-line
extractor feeding a full-grid grader), so each grader gets a test that would
have caught it.
"""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import pytest

_HARNESS = Path(__file__).resolve().parents[2] / "examples" / "dataset_selection" / "llm_band_probe"


def _load(name: str):
    pytest.importorskip("requests")
    sys.path.insert(0, str(_HARNESS))
    try:
        spec = importlib.util.spec_from_file_location(name, _HARNESS / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        if str(_HARNESS) in sys.path:
            sys.path.remove(str(_HARNESS))


@pytest.fixture(scope="module")
def band():
    return _load("band_locate")


# ── spec table ────────────────────────────────────────────────────────────────
def test_spec_names_are_unique(band):
    names = [s.name for s in band.SPECS]
    assert len(names) == len(set(names))


def test_every_spec_has_a_chapter_and_a_budget(band):
    for spec in band.SPECS:
        assert spec.chapter and spec.dataset and spec.split
        assert spec.max_tokens >= 1024


def test_multiline_gold_specs_grade_the_raw_output(band):
    """extract_answer is structurally single-line, so a grader that parses a
    multi-line gold must be fed the whole generation."""
    for spec in band.SPECS:
        if spec.grader in (band._grade_zebra, band._grade_structural):
            assert spec.grades_raw_output, f"{spec.name} would grade one line of a grid"


# ── graders ───────────────────────────────────────────────────────────────────
_ZEBRA_GOLD = (
    "House 1: Name=Arnold, Color=white\n"
    "House 2: Name=Peter, Color=yellow\n"
    "House 3: Name=Eric, Color=red"
)


def test_zebra_grader_accepts_a_correct_grid_in_either_layout(band):
    assert band._grade_zebra(f"reasoning\nAnswer:\n{_ZEBRA_GOLD}", _ZEBRA_GOLD)
    one_line = (
        "Answer: House 1: Name=Arnold, Color=white; House 2: Name=Peter, "
        "Color=yellow; House 3: Name=Eric, Color=red"
    )
    assert band._grade_zebra(one_line, _ZEBRA_GOLD)


def test_zebra_grader_reads_the_stated_answer_not_the_chain(band):
    """A 2*2 puzzle has two possible assignments and the chain enumerates both.
    Grading the whole generation would pass every run regardless of what the
    model concluded."""
    chain = (
        "Maybe House 1: Name=Arnold, Color=white and House 2: Name=Peter, "
        "Color=yellow and House 3: Name=Eric, Color=red. No, that is wrong.\n"
        "Answer:\nHouse 1: Name=Peter, Color=white\n"
        "House 2: Name=Arnold, Color=yellow\nHouse 3: Name=Eric, Color=red"
    )
    assert not band._grade_zebra(chain, _ZEBRA_GOLD)
    # truncated before it ever stated an answer -> a failure, not a search
    assert not band._grade_zebra(f"thinking... {_ZEBRA_GOLD} ...still thinking",
                                 _ZEBRA_GOLD)
    # the LAST marker wins
    assert band._grade_zebra(
        f"my answer: is coming\nAnswer:\n{_ZEBRA_GOLD}", _ZEBRA_GOLD
    )


def test_structural_grader_does_not_mine_the_middle_of_the_chain(band):
    gold = "[[1, 2], [3, 4]]"
    filler = "reasoning. " * 400
    assert not band._grade_structural(f"[[1, 2], [3, 4]]{filler}no idea", gold)


def test_zebra_grader_binds_cells_to_houses(band):
    """Every value right but assigned to the wrong house is the whole puzzle."""
    swapped = (
        "House 1: Name=Peter, Color=yellow\n"
        "House 2: Name=Arnold, Color=white\n"
        "House 3: Name=Eric, Color=red"
    )
    assert not band._grade_zebra(swapped, _ZEBRA_GOLD)
    assert not band._grade_zebra("House 1: Name=Arnold, Color=white", _ZEBRA_GOLD)


def test_latex_grader_collapses_notation_but_not_meaning(band):
    same = [
        (r"Answer: $\dfrac{\pi}{3}$", r"\frac{\pi}{3}"),
        (r"so \boxed{\frac{\pi}{3}}", r"\frac{\pi}{3}"),
        (r"Answer: 2\sqrt {2}", r"2 \sqrt{2}"),
        ("Answer: 18", "18"),
        (r"Answer: \leftarrow x", r"\leftarrow x"),  # \left must not eat \leftarrow
    ]
    for pred, gold in same:
        assert band._grade_latex(pred, gold), (pred, gold)
    assert not band._grade_latex(r"Answer: \frac{\pi}{6}", r"\frac{\pi}{3}")
    # documented miss: this is a surface-form matcher, not a CAS
    assert not band._grade_latex("Answer: 0.5", r"\frac{1}{2}")


def test_alias_grader_accepts_any_listed_surface_form(band):
    gold = ["Alfredo Stroessner's Paraguay", "Alfredo Stroessner"]
    assert band._grade_aliases("Answer: Alfredo Stroessner", gold)
    assert not band._grade_aliases("Answer: Someone Else", gold)


# ── band classification ───────────────────────────────────────────────────────
def test_band_classification_uses_the_interval_not_the_point(band):
    lo, hi = band.wilson(30, 60)
    assert band.band_of(0.5, lo, hi) == "USABLE"
    assert band.band_of(58 / 60, *band.wilson(58, 60)) == "saturated"
    assert band.band_of(2 / 60, *band.wilson(2, 60)) == "floor"


def test_truncation_short_circuits_the_band(band):
    """A tag-less majority means the budget was measured, not the model."""
    lo, hi = band.wilson(30, 60)
    assert band.band_of(0.5, lo, hi, 0.0) == "USABLE"
    assert band.band_of(0.5, lo, hi, 0.5) == "budget_limited"


def _hotpot_row():
    return {
        "question": "Which magazine was started first, A or B?",
        "answer": "A Magazine",
        "context": {
            "title": ["A Magazine", "Distractor", "B Magazine"],
            "sentences": [["A ran from 1844."], ["Irrelevant."], ["B ran from 1989."]],
        },
    }


def test_hotpot_context_zips_parallel_lists(band):
    """`context` is {title: [...], sentences: [[...]]}, not a list of paragraphs.

    Indexing it the way MuSiQue's `paragraphs` is indexed yields nothing at all,
    and the spec would silently measure closed-book while claiming open-book.
    """
    ctx = band._hotpot_context(_hotpot_row())
    assert "A Magazine: A ran from 1844." in ctx
    assert "B Magazine: B ran from 1989." in ctx
    assert ctx.count("\n\n") == 2  # three paragraphs, two separators


def test_hotpot_open_and_closed_differ_only_in_context(band):
    row = _hotpot_row()
    open_prompt, open_gold = band._adapter_hotpot(row)
    closed_prompt, closed_gold = band._adapter_hotpot_closed(row)
    assert open_gold == closed_gold == "A Magazine"
    assert closed_prompt == row["question"]
    assert "A ran from 1844." in open_prompt
    assert "A ran from 1844." not in closed_prompt
    # a row without usable paragraphs must drop, not silently become closed-book
    assert band._adapter_hotpot({**row, "context": {}}) is None
    assert band._adapter_hotpot_closed({**row, "context": {}}) is not None


def test_scientific_notation_survives_the_notation_gap(band):
    """Minerva writes golds as `4.5e33`; a model writing maths writes LaTeX.

    23.5% of minervamath's golds are program-form scientific notation, so
    surface comparison marked correct physics answers wrong.
    """
    for pred in (
        r"Answer: 4.5 \times 10^{33}",
        r"Answer: 4.5 \cdot 10^{33}",
        "Answer: 4.5 x 10^33",
        "Answer: 4.5e33",
        "reasoning\nAnswer: $4.5 \\times 10^{33}$",
    ):
        assert band._grade_latex(pred, "4.5e33"), pred
    assert not band._grade_latex(r"Answer: 9.1 \times 10^{33}", "4.5e33")
    # a negative exponent is the same notation, not a different case
    assert band._grade_latex(r"Answer: 6.6 \times 10^{-27}", "6.6e-27")


def test_rounding_tolerance_is_confined_to_scientific_notation(band):
    """2-3 sig figs earn slack in the last digit; a plain integer does not."""
    # 4.47e33 vs 4.5e33 is the same derivation rounded differently
    assert band._grade_latex(r"Answer: 4.47 \times 10^{33}", "4.5e33")
    # but 99.5 is simply not 100, and competition golds are exact
    assert not band._grade_latex("Answer: 99.5", "100")
    assert not band._grade_latex("Answer: 1.59", "1.6")


def test_budget_bracket_only_ever_reaches_upward(band):
    """Resolving an unfinished item can add a correct answer, never remove one."""
    assert band.budget_bracket(0.32, 0.12) == (0.32, 0.44)
    assert band.budget_bracket(0.32, 0.0) == (0.32, 0.32)
    # the ceiling is a probability, not an unbounded sum
    assert band.budget_bracket(0.38, 0.72) == (0.38, 1.0)


def test_bracket_flags_which_budget_limited_rows_a_rerun_would_settle(band):
    # minervamath: 0.320 at 12% truncation cannot climb past 0.440
    assert band.bracket_in_band(0.32, 0.12)
    # olymmath_en_hard: 0.180 at 62% could land anywhere — a rerun is a coin flip
    assert not band.bracket_in_band(0.18, 0.62)
    # a ceiling landing exactly on the edge still counts as inside
    assert band.bracket_in_band(0.60, 0.10)
    assert not band.bracket_in_band(0.60, 0.11)


def test_bracket_does_not_overturn_the_budget_veto(band):
    """Band position and label quality are two conditions, not one.

    mmlu_pro measured 0.320 at a 36% tag-less rate: the bracket (0.32, 0.68)
    cannot leave the band, yet over a third of its FAIL labels are budget
    artefacts, so M2 would attribute truncation to capability.
    """
    lo, hi = band.wilson(16, 50)
    assert band.bracket_in_band(0.32, 0.36)
    assert band.band_of(0.32, lo, hi, 0.36) == "budget_limited"


def test_wilson_interval_brackets_the_point_estimate(band):
    for k, n in [(0, 50), (1, 50), (25, 50), (49, 50), (50, 50)]:
        lo, hi = band.wilson(k, n)
        assert 0.0 <= lo <= k / n <= hi <= 1.0


# ── adapters (offline: shaped like the real rows, no network) ─────────────────
def test_polymath_adapter_strips_the_latex_delimiters(band):
    row = {"id": "medium-en-0", "question": "q", "answer": r"$\frac{\pi}{3}$"}
    assert band._adapter_polymath(row) == ("q", r"\frac{\pi}{3}")
    assert band._adapter_polymath({"question": "", "answer": "1"}) is None


def test_mc_adapter_drops_rows_with_more_options_than_letters(band):
    adapter = band._adapter_mc(("question",), "options", "answer_letter")
    row = {"question": "q", "options": [f"o{i}" for i in range(10)], "answer_letter": "J"}
    prompt, gold = adapter(row)
    assert gold == "J" and "J. o9" in prompt
    too_many = dict(row, options=[f"o{i}" for i in range(11)])
    assert adapter(too_many) is None


# ── decoding + endpoint contract ──────────────────────────────────────────────
def test_generate_reports_the_finish_reason(band, monkeypatch):
    """The endpoint's finish_reason is the truncation signal the band rule
    depends on; no_answer_tag_rate was only ever a proxy for it."""

    class _Resp:
        status_code = 200
        def raise_for_status(self): pass
        def json(self):
            return {"choices": [{"message": {"content": "hi"},
                                 "finish_reason": "length"}]}

    monkeypatch.setattr(band.requests, "post", lambda *a, **k: _Resp())
    assert band.generate("p", 16) == ("hi", "length")


def test_generate_marks_a_dead_request_as_an_error_not_an_empty_answer(band, monkeypatch):
    monkeypatch.setattr(band.time, "sleep", lambda s: None)

    def _boom(*a, **k):
        raise band.requests.RequestException("down")

    monkeypatch.setattr(band.requests, "post", _boom)
    text, reason = band.generate("p", 16, retries=2)
    # the CAUSE is carried, not just the fact: a read timeout, a refused
    # connection and a malformed body need different fixes
    assert text == "" and reason == "error:RequestException"


def test_generate_passes_sampling_through(band, monkeypatch):
    sent = {}

    class _Resp:
        status_code = 200
        def raise_for_status(self): pass
        def json(self):
            return {"choices": [{"message": {"content": "x"},
                                 "finish_reason": "stop"}]}

    def _post(url, json=None, timeout=None):
        sent.update(json)
        return _Resp()

    monkeypatch.setattr(band.requests, "post", _post)
    band.generate("p", 16, sampling={"top_p": 0.95, "top_k": 20})
    assert sent["top_p"] == 0.95 and sent["top_k"] == 20


def test_generate_sends_thinking_off_unless_opted_in(band, monkeypatch):
    """Every request carries chat_template_kwargs.enable_thinking explicitly: the
    Qwen3.5-2B template defaults OFF and the 9B template defaults ON when the
    kwarg is absent, so only an explicit value renders the same everywhere."""
    sent = {}

    class _Resp:
        status_code = 200
        def raise_for_status(self): pass
        def json(self):
            return {"choices": [{"message": {"content": "x"},
                                 "finish_reason": "stop"}]}

    def _post(url, json=None, timeout=None):
        sent.clear()
        sent.update(json)
        return _Resp()

    monkeypatch.setattr(band.requests, "post", _post)
    monkeypatch.setattr(band, "ENABLE_THINKING", False)
    band.generate("p", 16)
    assert sent["chat_template_kwargs"] == {"enable_thinking": False}
    band.generate("p", 16, enable_thinking=True)  # per-call opt-in
    assert sent["chat_template_kwargs"] == {"enable_thinking": True}
    monkeypatch.setattr(band, "ENABLE_THINKING", True)  # module-wide opt-in
    band.generate("p", 16)
    assert sent["chat_template_kwargs"] == {"enable_thinking": True}


def test_thinking_env_opt_in_is_off_by_default(monkeypatch):
    """BAND_ENABLE_THINKING=1 is the only way the band probe turns thinking on."""
    import importlib.util
    from pathlib import Path

    path = Path(__file__).parents[2] / "examples/dataset_selection/llm_band_probe/band_locate.py"

    def _fresh():
        import sys

        spec = importlib.util.spec_from_file_location("_band_fresh", path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules["_band_fresh"] = mod  # @dataclass resolves the module by name
        try:
            spec.loader.exec_module(mod)
        finally:
            sys.modules.pop("_band_fresh", None)
        return mod

    monkeypatch.delenv("BAND_ENABLE_THINKING", raising=False)
    assert _fresh().ENABLE_THINKING is False
    monkeypatch.setenv("BAND_ENABLE_THINKING", "1")
    assert _fresh().ENABLE_THINKING is True


def test_default_sampling_is_not_greedy(band):
    """Greedy decoding sends this model into verbatim self-verification loops
    that run to the token cap on puzzles it has already solved."""
    assert band.SAMPLING["temperature"] > 0
    assert "top_p" in band.SAMPLING


def test_probe_validate_unpacks_the_generate_tuple(monkeypatch):
    """generate() returns (text, finish_reason); a caller that forgets to unpack
    hands every probe a tuple and every grader scores 0."""
    pytest.importorskip("evalrx")
    band = _load("band_locate")
    pv = _load("probe_validate")
    monkeypatch.setattr(band, "generate", lambda *a, **k: ("the text", "length"))
    monkeypatch.setattr(pv.B, "generate", lambda *a, **k: ("the text", "length"))
    model = pv.EndpointModel()

    class _In:
        prompt = "q"

    assert model.generate(_In()) == "the text"
    assert model.n_truncated == 1


# ── ladder slicing ────────────────────────────────────────────────────────────
def test_zebra_tiers_partition_every_grid_size_exactly_once(band):
    seen = [size for _, sizes in band.ZEBRA_TIERS for size in sizes]
    assert sorted(seen) == sorted(band._ZEBRA_SIZES)
    assert len(seen) == len(set(seen)) == 25


def test_zebra_tiers_are_ordered_by_difficulty(band):
    """The rungs only mean anything if tier N is uniformly harder than N-1."""
    spans = [
        (min(map(band._zebra_logspace, s)), max(map(band._zebra_logspace, s)))
        for _, s in band.ZEBRA_TIERS
    ]
    for (_, prev_hi), (next_lo, _) in zip(spans, spans[1:]):
        assert prev_hi < next_lo


def test_zebra_logspace_counts_assignments(band):
    # 3 houses, 2 attributes -> (3!)^2 = 36
    assert band._zebra_logspace("3*2") == pytest.approx(math.log10(36))
    assert band._zebra_logspace("2*2") < band._zebra_logspace("6*6")


def test_row_filter_keeps_only_the_named_rung(band):
    keep = band._row_in("size", ["3*3", "4*2"])
    assert keep({"size": "3*3"}) and keep({"size": "4*2"})
    assert not keep({"size": "6*6"}) and not keep({})


def _fake_server(monkeypatch, band, rows, total=None):
    """Serve `rows` through a paged, offset-addressed stub of /rows."""
    total = len(rows) if total is None else total
    calls = []

    def _get(params, timeout, retries=5, url=None):
        calls.append({**params, "_url": url})
        off, length = params["offset"], params["length"]
        page = rows[off:off + length]
        return {
            "num_rows_total": total,
            "rows": [{"row_idx": off + i, "row": r} for i, r in enumerate(page)],
        }

    monkeypatch.setattr(band, "_get_rows", _get)
    return calls


def test_where_clause_routes_to_the_filter_endpoint(band, monkeypatch):
    """A server-side slice must be addressed through /filter, not sieved from /rows.

    SuperGPQA Medicine/hard is 217 of 26,529 rows; pulling the split and
    discarding 99% would be ~6,000 wasted rows per sweep and a 429 magnet.
    """
    rows = [{"discipline": "Law", "id": i} for i in range(400)]
    calls = _fake_server(monkeypatch, band, rows)
    spec = band.Spec("law", "ch4", "m-a-p/SuperGPQA", split="train",
                     where="\"discipline\"='Law'")
    got = band.fetch_rows(spec, 40)

    assert len(got) >= 40
    assert all(c["_url"] == band.FILTER_API for c in calls)
    assert all(c["where"] == "\"discipline\"='Law'" for c in calls)


def test_no_where_clause_still_uses_the_rows_endpoint(band, monkeypatch):
    rows = [{"id": i} for i in range(400)]
    calls = _fake_server(monkeypatch, band, rows)
    spec = band.Spec("plain", "ch4", "some/dataset", split="train")
    band.fetch_rows(spec, 40)

    assert all(c["_url"] == band.ROWS_API for c in calls)
    assert all("where" not in c for c in calls)


def test_fetch_rows_applies_the_filter_and_counts_only_survivors(band, monkeypatch):
    """The early-stop must count KEPT rows.

    Counting raw rows would stop a 1-in-25 slice after a handful of usable
    items and still report a band on them.
    """
    rows = [{"size": f"{i % 25}", "id": i} for i in range(1000)]
    _fake_server(monkeypatch, band, rows)
    spec = band.Spec(
        "t", "ch3", "ds", row_filter=band._row_in("size", ["7"]),
        n_windows=10, fetch_multiplier=2,
    )
    got = band.fetch_rows(spec, 20)
    assert got, "filter returned nothing"
    assert {r["size"] for r in got} == {"7"}
    assert len(got) >= 20


def test_fetch_rows_deduplicates_overlapping_windows(band, monkeypatch):
    rows = [{"size": "a", "id": i} for i in range(300)]
    _fake_server(monkeypatch, band, rows)
    spec = band.Spec("t", "ch3", "ds", row_filter=band._row_in("size", ["a"]),
                     n_windows=12, fetch_multiplier=20)
    got = band.fetch_rows(spec, 40)
    assert len(got) == len({r["id"] for r in got})


def test_fetch_rows_survives_a_dead_window_and_counts_it(band, monkeypatch):
    rows = [{"size": "a", "id": i} for i in range(500)]
    _fake_server(monkeypatch, band, rows)
    real = band._get_rows
    state = {"n": 0}

    def _flaky(params, timeout, retries=5, url=None):
        state["n"] += 1
        if state["n"] % 3 == 0 and params["length"] > 1:
            return None  # exhausted its retries
        return real(params, timeout, retries)

    monkeypatch.setattr(band, "_get_rows", _flaky)
    spec = band.Spec("t", "ch3", "ds", row_filter=band._row_in("size", ["a"]),
                     n_windows=10, fetch_multiplier=20)
    got = band.fetch_rows(spec, 40)
    assert got
    assert band.fetch_rows.last_failed_windows > 0


def test_fetch_rows_raises_rather_than_reporting_an_empty_split(band, monkeypatch):
    """A dead metadata call must not look like a dataset with no rows."""
    monkeypatch.setattr(band, "_get_rows", lambda *a, **k: None)
    with pytest.raises(RuntimeError):
        band.fetch_rows(band.Spec("t", "ch3", "ds"), 10)


def test_get_rows_retries_rate_limits_then_gives_up(band, monkeypatch):
    attempts = []

    class _Resp:
        status_code = 429
        def json(self): return {}

    monkeypatch.setattr(band.time, "sleep", lambda s: None)
    monkeypatch.setattr(
        band.requests, "get", lambda *a, **k: (attempts.append(1), _Resp())[1]
    )
    assert band._get_rows({"offset": 0, "length": 1}, 5, retries=4) is None
    assert len(attempts) == 4


def test_get_rows_does_not_retry_a_client_error(band, monkeypatch):
    attempts = []

    class _Resp:
        status_code = 404
        def json(self): return {}

    monkeypatch.setattr(
        band.requests, "get", lambda *a, **k: (attempts.append(1), _Resp())[1]
    )
    assert band._get_rows({"offset": 0, "length": 1}, 5, retries=4) is None
    assert len(attempts) == 1


def test_every_sliced_spec_declares_enough_windows_to_reach_its_rows(band):
    """A filtered spec sampling 12 windows of a 4758-row split sees a quarter of
    it; if its rung lives outside those windows the spec silently returns
    nothing."""
    for spec in band.SPECS:
        if spec.row_filter is None:
            continue
        assert spec.fetch_multiplier >= 3, spec.name
        assert spec.n_windows >= 10, spec.name


# ── Enigmata grading ──────────────────────────────────────────────────────────
def test_structural_grader_ignores_formatting_but_not_content(band):
    gold = "[[3, 1, 2], [7, 8, 6]]"
    for pred in [
        "Answer: [[3, 1, 2], [7, 8, 6]]",
        "Answer: [[3,1,2],[7,8,6]]",
        "reasoning...\nAnswer:\n[[3, 1, 2],\n [7, 8, 6]]",
    ]:
        assert band._grade_structural(pred, gold), pred
    assert not band._grade_structural("Answer: [[3, 1, 2], [7, 6, 8]]", gold)
    assert not band._grade_structural("Answer: [[3, 1, 2]]", gold)


def test_structural_grader_reads_the_LAST_matrix_not_a_working_one(band):
    """The chain is full of discarded intermediate grids; only the final one is
    the answer."""
    gold = "[[1, 2], [3, 4]]"
    text = "first try [[9, 9], [9, 9]] no wait\nAnswer: [[1, 2], [3, 4]]"
    assert band._grade_structural(text, gold)
    assert not band._grade_structural(
        "Answer: [[1, 2], [3, 4]]\nactually no, [[9, 9], [9, 9]]", gold
    )


def test_structural_grader_falls_back_for_scalar_golds(band):
    """A gold with no brackets is a short-string task; grading it structurally
    would return False for every row."""
    assert band._grade_structural("Answer: NO", "NO")
    assert band._grade_structural("so the answer is\nAnswer: False", "False")
    assert not band._grade_structural("Answer: YES", "NO")


def test_structural_grader_survives_adversarial_output(band):
    gold = "[[1, 2]]"
    for junk in ["", "[[[[[[", "]]]]]]", "Answer: [1, 2", "[" * 5000,
                 "Answer: [__import__('os')]", "[[1, 2]" + "]" * 100]:
        assert band._grade_structural(junk, gold) in (True, False)


def test_structural_grader_accepts_the_fenced_grid_the_dataset_asks_for(band):
    """Enigmata's own prompt says "output a grid of numbers in a code block",
    while its gold is a JSON matrix. A bracket-only grader scores every correct
    fenced answer wrong — which is most of the split."""
    gold = "[[3, 4, 1, 2], [1, 2, 3, 4]]"
    fenced = "reasoning here\n```\n3 4 1 2\n1 2 3 4\n```"
    assert band._grade_structural(fenced, gold)
    assert not band._grade_structural("```\n3 4 1 2\n1 2 4 3\n```", gold)
    # a flat path must still be read as a list, not as a one-row grid
    assert band._grade_structural("```\n[0, 2, 3, 1, 0]\n```", "[0, 2, 3, 1, 0]")
    # scalar gold inside the fence
    assert band._grade_structural("```\nNO\n```", "NO")
    assert not band._grade_structural("```\nYES\n```", "NO")


def test_structural_grader_reads_the_last_fence_not_a_worked_example(band):
    gold = "[[1, 2], [3, 4]]"
    text = "like this:\n```\n9 9\n9 9\n```\nso the answer is\n```\n1 2\n3 4\n```"
    assert band._grade_structural(text, gold)


def test_specs_that_ship_their_own_format_do_not_get_a_second_one(band):
    """Two format orders in one prompt is what made the model loop to the token
    cap instead of answering."""
    for spec in band.SPECS:
        if spec.dataset.endswith("Enigmata-Eval") and spec.row_filter is not None:
            assert not spec.append_instruction, spec.name
        if spec.name.startswith("zebra_"):
            assert not spec.append_instruction, spec.name


def test_zebra_instruction_names_this_puzzles_attributes(band):
    """A fixed 'Name=..., Color=...' example on a puzzle with no Color attribute
    contradicts 'use the attribute names from the puzzle', and the model
    oscillates between them until the budget runs out."""
    text = band._zebra_instruction(["House", "Name", "CarModel"])
    assert "CarModel=<value>" in text and "Name=<value>" in text
    assert "Color" not in text


def test_zebra_adapter_puts_the_format_order_in_the_prompt(band):
    row = {
        "puzzle": "There are 2 houses...",
        "solution": {
            "header": ["House", "Name", "CarModel"],
            "rows": [["1", "Arnold", "ford f150"], ["2", "Eric", "tesla model 3"]],
        },
    }
    prompt, gold = band._adapter_zebra(row)
    assert "CarModel=<value>" in prompt and "There are 2 houses" in prompt
    assert gold == (
        "House 1: Name=Arnold, CarModel=ford f150\n"
        "House 2: Name=Eric, CarModel=tesla model 3"
    )
    assert band._grade_zebra(f"Answer:\n{gold}", gold)


def test_zebra_grader_ignores_attribute_label_spacing(band):
    """The attribute is a column label; a model penalised for writing
    'Car Model' would be scored on typography, not on the puzzle."""
    gold = "House 1: CarModel=ford f150"
    assert band._grade_zebra("Answer:\nHouse 1: Car Model=ford f150", gold)
    assert not band._grade_zebra("Answer:\nHouse 1: CarModel=tesla model 3", gold)


def test_every_enigmata_task_is_classified_exactly_once(band):
    """A task that is in no bucket is silently dropped from every slice."""
    buckets = [band._ENIGMATA_STRUCTURAL, band._ENIGMATA_SHORT,
               band._ENIGMATA_UNGRADED]
    seen = [t for b in buckets for t in b]
    assert len(seen) == len(set(seen)), "a task is in two buckets"
    # the 36 tasks measured across the full split
    assert len(seen) == 36


def test_enigmata_filter_drops_certificate_answers(band):
    """A Hamiltonian cycle can start at any vertex and run either way, so exact
    match against the one stored cycle marks correct answers wrong. Only the
    decision instances of those tasks are uniquely gradable."""
    keep = band._enigmata_filter(tasks=band._ENIGMATA_SHORT)
    assert keep({"task_name": "hamiltonian_cycle", "answer": "NO"})
    assert not keep({"task_name": "hamiltonian_cycle", "answer": "[0, 2, 3, 1, 0]"})
    # a task with a genuinely unique gold is untouched
    assert keep({"task_name": "FOLIO", "answer": "False"})


def test_enigmata_filter_reads_difficulty_out_of_the_meta_blob(band):
    keep = band._enigmata_filter(difficulty="easy")
    assert keep({"task_name": "sudoku", "meta": '{"difficulty": "easy"}'})
    assert keep({"task_name": "sudoku", "meta": {"difficulty": "easy"}})
    assert not keep({"task_name": "sudoku", "meta": '{"difficulty": "hard"}'})
    # an excluded task never enters a rung, whatever its difficulty
    assert not keep({"task_name": "zebra_logic", "meta": '{"difficulty": "easy"}'})
    # unparseable meta must not raise
    assert not keep({"task_name": "sudoku", "meta": "not json"})
    assert not keep({"task_name": "sudoku"})


# ── LiveBench-Math ────────────────────────────────────────────────────────────
def test_livebench_grader_compares_the_olympiad_answer_as_an_ORDERING(band):
    """The olympiad task answers with an ordering of expression identifiers;
    the order IS the answer, so a set-match would pass a wrong permutation."""
    gold = "1,6,7,2,3,4,5"
    assert band._grade_livebench_math("Answer: 1,6,7,2,3,4,5", gold)
    assert band._grade_livebench_math("Answer: 1, 6, 7, 2, 3, 4, 5", gold)
    assert not band._grade_livebench_math("Answer: 1,7,6,2,3,4,5", gold)  # permuted
    assert not band._grade_livebench_math("Answer: 1,6,7,2,3,4", gold)    # short


def test_livebench_grader_falls_through_to_latex_and_numeric(band):
    assert band._grade_livebench_math(r"so \boxed{63 \sqrt[3]{2}}", r"63 \sqrt[3]{2}")
    # math_comp golds are zero-padded three-digit strings
    assert band._grade_livebench_math("the answer is 025", "025")
    assert not band._grade_livebench_math("the answer is 026", "025")


def test_livebench_adapter_takes_the_first_turn_as_the_prompt(band):
    row = {"turns": ["Compute the geometric mean of ${8, -10}$."],
           "ground_truth": r"4 i \sqrt{5}", "task": "AMPS_Hard"}
    prompt, gold = band._adapter_livebench_math(row)
    assert prompt.startswith("Compute") and gold == r"4 i \sqrt{5}"
    assert band._adapter_livebench_math({"turns": [], "ground_truth": "x"}) is None


def test_livebench_live_filter_keeps_only_unretired_questions(band):
    assert band._still_live({"livebench_removal_date": ""})
    assert band._still_live({})
    assert not band._still_live({"livebench_removal_date": "2025-04-02T00:00:00"})


def test_livebench_specs_do_not_add_a_fourth_format_order(band):
    """Each item ships its own answer-format instruction inside turns[0]."""
    for spec in band.SPECS:
        if spec.dataset == "livebench/math":
            assert not spec.append_instruction, spec.name


def test_imo_answerbench_adapter_handles_spaced_capitalised_fields(band):
    row = {"Problem ID": "imo-bench-algebra-001", "Problem": "For a given N...",
           "Short Answer": "3", "Category": "Algebra"}
    assert band._adapter_imo_answerbench(row) == ("For a given N...", "3")
    assert band._adapter_imo_answerbench({"Problem": "", "Short Answer": "3"}) is None


def test_musr_adapter_parses_string_repr_choices(band):
    row = {
        "narrative": "n", "question": "Who?",
        "choices": "['Mackenzie', 'Ana']", "answer_choice": "Ana",
    }
    prompt, gold = band._adapter_musr(row)
    assert gold == "B" and "A. Mackenzie" in prompt


# ----------------------------------------------------------------------
# Hub-file fallback for a datasets-server that will not serve the slice
# ----------------------------------------------------------------------
def test_parse_where_accepts_the_spec_shapes_and_rejects_anything_wider(band):
    assert band.parse_where("\"discipline\"='Law'") == [("discipline", "Law")]
    assert band.parse_where("\"discipline\"='Medicine' AND \"difficulty\"='hard'") == [
        ("discipline", "Medicine"), ("difficulty", "hard")]
    assert band.parse_where("\"a\"='x'  and  \"b\"='y'") == [("a", "x"), ("b", "y")]
    for wider in ("\"discipline\"='Law' OR \"discipline\"='Economics'",
                  "\"difficulty\"!='easy'", "discipline='Law'", "\"discipline\" LIKE 'L%'"):
        with pytest.raises(ValueError, match="unsupported where clause"):
            band.parse_where(wider)


def test_hub_fallback_draws_the_sample_the_server_would_have(band, monkeypatch):
    """/filter at HTTP 500 must not turn the freeze into a different (or wider) sample.

    The server pages through the slice in file order; the fallback filters the
    split locally and hands the same seeded window draw the same slice.
    """
    split = [{"discipline": ("Law", "Economics", "Medicine")[i % 3],
              "difficulty": "hard" if i % 4 == 0 else "easy", "id": i} for i in range(2000)]
    spec = band.Spec("medhard", "ch4", "m-a-p/SuperGPQA", split="train",
                     where="\"discipline\"='Medicine' AND \"difficulty\"='hard'")
    sliced = [r for r in split if r["discipline"] == "Medicine" and r["difficulty"] == "hard"]
    _fake_server(monkeypatch, band, sliced)
    via_server = band.fetch_rows(spec, 40, seed=3)

    monkeypatch.setattr(band, "_load_hub_rows", lambda s: list(split))   # whole split, unfiltered
    via_hub = band.fetch_rows_hub(spec, 40, seed=3)

    assert len(via_hub) >= 40
    assert via_hub == via_server
    assert all(r["discipline"] == "Medicine" and r["difficulty"] == "hard" for r in via_hub)
    assert band.fetch_rows.last_failed_windows == 0


def test_hub_fallback_without_a_where_clause_is_the_plain_draw(band, monkeypatch):
    split = [{"id": i} for i in range(300)]
    spec = band.Spec("plain", "ch4", "some/dataset", split="train")
    monkeypatch.setattr(band, "_load_hub_rows", lambda s: list(split))
    via_hub = band.fetch_rows_hub(spec, 40, seed=0)
    _fake_server(monkeypatch, band, split)
    assert via_hub == band.fetch_rows(spec, 40, seed=0)


def test_hub_fallback_keeps_the_row_filter_and_counts_survivors(band, monkeypatch):
    split = [{"size": f"{i % 25}", "id": i} for i in range(1000)]
    spec = band.Spec("rung", "ch4", "some/dataset", split="train",
                     row_filter=band._row_in("size", ["3"]))
    monkeypatch.setattr(band, "_load_hub_rows", lambda s: list(split))
    got = band.fetch_rows_hub(spec, 20, seed=0)
    assert got and all(r["size"] == "3" for r in got)
    assert len(got) >= 20
