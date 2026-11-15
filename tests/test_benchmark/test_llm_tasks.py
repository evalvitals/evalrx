"""The text-LLM benchmark tasks: slice specs, graders, row fetching, catalog.

Everything here runs offline — the datasets-server and the hub are stubbed.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

from evalrx.benchmark.tasks import llm
from evalrx.benchmark.tasks.llm import catalog, fetch
from evalrx.benchmark.tasks.llm.specs import SPECS, Spec, grade_latex, mc_adapter


def _row_in(column: str, allowed):
    allowed = set(allowed)
    return lambda row: row.get(column) in allowed


# ── registry ──────────────────────────────────────────────────────────────────
def test_every_task_has_a_catalog_entry_and_a_spec():
    assert set(llm.DATASETS) == set(SPECS) == set(catalog.BY_NAME)
    assert {t.name for t in llm.TASKS} == set(llm.DATASETS)
    for name in llm.DATASETS:
        assert catalog.get(name).spec is SPECS[name]
        assert SPECS[name].name == name


def test_unknown_dataset_is_a_key_error_naming_the_choices():
    with pytest.raises(KeyError, match="bbh_causal_judgement"):
        catalog.get("no_such_slice")


def test_the_task_registry_imports_without_requests():
    """``requests`` is in the ``data`` extra; the registry must not need it."""
    code = (
        "import sys\n"
        "import evalrx\n"
        "assert 'evalrx.benchmark.tasks' not in sys.modules\n"
        "sys.modules['requests'] = None\n"
        "from evalrx.benchmark.tasks import llm\n"
        "assert llm.grade('bbh_causal_judgement', 'Answer: Yes', 'Yes')\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_grade_uses_the_slice_grader_on_the_extracted_answer():
    assert llm.grade("bbh_causal_judgement", "Reasoning...\nAnswer: Yes", "Yes")
    assert not llm.grade("bbh_causal_judgement", "Answer: No", "Yes")
    assert llm.grade("minervamath", "work\nAnswer: 4.5 \\times 10^{33}", "4.5e33")


# ── graders and adapters ─────────────────────────────────────────────────────
def test_latex_grader_collapses_notation_but_not_meaning():
    same = [
        (r"Answer: $\dfrac{\pi}{3}$", r"\frac{\pi}{3}"),
        (r"so \boxed{\frac{\pi}{3}}", r"\frac{\pi}{3}"),
        (r"Answer: 2\sqrt {2}", r"2 \sqrt{2}"),
        ("Answer: 18", "18"),
        (r"Answer: \leftarrow x", r"\leftarrow x"),  # \left must not eat \leftarrow
    ]
    for pred, gold in same:
        assert grade_latex(pred, gold), (pred, gold)
    assert not grade_latex(r"Answer: \frac{\pi}{6}", r"\frac{\pi}{3}")
    # documented miss: this is a surface-form matcher, not a CAS
    assert not grade_latex("Answer: 0.5", r"\frac{1}{2}")


def test_scientific_notation_survives_the_notation_gap():
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
        assert grade_latex(pred, "4.5e33"), pred
    assert not grade_latex(r"Answer: 9.1 \times 10^{33}", "4.5e33")
    # a negative exponent is the same notation, not a different case
    assert grade_latex(r"Answer: 6.6 \times 10^{-27}", "6.6e-27")


def test_rounding_tolerance_is_confined_to_scientific_notation():
    """2-3 sig figs earn slack in the last digit; a plain integer does not."""
    # 4.47e33 vs 4.5e33 is the same derivation rounded differently
    assert grade_latex(r"Answer: 4.47 \times 10^{33}", "4.5e33")
    # but 99.5 is simply not 100, and competition golds are exact
    assert not grade_latex("Answer: 99.5", "100")
    assert not grade_latex("Answer: 1.59", "1.6")


def test_mc_adapter_drops_rows_with_more_options_than_letters():
    adapter = mc_adapter(("question",), "options", "answer_letter")
    row = {"question": "q", "options": [f"o{i}" for i in range(10)], "answer_letter": "J"}
    prompt, gold = adapter(row)
    assert gold == "J" and "J. o9" in prompt
    too_many = dict(row, options=[f"o{i}" for i in range(11)])
    assert adapter(too_many) is None


# ── fetching (datasets-server stubbed) ───────────────────────────────────────
def _fake_server(monkeypatch, rows, total=None):
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

    monkeypatch.setattr(fetch, "_get_rows", _get)
    return calls


def test_where_clause_routes_to_the_filter_endpoint(monkeypatch):
    """A server-side slice must be addressed through /filter, not sieved from /rows.

    SuperGPQA Medicine/hard is 217 of 26,529 rows; pulling the split and
    discarding 99% would be ~6,000 wasted rows per sweep and a 429 magnet.
    """
    rows = [{"discipline": "Law", "id": i} for i in range(400)]
    calls = _fake_server(monkeypatch, rows)
    spec = Spec("law", "ch4", "m-a-p/SuperGPQA", split="train",
                     where="\"discipline\"='Law'")
    got = fetch.fetch_rows(spec, 40)

    assert len(got) >= 40
    assert all(c["_url"] == fetch.FILTER_API for c in calls)
    assert all(c["where"] == "\"discipline\"='Law'" for c in calls)


def test_no_where_clause_still_uses_the_rows_endpoint(monkeypatch):
    rows = [{"id": i} for i in range(400)]
    calls = _fake_server(monkeypatch, rows)
    spec = Spec("plain", "ch4", "some/dataset", split="train")
    fetch.fetch_rows(spec, 40)

    assert all(c["_url"] == fetch.ROWS_API for c in calls)
    assert all("where" not in c for c in calls)


def test_fetch_rows_applies_the_filter_and_counts_only_survivors(monkeypatch):
    """The early-stop must count KEPT rows.

    Counting raw rows would stop a 1-in-25 slice after a handful of usable
    items and still report a band on them.
    """
    rows = [{"size": f"{i % 25}", "id": i} for i in range(1000)]
    _fake_server(monkeypatch, rows)
    spec = Spec(
        "t", "ch3", "ds", row_filter=_row_in("size", ["7"]),
        n_windows=10, fetch_multiplier=2,
    )
    got = fetch.fetch_rows(spec, 20)
    assert got, "filter returned nothing"
    assert {r["size"] for r in got} == {"7"}
    assert len(got) >= 20


def test_fetch_rows_deduplicates_overlapping_windows(monkeypatch):
    rows = [{"size": "a", "id": i} for i in range(300)]
    _fake_server(monkeypatch, rows)
    spec = Spec("t", "ch3", "ds", row_filter=_row_in("size", ["a"]),
                     n_windows=12, fetch_multiplier=20)
    got = fetch.fetch_rows(spec, 40)
    assert len(got) == len({r["id"] for r in got})


def test_fetch_rows_survives_a_dead_window_and_counts_it(monkeypatch):
    rows = [{"size": "a", "id": i} for i in range(500)]
    _fake_server(monkeypatch, rows)
    real = fetch._get_rows
    state = {"n": 0}

    def _flaky(params, timeout, retries=5, url=None):
        state["n"] += 1
        if state["n"] % 3 == 0 and params["length"] > 1:
            return None  # exhausted its retries
        return real(params, timeout, retries)

    monkeypatch.setattr(fetch, "_get_rows", _flaky)
    spec = Spec("t", "ch3", "ds", row_filter=_row_in("size", ["a"]),
                     n_windows=10, fetch_multiplier=20)
    got = fetch.fetch_rows(spec, 40)
    assert got
    assert fetch.fetch_rows.last_failed_windows > 0


def test_fetch_rows_raises_rather_than_reporting_an_empty_split(monkeypatch):
    """A dead metadata call must not look like a dataset with no rows."""
    monkeypatch.setattr(fetch, "_get_rows", lambda *a, **k: None)
    with pytest.raises(RuntimeError):
        fetch.fetch_rows(Spec("t", "ch3", "ds"), 10)


def test_get_rows_retries_rate_limits_then_gives_up(monkeypatch):
    requests = pytest.importorskip("requests")
    attempts = []

    class _Resp:
        status_code = 429
        def json(self): return {}

    monkeypatch.setattr(fetch.time, "sleep", lambda s: None)
    monkeypatch.setattr(
        requests, "get", lambda *a, **k: (attempts.append(1), _Resp())[1]
    )
    assert fetch._get_rows({"offset": 0, "length": 1}, 5, retries=4) is None
    assert len(attempts) == 4


def test_get_rows_does_not_retry_a_client_error(monkeypatch):
    requests = pytest.importorskip("requests")
    attempts = []

    class _Resp:
        status_code = 404
        def json(self): return {}

    monkeypatch.setattr(
        requests, "get", lambda *a, **k: (attempts.append(1), _Resp())[1]
    )
    assert fetch._get_rows({"offset": 0, "length": 1}, 5, retries=4) is None
    assert len(attempts) == 1


def test_parse_where_accepts_the_spec_shapes_and_rejects_anything_wider():
    assert fetch.parse_where("\"discipline\"='Law'") == [("discipline", "Law")]
    assert fetch.parse_where("\"discipline\"='Medicine' AND \"difficulty\"='hard'") == [
        ("discipline", "Medicine"), ("difficulty", "hard")]
    assert fetch.parse_where("\"a\"='x'  and  \"b\"='y'") == [("a", "x"), ("b", "y")]
    for wider in ("\"discipline\"='Law' OR \"discipline\"='Economics'",
                  "\"difficulty\"!='easy'", "discipline='Law'", "\"discipline\" LIKE 'L%'"):
        with pytest.raises(ValueError, match="unsupported where clause"):
            fetch.parse_where(wider)


def test_hub_fallback_draws_the_sample_the_server_would_have(monkeypatch):
    """/filter at HTTP 500 must not turn the freeze into a different (or wider) sample.

    The server pages through the slice in file order; the fallback filters the
    split locally and hands the same seeded window draw the same slice.
    """
    split = [{"discipline": ("Law", "Economics", "Medicine")[i % 3],
              "difficulty": "hard" if i % 4 == 0 else "easy", "id": i} for i in range(2000)]
    spec = Spec("medhard", "ch4", "m-a-p/SuperGPQA", split="train",
                     where="\"discipline\"='Medicine' AND \"difficulty\"='hard'")
    sliced = [r for r in split if r["discipline"] == "Medicine" and r["difficulty"] == "hard"]
    _fake_server(monkeypatch, sliced)
    via_server = fetch.fetch_rows(spec, 40, seed=3)

    monkeypatch.setattr(fetch, "_load_hub_rows", lambda s: list(split))   # whole split, unfiltered
    via_hub = fetch.fetch_rows_hub(spec, 40, seed=3)

    assert len(via_hub) >= 40
    assert via_hub == via_server
    assert all(r["discipline"] == "Medicine" and r["difficulty"] == "hard" for r in via_hub)
    assert fetch.fetch_rows.last_failed_windows == 0


def test_hub_fallback_without_a_where_clause_is_the_plain_draw(monkeypatch):
    split = [{"id": i} for i in range(300)]
    spec = Spec("plain", "ch4", "some/dataset", split="train")
    monkeypatch.setattr(fetch, "_load_hub_rows", lambda s: list(split))
    via_hub = fetch.fetch_rows_hub(spec, 40, seed=0)
    _fake_server(monkeypatch, split)
    assert via_hub == fetch.fetch_rows(spec, 40, seed=0)


def test_hub_fallback_keeps_the_row_filter_and_counts_survivors(monkeypatch):
    split = [{"size": f"{i % 25}", "id": i} for i in range(1000)]
    spec = Spec("rung", "ch4", "some/dataset", split="train",
                     row_filter=_row_in("size", ["3"]))
    monkeypatch.setattr(fetch, "_load_hub_rows", lambda s: list(split))
    got = fetch.fetch_rows_hub(spec, 20, seed=0)
    assert got and all(r["size"] == "3" for r in got)
    assert len(got) >= 20


def test_rows_without_a_server_index_are_deduplicated_by_content(monkeypatch):
    """The hub-file pager and some servers give no ``row_idx``; identical rows
    from overlapping windows must still count once."""
    rows = [{"id": i} for i in range(300)]

    def _get(params, timeout, retries=5, url=None):
        off, length = params["offset"], params["length"]
        return {"num_rows_total": len(rows),
                "rows": [{"row": r} for r in rows[off:off + length]]}

    monkeypatch.setattr(fetch, "_get_rows", _get)
    got = fetch.fetch_rows(Spec("t", "ch4", "ds", n_windows=12, fetch_multiplier=20), 40)
    assert got and len(got) == len({r["id"] for r in got})
