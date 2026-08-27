"""``<MARKER>=<json>`` result lines must survive a pretty-printed payload.

Live failure (qwen3.5-2b / gsm8k, 2026-08-27): the explorer agent printed
``EXPLORATORY_RESULT_JSON={json.dumps(result, indent=2)}`` in all three attempts.
The parser read only the remainder of the marker line — ``{`` — and the EXPLORE
stage failed with "unparseable ... JSON" although the object was valid. The M2
stats-tool parser had the same line-bound reading.
"""

from __future__ import annotations

import json

from evalvitals.analysis.explorer import _parse_result_json
from evalvitals.analysis.result_marker import extract_marker_json
from evalvitals.analysis.stats_tool_generator import _parse_result

MARK = "EXPLORATORY_RESULT_JSON="
PAYLOAD = {"plain_question": "q", "observations": ["o1", "o2"], "charts": [{"name": "c"}]}


def test_single_line_payload_still_parses():
    out = "log line\n" + MARK + json.dumps(PAYLOAD) + "\n"
    assert extract_marker_json(out, MARK) == (PAYLOAD, "")
    assert _parse_result_json(out) == (PAYLOAD, "")


def test_pretty_printed_payload_parses_the_whole_object():
    out = "log line\n" + MARK + json.dumps(PAYLOAD, indent=2) + "\n"
    assert extract_marker_json(out, MARK) == (PAYLOAD, "")
    parsed, err = _parse_result_json(out)
    assert err == "" and parsed == PAYLOAD


def test_trailing_output_after_the_object_is_ignored():
    out = MARK + json.dumps(PAYLOAD, indent=2) + "\nFigure saved to figures/x.png\n{not json\n"
    assert _parse_result_json(out) == (PAYLOAD, "")


def test_last_marker_wins():
    first = {"observations": ["stale"]}
    out = MARK + json.dumps(first) + "\n" + MARK + json.dumps(PAYLOAD, indent=2) + "\n"
    assert _parse_result_json(out) == (PAYLOAD, "")


def test_indented_marker_line_is_found():
    out = "   " + MARK + json.dumps(PAYLOAD) + "\n"
    assert _parse_result_json(out) == (PAYLOAD, "")


def test_missing_marker_and_broken_json_report_errors():
    assert _parse_result_json("nothing here") == ({}, f"no {MARK} line in output")
    parsed, err = _parse_result_json(MARK + "{\n  'single': quotes\n}\n")
    assert parsed == {} and err.startswith(f"unparseable {MARK} JSON")
    parsed, err = _parse_result_json(MARK + "[1, 2]\n")
    assert parsed == {} and "must be a JSON object" in err
    parsed, err = _parse_result_json(MARK + "\n")
    assert parsed == {} and err.startswith("empty")


def test_stats_tool_result_accepts_pretty_printed_payload():
    body = {"summary": "s", "effect": 0.25, "ci": [0.1, 0.4], "details": {"n": 10}}
    single = _parse_result("STATS_RESULT_JSON=" + json.dumps(body) + "\n", "tool")
    pretty = _parse_result("STATS_RESULT_JSON=" + json.dumps(body, indent=2) + "\n", "tool")
    assert single.ok and pretty.ok
    assert pretty.effect == single.effect == 0.25
    assert pretty.ci == single.ci == (0.1, 0.4)


def test_stats_tool_result_errors_keep_their_wording():
    missing = _parse_result("no marker at all\n", "tool")
    assert not missing.ok and missing.error == "no STATS_RESULT_JSON line in output"
    bad = _parse_result("STATS_RESULT_JSON={oops\n", "tool")
    assert not bad.ok and bad.error.startswith("unparseable STATS_RESULT_JSON")
    array = _parse_result("STATS_RESULT_JSON=[1]\n", "tool")
    assert not array.ok and "JSON object" in array.error
