from __future__ import annotations


def test_reader_report_is_generic_and_keeps_observation_distinct_from_proof():
    from evalrx.reporting.compiler import compile_reader_report

    report = compile_reader_report({
        "run": {"model": "demo", "benchmark_name": "audio questions", "n_cases": 10, "protocol": "Why errors?"},
        "m1": {"analyzers": ["format_sensitivity"]},
        "m2": {"explore": {"candidate_signals": [{
            "name": "arbitrary_task_signal", "display_name": "Arbitrary task signal",
            "rationale": "It differed between the two outcome groups.",
        }]}},
        "m3": {"hypotheses": []}, "m4": {"ran": False}, "m5_fix": {"ran": False},
    })

    payload = report.to_dict()
    assert "not a proven cause" in payload["headline"]
    assert payload["key_findings"][0]["title"] == "Arbitrary task signal"
    assert payload["key_findings"][0]["summary"] == "It differed between the two outcome groups."
    assert "does not prove" in payload["key_findings"][0]["limitation"]
