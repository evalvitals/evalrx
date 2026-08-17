"""Generalized M2 profile/planner behavior."""

from __future__ import annotations

from evalvitals.analysis import StatsAnalysisAgent, profile_records
from evalvitals.analysis.stats_tools import StatsInput, default_plan


def test_profile_records_infers_roles_and_grain():
    rows = [
        {"case_id": "a", "label": "fail", "model": "m1", "score": 0.8},
        {"case_id": "b", "label": "pass", "model": "m1", "score": 0.2},
    ]

    profile = profile_records(rows)

    assert profile.grain == "case"
    assert profile.columns["case_id"].role == "id"
    assert profile.columns["label"].role == "outcome"
    assert profile.columns["model"].role == "group"
    assert profile.columns["score"].dtype == "numeric"


def test_default_plan_ranks_signals_before_applying_cap():
    labels = {f"f{i}": True for i in range(6)}
    labels.update({f"p{i}": False for i in range(6)})
    weak = {cid: 0.5 for cid in labels}
    strong = {
        cid: (0.9 + i * 0.01 if is_fail else 0.1 + i * 0.01)
        for i, (cid, is_fail) in enumerate(labels.items())
    }
    inp = StatsInput(labels=labels, per_case={"zzz_constant": weak, "aaa_strong": strong})

    plan = default_plan(inp, max_signals=1)

    assert plan[0][0] == "signal_label_assoc"
    assert plan[0][1]["signal"] == "aaa_strong"


def test_stats_analysis_agent_tests_more_than_four_signals_by_default():
    rows = []
    for i in range(12):
        is_fail = i < 6
        row = {"case_id": f"c{i}", "label": "fail" if is_fail else "pass"}
        for j in range(6):
            row[f"signal_{j}"] = (1.0 + 0.01 * i + 0.001 * j) if is_fail else (0.01 * i)
        rows.append(row)

    report = StatsAnalysisAgent().analyze_records(rows)
    signal_results = [
        r for r in report.stats_results
        if r.tool == "signal_label_assoc" and r.ok
    ]

    assert len(signal_results) == 6
    assert report.corrected_rejections["method"] == "BH"
    assert report.corrected_rejections["families"]["bh"]["n_tested"] >= 6


# ── the M2 section parser must survive how judges actually write ─────────────
def _base_report():
    from evalvitals.analysis.analysis_module import AnalysisReport
    return AnalysisReport(model_name="m",
                          narrative="Model: EndpointModel(qwen3.5-2b)\nrest")


def test_markdown_headings_are_parsed_not_dropped():
    """A judge wrote '## CONCLUSION'; the matcher demanded 'CONCLUSION:'.

    No section was ever entered, so 4,158 characters of analysis fell through to
    the base narrative's first line and M3 received nothing to hypothesise from.
    The run then reported stopped_by=no_hypotheses as though that were a
    finding. Earlier runs parsed only because the judge happened to use a colon.
    """
    from evalvitals.analysis.stats_agent import _parse_llm_analysis

    raw = (
        "## CONCLUSION\n\n"
        "The failures are a capability defect in list-state maintenance.\n\n"
        "## EVIDENCE_CHAIN\n\n"
        "- Step 1 - kendall tau 0.92 with edit distance 7.82\n"
        "- Step 2 - 66% token loss\n\n"
        "## QUALITATIVE\n\n"
        "- probe1 has an internally incoherent signature on ~45% of fails\n"
    )
    conclusion, evidence, qualitative = _parse_llm_analysis(raw, _base_report())
    assert "capability defect" in conclusion
    assert len(evidence) == 2 and "kendall tau" in evidence[0]
    assert len(qualitative) == 1


def test_the_original_colon_contract_still_parses():
    from evalvitals.analysis.stats_agent import _parse_llm_analysis

    raw = ("CONCLUSION: it broke\n"
           "EVIDENCE_CHAIN:\n- step one\n- step two\n"
           "QUALITATIVE:\n- note")
    conclusion, evidence, qualitative = _parse_llm_analysis(raw, _base_report())
    assert conclusion == "it broke"
    assert evidence == ["step one", "step two"]
    assert qualitative == ["note"]


def test_bold_headings_parse_too():
    from evalvitals.analysis.stats_agent import _parse_llm_analysis

    raw = "**CONCLUSION**\nthe thing happened\n**EVIDENCE_CHAIN**\n- because\n"
    conclusion, evidence, _ = _parse_llm_analysis(raw, _base_report())
    assert conclusion == "the thing happened"
    assert evidence == ["because"]


def test_prose_without_sections_still_falls_back():
    from evalvitals.analysis.stats_agent import _parse_llm_analysis

    conclusion, _, _ = _parse_llm_analysis("just prose", _base_report())
    assert conclusion == "Model: EndpointModel(qwen3.5-2b)"
