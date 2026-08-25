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


def test_prose_without_sections_yields_no_conclusion_not_a_synthesized_one():
    """No section => "" — so the caller keeps the deterministic conclusion it
    already built. Returning base.narrative's first line here ("Model: <repr>")
    was how a quota message from the CLI silently replaced a real M2 verdict.
    """
    from evalvitals.analysis.stats_agent import _parse_llm_analysis

    conclusion, _, _ = _parse_llm_analysis("just prose", _base_report())
    assert conclusion == ""


# ── what the judge is shown about multiplicity ───────────────────────────────
def _bh_pair():
    """One survivor and one that BH kills, as ``signal_label_assoc`` emits them.

    ``summary`` says REJECT H0 on BOTH because it is baked before correction,
    and ``reject`` stays raw for the BH family on purpose — so the rendered
    block is the only place the difference can appear.
    """
    from evalvitals.analysis.stats_tools import StatsToolResult, fdr_correct

    strong = StatsToolResult(
        tool="signal_label_assoc", ok=True, effect=0.46, reject=True, p_value=3e-7,
        summary="signal 'probe1.dropped' vs FAIL: effect=+0.4635 -> REJECT H0",
        analysis_key="signal_label_assoc:probe1.dropped",
        correction_family="bh", raw_reject=True, details={"n_signal": 54},
    )
    noise = StatsToolResult(
        tool="signal_label_assoc", ok=True, effect=-0.50, reject=True, p_value=1.0,
        summary="signal 'cot.drift_away' vs FAIL: effect=-0.5000 -> REJECT H0",
        analysis_key="signal_label_assoc:cot.drift_away",
        correction_family="bh", raw_reject=True, details={"n_signal": 1},
    )
    corrected = fdr_correct([strong, noise], alpha=0.05)
    return [strong, noise], corrected


def test_uncorrected_reject_is_marked_as_not_surviving():
    """A p=1.000, n_signal=1 result printed a bare "REJECT H0" to the judge.

    ``summary`` is frozen at tool-run time, so correcting ``reject`` alone would
    not have changed one character of what the judge reads.
    """
    from evalvitals.analysis.stats_agent import _format_stats_for_prompt

    results, corrected = _bh_pair()
    block = _format_stats_for_prompt(results, corrected)

    survivor, killed = [ln for ln in block.splitlines() if "vs FAIL" in ln]
    assert "[BH: survived, p=3e-07, n_signal=54]" in survivor
    assert "[BH: NOT survived, p=1, n_signal=1]" in killed


def test_survivors_are_listed_per_signal_not_per_tool():
    """The old footer printed tool NAMES, so 42 signals collapsed to one word."""
    from evalvitals.analysis.stats_agent import _format_stats_for_prompt

    results, corrected = _bh_pair()
    block = _format_stats_for_prompt(results, corrected)

    assert "1 of 2 corrected test(s) survive" in block
    assert "* signal_label_assoc:probe1.dropped" in block
    assert "cot.drift_away" not in block.split("survive:")[1]


def test_a_judge_that_never_answered_says_so_in_the_report():
    """A crashed judge and an uninformative one produced identical runs.

    On qwen3.5-2b / bbh_word_sorting the judge call raised OSError(E2BIG) —
    the prompt exceeded one argv entry. M2 caught it, fell back to the
    threshold narrative, M3 had nothing to hypothesise from, and the chain
    finished rc=0 with stopped_by=no_hypotheses. The only record was a
    logger.warning that reached neither the console nor run_log.jsonl, so the
    run was indistinguishable from a clean "nothing found".
    """
    from evalvitals.analysis import StatsAnalysisAgent
    from evalvitals.core.result import Result
    from evalvitals.eval_agent.stages.protocol import ExperimentProtocol

    class DeadJudge:
        def generate(self, prompt, **kwargs):
            raise OSError(7, "Argument list too long")

    res = {"a": Result(analyzer="a", model="m", findings={"x": 1.0})}
    report = StatsAnalysisAgent(judge=DeadJudge()).analyze(
        res, model_name="m", protocol=ExperimentProtocol(description="d"))

    assert "Argument list too long" in report.llm_fallback_reason
    assert report.stats_tool != "llm_guided"


def test_a_judge_that_answered_leaves_no_fallback_reason():
    """"" must never have to be read as "it failed but we don't know why"."""
    from evalvitals.analysis import StatsAnalysisAgent
    from evalvitals.core.result import Result
    from evalvitals.eval_agent.stages.protocol import ExperimentProtocol

    class LiveJudge:
        def generate(self, prompt, **kwargs):
            return "CONCLUSION: it worked\nEVIDENCE_CHAIN:\n- because"

    res = {"a": Result(analyzer="a", model="m", findings={"x": 1.0})}
    report = StatsAnalysisAgent(judge=LiveJudge()).analyze(
        res, model_name="m", protocol=ExperimentProtocol(description="d"))

    assert report.llm_fallback_reason == ""
    assert report.stats_tool == "llm_guided"
    assert report.conclusion == "it worked"


def test_descriptive_results_are_not_labelled_as_failing_correction():
    """``rank_corr`` never enters a family; absence of a verdict is not a No."""
    from evalvitals.analysis.stats_agent import _format_stats_for_prompt
    from evalvitals.analysis.stats_tools import StatsToolResult

    descriptive = StatsToolResult(
        tool="rank_corr", ok=True, effect=0.24, reject=None,
        summary="Kendall tau between 'probe1.dropped' and FAIL = +0.243",
        correction_family=None,
    )
    block = _format_stats_for_prompt([descriptive], {"n_tested": 0})

    assert "Kendall tau" in block
    assert "NOT survived" not in block and "survived" not in block
