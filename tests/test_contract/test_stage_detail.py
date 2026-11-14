"""ReportData.stage_detail conforms to evalrx.contract.stage_detail.

The report compiler builds plain dicts and never validates them at run time,
so this is where a key it adds without declaring it (or a value of the wrong
type) gets caught -- before the UI's generated types silently lie.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("pydantic")

from pydantic import ValidationError  # noqa: E402

from evalrx.contract.stage_detail import StageDetail  # noqa: E402

FIXTURE = Path(__file__).resolve().parents[2] / "evalrx/reporting/web/fixtures/chartqa-gemini.json"


def test_a_real_runs_stage_detail_conforms():
    data = json.loads(FIXTURE.read_text(encoding="utf-8"))["data"]
    StageDetail.model_validate(data["stage_detail"])


@pytest.mark.parametrize("example", ["synthetic_yield_explore", "deco_hallu_explore"])
def test_an_explore_output_conforms(example):
    from evalrx.reporting.dynamic import build_report_data

    root = Path(__file__).resolve().parents[2] / "examples/m2_m3" / example / "reference_output"
    StageDetail.model_validate(build_report_data(root)["stage_detail"])


def test_a_fused_output_conforms(tmp_path):
    from evalrx.analysis.fused_pipeline import FusedReport, FusedSignal
    from evalrx.reporting.dynamic import build_report_data

    report = FusedReport(
        question="q", ok=True, observations=["o"], caveats=["c"],
        candidate_signals=[FusedSignal(name="s", source="explorer", reject=True, effect=0.3,
                                       ci=(0.1, 0.5), p_value=0.01)],
        recommended_confirmatory_tests=["t"], conclusion="c",
    )
    (tmp_path / "fused_report.json").write_text(json.dumps(report.to_dict()), encoding="utf-8")
    StageDetail.model_validate(build_report_data(tmp_path)["stage_detail"])


def test_a_full_m1_to_m5_run_conforms(tmp_path):
    from evalrx.reporting.dynamic import build_report_data
    from tests.test_analysis.test_case_study_sheet import make_run

    detail = build_report_data(make_run(tmp_path))["stage_detail"]
    StageDetail.model_validate(detail)
    assert detail["m5"]["ran"] and detail["m4"]["ran"]


def test_a_logged_run_with_examples_and_a_repair_conforms(tmp_path):
    from evalrx.core import CaseBatch, FailureCase, Label, Result
    from evalrx.eval_agent.run_logger import RunLogger
    from evalrx.reporting.dynamic import build_report_data

    cases = CaseBatch([
        FailureCase.from_prompt("Q1", id="c1", expected="a", observed="b", label=Label.FAIL),
        FailureCase.from_prompt("Q2", id="c2", expected="a", observed="a", label=Label.PASS),
    ])
    logger = RunLogger(tmp_path, observability_mode="offline")
    logger.log_run_start({"model": "demo", "n_cases": 2})
    logger.log_cases(cases)
    logger.log_probe(0, {"format_check": Result(
        analyzer="format_check", model="demo",
        findings={"per_case": [{"sample_id": "c1", "answer_flipped": 1}]},
    )}, cases=cases)
    winner = {"name": "reask", "tier": "L1", "n_pairs": 2, "n_fixed": 1, "n_broken": 0,
              "effect": 0.5, "fixed": True, "verdict": "fixed", "fixed_cases": ["c1"]}
    logger.log_fix(SimpleNamespace(to_dict=lambda: {
        "attempted": [winner], "best": dict(winner), "fixed": True,
        "recommendation": {"action": "promote", "reason": "fixed one, broke none"},
    }))
    logger.close()

    detail = build_report_data(tmp_path)["stage_detail"]
    StageDetail.model_validate(detail)
    assert detail["m1"]["examples"] and detail["m5"]["candidates"]


def test_an_empty_run_conforms(tmp_path):
    from evalrx.eval_agent.run_logger import RunLogger
    from evalrx.reporting.dynamic import build_report_data

    logger = RunLogger(tmp_path, observability_mode="offline")
    logger.log_run_start({"model": "demo", "n_cases": 0})
    logger.close()
    StageDetail.model_validate(build_report_data(tmp_path)["stage_detail"])


def test_an_undeclared_key_from_the_compiler_is_rejected():
    data = json.loads(FIXTURE.read_text(encoding="utf-8"))["data"]
    detail = data["stage_detail"]
    detail["m1"]["probes"][0]["new_field"] = 1
    with pytest.raises(ValidationError, match="new_field"):
        StageDetail.model_validate(detail)
