"""Phase A — host adjudication of explorer candidate signals.

These tests pin the firewall contract: the explorer PROPOSES (via ``sufficient``)
and the host DECIDES (recomputes effect/e-value/reject with the validated core,
applies one e-BH family, never trusts a self-declared verdict).
"""

from __future__ import annotations

from evalrx.agent_runtime.sandbox import ExperimentSandbox
from evalrx.analysis import (
    CandidateSignal,
    ExploratoryAnalysisAgent,
    ExploratoryAnalysisReport,
    adjudicate_report,
    adjudicate_signals,
)


def _sig(name: str, sufficient=None) -> CandidateSignal:
    return CandidateSignal(name=name, sufficient=sufficient)


# ---------------------------------------------------------------------------
# Per-shape host reconstruction
# ---------------------------------------------------------------------------

def test_paired_binary_strong_rejects_via_ebh():
    c = _sig("intervention_helps", {"kind": "paired_binary", "b": 15, "c": 0})
    meta = adjudicate_signals([c])

    assert c.host_adjudicated and not c.descriptive_only
    assert c.e_value is not None and c.e_value > 20  # 2**11 for b=15/n=15
    assert c.fdr_corrected is True
    assert c.reject is True
    assert meta["n_in_family"] == 1 and meta["n_rejected"] == 1


def test_balanced_paired_binary_does_not_reject():
    c = _sig("noise", {"kind": "paired_binary", "b": 4, "c": 4})
    adjudicate_signals([c])

    assert c.host_adjudicated
    assert c.e_value is not None and c.e_value < 20
    assert c.reject is False


def test_two_group_uses_ci_reject_and_is_not_in_ebh_family():
    # Signal-absent cases all pass (0), signal-present cases all fail (1).
    c = _sig("strong_signal", {"kind": "two_group", "a": [0] * 8, "b": [1] * 8})
    meta = adjudicate_signals([c])

    assert c.host_adjudicated
    assert c.e_value is None          # unpaired path produces no e-value
    assert c.fdr_corrected is False   # therefore not part of the e-BH family
    assert c.reject is True           # but CI excludes 0 -> host rejects
    assert c.effect is not None and c.ci is not None
    assert meta["n_in_family"] == 0   # no e-value-bearing candidate


def test_no_sufficient_is_descriptive_only():
    c = _sig("vibes")  # no sufficient attached
    adjudicate_signals([c])

    assert c.host_adjudicated is False
    assert c.descriptive_only is True
    assert c.reject is False
    assert c.e_value is None


def test_unadjudicable_sufficient_is_descriptive_only():
    c = _sig("weird", {"kind": "spearman", "rho": 0.9})  # not a host-adjudicable shape
    adjudicate_signals([c])

    assert c.host_adjudicated is False
    assert c.descriptive_only is True
    assert c.reject is False


# ---------------------------------------------------------------------------
# Multiplicity: one e-BH family across e-value-bearing candidates
# ---------------------------------------------------------------------------

def test_ebh_family_controls_multiplicity():
    strong = _sig("strong", {"kind": "paired_binary", "b": 20, "c": 0})
    weak = _sig("weak", {"kind": "paired_binary", "b": 5, "c": 4})
    meta = adjudicate_signals([strong, weak])

    assert meta["n_in_family"] == 2
    assert strong.reject is True
    assert weak.reject is False          # survives marginally but loses to e-BH
    assert meta["rejected"] == ["strong"]


# ---------------------------------------------------------------------------
# Report-level wiring + honest in-sample caveat
# ---------------------------------------------------------------------------

def test_adjudicate_report_records_metadata_and_in_sample_caveat():
    report = ExploratoryAnalysisReport(
        ok=True,
        candidate_signals=[_sig("s", {"kind": "paired_binary", "b": 12, "c": 0})],
    )
    adjudicate_report(report, split_label="in_sample")

    assert report.adjudication["method"] == "e-BH"
    assert report.adjudication["split"] == "in_sample"
    assert any("IN-SAMPLE" in c for c in report.caveats)


def test_adjudicate_report_no_caveat_when_split_is_confirm():
    report = ExploratoryAnalysisReport(
        ok=True,
        candidate_signals=[_sig("s", {"kind": "paired_binary", "b": 12, "c": 0})],
    )
    adjudicate_report(report, split_label="confirm")

    assert report.adjudication["split"] == "confirm"
    assert not any("IN-SAMPLE" in c for c in report.caveats)


# ---------------------------------------------------------------------------
# A self-declared verdict in the explorer's output is structurally ignored
# ---------------------------------------------------------------------------

_CODE_WITH_SELF_DECLARED_VERDICT = """
import json
from pathlib import Path

rows = json.loads(Path("records.json").read_text())
absent = [int(r["label"] == "fail") for r in rows if r["flag"] == 0]
present = [int(r["label"] == "fail") for r in rows if r["flag"] == 1]
payload = {
    "observations": ["flag tracks failure"],
    "candidate_signals": [
        {
            "name": "flag",
            "rationale": "flag concentrated in fails",
            "suggested_test": "signal_label_assoc",
            "sufficient": {"kind": "two_group", "a": absent, "b": present},
            "reject": True,          # self-declared — MUST be ignored
            "e_value": 999999.0,     # self-declared — MUST be ignored
        }
    ],
    "plots": [], "tables": {}, "charts": [], "caveats": [],
    "recommended_confirmatory_tests": [],
}
print("EXPLORATORY_RESULT_JSON=" + json.dumps(payload))
"""


class _ScriptedJudge:
    def __init__(self, code: str) -> None:
        self._code = code

    def generate(self, prompt: str, **kwargs) -> str:
        return f"```python\n{self._code}\n```"


def test_self_declared_reject_and_evalue_are_ignored(tmp_path):
    rows = [
        {"case_id": f"c{i}", "label": "fail" if i < 4 else "pass", "flag": int(i < 4)}
        for i in range(8)
    ]
    agent = ExploratoryAnalysisAgent(
        judge=_ScriptedJudge(_CODE_WITH_SELF_DECLARED_VERDICT),
        sandbox=ExperimentSandbox(workdir=tmp_path, cleanup=False),
    )
    report = agent.explore_records(rows, question="What predicts failure?")
    assert report.ok
    (cand,) = report.candidate_signals

    # The parser carries `sufficient` through but NEVER the self-declared verdict.
    # absent (flag==0) are the pass rows -> 0; present (flag==1) are the fails -> 1.
    assert cand.sufficient == {"kind": "two_group", "a": [0, 0, 0, 0], "b": [1, 1, 1, 1]}
    assert cand.reject is None
    assert cand.e_value is None

    # The host recomputes from `sufficient` alone — the bogus 999999 never appears.
    adjudicate_report(report)
    assert cand.host_adjudicated is True
    assert cand.e_value is None          # two_group -> no e-value (not 999999)
    assert cand.fdr_corrected is False
