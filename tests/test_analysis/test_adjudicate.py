"""Phase A — host adjudication of explorer candidate signals.

These tests pin the firewall contract: the explorer PROPOSES (via ``sufficient``)
and the host DECIDES (recomputes effect/e-value/reject with the validated core,
applies one e-BH family, never trusts a self-declared verdict).
"""

from __future__ import annotations

import pytest

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


def test_two_group_gets_a_p_value_and_its_own_bh_family():
    # Signal-absent cases all pass (0), signal-present cases all fail (1).
    c = _sig("strong_signal", {"kind": "two_group", "a": [0] * 8, "b": [1] * 8})
    meta = adjudicate_signals([c])

    assert c.host_adjudicated
    assert c.e_value is None          # unpaired path produces no e-value
    # Exact test: one table in C(16, 8) is this extreme, on each side.
    assert c.p_value == pytest.approx(2 / 12870)
    assert c.reject is True and c.fdr_corrected is True
    assert c.effect is not None and c.ci is not None
    assert meta["n_in_family"] == 0   # no e-value-bearing candidate
    assert meta["n_in_p_family"] == 1
    assert meta["method"] == "BH"     # the correction that was actually applied


def test_two_group_candidates_pay_for_multiplicity():
    # 9/20 against 2/20 failures: p is about 0.03 on its own, so it rejects
    # alone but not among twenty candidates of which it is the best.
    borderline = {"kind": "two_group", "a": [1] * 2 + [0] * 18, "b": [1] * 9 + [0] * 11}
    alone = _sig("alone", dict(borderline))
    assert adjudicate_signals([alone])["rejected"] == ["alone"]

    noise = {"kind": "two_group", "a": [1, 0] * 10, "b": [1, 0] * 10}
    crowd = [_sig("candidate", dict(borderline))] + [
        _sig(f"noise{i}", dict(noise)) for i in range(19)
    ]
    meta = adjudicate_signals(crowd)
    assert meta["n_in_p_family"] == 20
    assert meta["rejected"] == []
    assert meta["raw_reject"] == ["candidate"]   # its own interval excluded zero
    assert crowd[0].reject is False and crowd[0].fdr_corrected is False


def test_two_group_with_group_ids_resamples_whole_groups():
    # Forty rows per side from five source items per side, and rows of an
    # item agree: the evidence is ten items, not eighty rows.
    a = [0] * 32 + [1] * 8
    b = [1] * 24 + [0] * 16
    rows = {"kind": "two_group", "a": a, "b": b}
    grouped = {**rows, "clusters_a": [f"a{i // 8}" for i in range(40)],
               "clusters_b": [f"b{i // 8}" for i in range(40)]}
    independent, clustered = _sig("rows", rows), _sig("groups", grouped)
    adjudicate_signals([independent])
    adjudicate_signals([clustered])
    assert independent.reject is True
    assert clustered.reject is False
    assert clustered.p_value > 0.05 > independent.p_value

    # Group ids that do not line up with the rows are not adjudicable.
    broken = _sig("broken", {**rows, "clusters_a": ["a1"], "clusters_b": ["b1"]})
    adjudicate_signals([broken])
    assert broken.host_adjudicated is False and broken.descriptive_only is True


def test_header_says_none_when_nothing_was_tested():
    meta = adjudicate_signals([_sig("vibes"), _sig("more_vibes")])
    assert meta["method"] == "none"
    assert meta["n_host_adjudicated"] == 0


def test_header_names_both_families_when_both_were_used():
    paired = _sig("paired", {"kind": "paired_binary", "b": 15, "c": 0})
    unpaired = _sig("unpaired", {"kind": "two_group", "a": [0] * 8, "b": [1] * 8})
    meta = adjudicate_signals([paired, unpaired])
    assert meta["method"] == "e-BH + BH"
    assert meta["rejected"] == ["paired", "unpaired"]


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
    # 4 against 4 with a clean split: exactly 2 of the C(8, 4) tables.
    assert cand.p_value == pytest.approx(2 / 70)
    assert cand.fdr_corrected is True    # survived its (one-member) BH family
