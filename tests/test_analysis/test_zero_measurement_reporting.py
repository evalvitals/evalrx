"""M2 must not present 'examined nothing' as 'nothing wrong'.

The 9B bbh_tracking7 run produced exactly this narrative::

    Selected healthy metrics (no threshold violations):
      format_sensitivity.n_scored = 0
      first_error_judge.n_trajectories = 0

Both analyzers had examined zero items — one could not parse the option block,
the other was handed single-turn cases with no trajectories. Reported that way,
a measurement that never happened reads as a clean bill of health, and M3 takes
it as evidence.
"""

from __future__ import annotations

from evalvitals.analysis.analysis_module import _build_narrative, _zero_counts


class _R:
    def __init__(self, findings):
        self.findings = findings


def _real_shape():
    return {
        "format_sensitivity": _R({"n_cases": 48, "n_scored": 0, "mean_flip_rate": None}),
        "first_error_judge": _R({"n_trajectories": 0}),
        "answer_extraction_audit": _R({"n_cases": 125, "n_gradable": 125}),
        "perturbation_battery": _R({"flip_rate": 0.0}),
    }


def test_zero_counts_finds_the_empty_measurements():
    assert _zero_counts(_real_shape()) == [
        "first_error_judge.n_trajectories",
        "format_sensitivity.n_scored",
    ]


def test_a_zero_that_is_a_real_measurement_is_not_flagged():
    """flip_rate = 0 means nothing flipped — that IS a finding, not an absence."""
    zeros = _zero_counts(_real_shape())
    assert "perturbation_battery.flip_rate" not in zeros


def test_booleans_are_not_treated_as_counts():
    assert _zero_counts({"a": _R({"n_ok": False})}) == []


def test_narrative_calls_out_absence_of_evidence():
    text = _build_narrative("m", [], _real_shape())
    assert "measured NOTHING" in text
    assert "NOT evidence of health" in text
    assert "first_error_judge.n_trajectories = 0" in text


def test_narrative_does_not_claim_all_metrics_normal_when_some_measured_nothing():
    text = _build_narrative("m", [], _real_shape())
    assert "all metrics within normal ranges" not in text


def test_all_clear_wording_survives_when_everything_did_measure():
    results = {"answer_extraction_audit": _R({"n_cases": 125, "n_gradable": 125})}
    text = _build_narrative("m", [], results)
    assert "all metrics within normal ranges" in text
    assert "measured NOTHING" not in text


def test_zero_counts_are_excluded_from_the_healthy_list():
    class _F:
        analyzer, metric = "perturbation_battery", "flip_rate"

        def __str__(self):
            return "[MEDIUM] perturbation_battery.flip_rate"

    text = _build_narrative("m", [_F()], _real_shape())
    healthy = text.split("Selected healthy metrics")[-1]
    assert "n_scored = 0" not in healthy
    assert "n_trajectories = 0" not in healthy
    assert "n_gradable" in healthy  # a real measurement still shows up
