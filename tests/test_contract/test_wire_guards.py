"""The contract's validators exist to catch specific, observed failure modes.

Each test here names the failure it prevents. A guard without a test is a
comment.
"""

import pytest
from pydantic import ValidationError

from evalvitals.contract import (
    CaseBatchRef, FailureCaseWire, HypothesisStatus, InputsWire, PerCaseRow,
    ProbeOutput, ResultWire, StageState, StageStatus,
)
from evalvitals.contract.m2 import CorrectedRejections, StatsToolResultWire
from evalvitals.contract.m3 import HypothesisWire
from evalvitals.contract.m4 import FixAttemptWire, FixOutput
from evalvitals.contract.methodology import MethodologyWire
from evalvitals.contract.m5 import HypothesisTestOutput, HypothesisTestResultWire


def _status(stage, state=StageState.SUCCEEDED, cycle=0):
    return StageStatus(stage=stage, state=state, cycle=cycle)


def _envelope(stage, **kw):
    return dict(
        trace_id="t1", cycle=0, produced_at="2026-08-19T00:00:00Z",
        status=_status(stage), **kw,
    )


def _hyp(hid="h1", **kw):
    kw.setdefault("statement", "dark images push the model onto text priors")
    kw.setdefault("target_model", "qwen")
    kw.setdefault("predicted_failure_mode", "visual_grounding")
    kw.setdefault("test_design", "attention.image_token_ratio")
    return HypothesisWire(id=hid, **kw)


# --- M1: the join key -------------------------------------------------------

def test_per_case_row_requires_sample_id():
    """Without it the analyzer's whole contribution vanishes with no error."""
    with pytest.raises(ValidationError):
        PerCaseRow(attention_entropy=0.73)


def test_per_case_row_rejects_nested_numeric_dict():
    """The harvester scans one level; nested numbers are silently dropped."""
    with pytest.raises(ValidationError, match="numeric dict"):
        PerCaseRow(sample_id="s1", layers={"l0": 0.1, "l1": 0.2})


def test_per_case_row_rejects_numeric_vector():
    """Live case: step_rollout_value emits step_values=[1.0, ...] and no stat sees it."""
    with pytest.raises(ValidationError, match="numeric sequence"):
        PerCaseRow(sample_id="s1", step_values=[1.0, 1.0, 0.5])


def test_per_case_row_allows_a_string_list():
    row = PerCaseRow(sample_id="s1", tags=["a", "b"], score=0.4)
    assert row.signals() == {"score": 0.4}


def test_per_case_row_keeps_free_form_extras():
    row = PerCaseRow(sample_id="s1", attention_entropy=0.73, note="a string")
    assert row.signals() == {"attention_entropy": 0.73}


def test_analyzer_name_may_not_contain_dot():
    """Signals are named '<analyzer>.<metric>'; a dot makes the split ambiguous."""
    with pytest.raises(ValidationError, match="must not contain"):
        ResultWire(analyzer="a.b", model="m", n_cases=1)


def test_probe_output_keys_must_match_analyzer_field():
    with pytest.raises(ValidationError, match="keys must match"):
        ProbeOutput(**_envelope("m1"), results={
            "attention": ResultWire(analyzer="confidence", model="m", n_cases=1)
        })


def test_probe_output_enumerates_signal_names():
    out = ProbeOutput(**_envelope("m1"), results={
        "attention": ResultWire(
            analyzer="attention", model="m", n_cases=2,
            findings={"summary_score": 0.41,
                      "per_case": [{"sample_id": "s1", "entropy": 0.7}]},
        )
    })
    assert out.signal_names() == ["attention.entropy", "attention.summary_score"]


# --- common: identity -------------------------------------------------------

def test_trajectory_sample_id_must_equal_case_id():
    """A divergent id registers the case under two keys and double-counts it."""
    with pytest.raises(ValidationError, match="double-count"):
        FailureCaseWire(
            id="c1", inputs=InputsWire(prompt="q"),
            trajectory={"sample_id": "OTHER", "steps": []},
        )


def test_modalities_are_derived_from_filled_slots():
    inp = InputsWire(prompt="what is said?", audio={"kind": "path", "value": "a.wav"})
    assert inp.modalities() == {"text", "audio"}


# --- M2: raw reject is not a verdict ---------------------------------------

def test_raw_reject_alone_is_not_decisive():
    r = StatsToolResultWire(
        tool="signal_label_assoc", ok=True, reject=True,
        correction_method="ebh", fdr_corrected=False,
        analysis_key="signal_label_assoc:attention.entropy",
    )
    assert r.is_decisive() is False


def test_corrected_rejection_is_decisive():
    r = StatsToolResultWire(
        tool="signal_label_assoc", ok=True, reject=True,
        correction_method="ebh", fdr_corrected=True,
        analysis_key="signal_label_assoc:attention.entropy",
    )
    assert r.is_decisive() is True


def test_descriptive_report_may_not_ship_a_verdict():
    from evalvitals.contract import StatsReportWire
    with pytest.raises(ValidationError, match="must not ship a validity verdict"):
        StatsReportWire(
            **_envelope("m2"), descriptive_only=True,
            corrected_rejections=CorrectedRejections(
                method="ebh", deferred=False, rejected_result_keys=["k1"],
            ),
        )


# --- M3: routability is reported, not enforced ------------------------------

def test_test_design_accepts_a_signal_reference():
    h = _hyp(test_design="attention.image_token_ratio")
    assert h.is_proposed and h.is_routable


def test_test_design_accepts_a_known_directive():
    h = _hyp(test_design="prompt_contrast describe_first")
    assert h.is_proposed and h.is_routable


def test_a_prose_design_that_names_its_signal_is_routable():
    """What a strong judge actually writes.

    Opus at high effort produced designs that name the analyzer and metric
    inside a paragraph of interventional protocol. M5 routed all of them
    (routed_by="test_design"), while a validator demanding the string BE a bare
    signal rejected the whole M3 payload — stricter than the consumer it exists
    to protect, so it discarded good work without preventing anything.
    """
    h = _hyp(test_design=(
        "Re-run `modality_ablation` in swap mode on the audio slot (substitute a "
        "mismatched track rather than dropping it) — predict "
        "`modality_ablation.grounded_in_audio` falls on Audio-Visual items."
    ))
    assert h.is_proposed and h.is_routable


def test_a_design_naming_nothing_measured_is_proposed_but_not_routable():
    """The third state, which used to be a rejection.

    A test WAS proposed — it just names a flag nobody has computed yet. That is
    work for the next M1 cycle, not a claim with no falsifier, and reporting it
    as "no test proposed" would erase a real experimental plan.
    """
    h = _hyp(test_design=(
        "Add a per-case flag question_has_unfilled_template_slot (regex over the "
        "question text) and cross it with FAIL."
    ))
    assert h.is_proposed and not h.is_routable


def test_no_test_design_at_all_is_untestable():
    h = _hyp(test_design="")
    assert not h.is_proposed and not h.is_routable


# --- M5: both gates ---------------------------------------------------------

def _result(status, consistent, grade="observational", hid="h1"):
    return HypothesisTestResultWire(
        hypothesis_id=hid, status=status, test_name="signal_label_assoc",
        effect_size=0.31, confidence=0.7, evidence_grade=grade,
        is_consistent_with_protocol=consistent, verdict="v",
    )


def test_supported_requires_protocol_consistency():
    with pytest.raises(ValidationError, match="protocol consistency"):
        _result(HypothesisStatus.SUPPORTED, consistent=False)


def test_supported_requires_evidence():
    with pytest.raises(ValidationError, match="nothing backing it"):
        _result(HypothesisStatus.SUPPORTED, consistent=True, grade="none")


def test_verified_is_derived_from_results():
    out = HypothesisTestOutput(**_envelope("m5"), results=[
        _result(HypothesisStatus.SUPPORTED, consistent=True, hid="h1"),
        _result(HypothesisStatus.INCONCLUSIVE, consistent=True, hid="h2"),
        _result(HypothesisStatus.SUPPORTED, consistent=True, hid="h3"),
    ])
    assert out.verified() == ["h1", "h3"]


def test_nothing_verified_when_no_result_passes_both_gates():
    out = HypothesisTestOutput(**_envelope("m5"), results=[
        _result(HypothesisStatus.REFUTED, consistent=True),
    ])
    assert out.verified() == []


# --- M4: selection is not confirmation --------------------------------------

_MIN_XML = (
    '<mxfile><diagram><mxGraphModel><root>'
    '<mxCell id="0"/><mxCell id="1" parent="0"/>'
    '<mxCell id="a" value="step" vertex="1" parent="1"/>'
    '<mxCell id="b" value="done" vertex="1" parent="1"/>'
    '<mxCell id="e" edge="1" parent="1" source="a" target="b"/>'
    '</root></mxGraphModel></diagram></mxfile>'
)


def _method(**kw):
    kw.setdefault("title", "restate then answer")
    kw.setdefault("drawio_xml", _MIN_XML)
    return MethodologyWire(**kw)


def _attempt(name="c1", **kw):
    kw.setdefault("verdict", "fixed")
    kw.setdefault("methodology", _method())
    return FixAttemptWire(tier="L1", name=name, **kw)


def _fix(**kw):
    kw.setdefault("max_tier", "L1")
    return FixOutput(**_envelope("m4_fix"), **kw)


def test_fixed_needs_the_winner_among_the_attempts():
    with pytest.raises(ValidationError, match="not among the validated attempts"):
        _fix(fixed=True, best="c1", attempted=[])


def test_fixed_requires_ebh_survivor():
    with pytest.raises(ValidationError, match="e-BH survivor"):
        _fix(fixed=True, attempted=[_attempt()], best="c1", ebh_survivors=["other"])


def test_intervention_flag_is_derived_from_strategy():
    from evalvitals.contract import InterventionOutput

    def _op(strategy):
        return InterventionOutput(
            **_envelope("m4_surgery"), hypothesis_id="h1",
            hypothesis_status=HypothesisStatus.SUPPORTED, strategy=strategy,
        )

    assert _op("passive_correlation").is_intervention is False
    assert _op("experiment_writer").is_intervention is True


# --- pre-M1 -----------------------------------------------------------------

def test_probe_search_counts_must_be_consistent():
    from evalvitals.contract import ProbeSearchOutput
    with pytest.raises(ValidationError, match="cannot exceed"):
        ProbeSearchOutput(
            **_envelope("pre_m1"), n_simulations=20, n_macro=12, n_micro=8,
            all_cases=CaseBatchRef(path="all.json", n_cases=5),
            failure_cases=CaseBatchRef(path="fail.json", n_cases=7),
        )


def test_probe_search_error_rate_is_derived():
    from evalvitals.contract import ProbeSearchOutput
    out = ProbeSearchOutput(
        **_envelope("pre_m1"), n_simulations=20, n_macro=12, n_micro=8,
        all_cases=CaseBatchRef(path="all.json", n_cases=20),
        failure_cases=CaseBatchRef(path="fail.json", n_cases=7),
    )
    assert out.error_rate == pytest.approx(0.35)


# --- methodology graph ------------------------------------------------------

def test_methodology_requires_a_diagram():
    with pytest.raises(ValidationError):
        MethodologyWire(title="t")


def test_malformed_xml_is_rejected():
    with pytest.raises(ValidationError, match="not well-formed"):
        MethodologyWire(title="t", drawio_xml="<mxfile><diagram>")


def test_xml_edge_into_empty_space_is_rejected():
    """Renders as a normal-looking diagram; nothing downstream would notice."""
    bad = _MIN_XML.replace('target="b"', 'target="ghost"')
    with pytest.raises(ValidationError, match="arrow into empty space"):
        MethodologyWire(title="t", drawio_xml=bad)


def test_duplicate_cell_ids_are_rejected():
    bad = _MIN_XML.replace('<mxCell id="a" value="step"', '<mxCell id="a" value="dupe"/><mxCell id="a" value="step"')
    with pytest.raises(ValidationError, match="duplicate mxCell ids"):
        MethodologyWire(title="t", drawio_xml=bad)


def test_fixed_requires_a_readable_methodology():
    with pytest.raises(ValidationError, match="must carry an explanation"):
        _fix(fixed=True, attempted=[_attempt(methodology=None)], best="c1")


def test_a_validated_fix_is_accepted():
    out = _fix(fixed=True, attempted=[_attempt()], best="c1", ebh_survivors=["c1"])
    assert out.best == "c1"


# --- imputed absence: measured on 8, tested on 125 --------------------------

def test_imputation_share_surfaces_unmeasured_control_cases():
    """Real result from bbh_tracking7: step_rollout_value.recoverable."""
    from evalvitals.contract.m2 import StatsToolResultWire
    r = StatsToolResultWire(
        tool="signal_label_assoc", ok=True,
        config={"signal": "step_rollout_value.recoverable"},
        effect=0.361, reject=True, correction_method="ebh", fdr_corrected=True,
        analysis_key="signal_label_assoc:step_rollout_value.recoverable",
        n_signal=7, n_control=118, n_measured=8, n_imputed_absent=117,
    )
    assert r.is_decisive() is True          # it does pass every statistical gate
    assert r.imputation_share == pytest.approx(117 / 125)
    assert r.is_mostly_imputed()            # ...on 94% never-measured controls


def test_fully_measured_result_is_not_flagged():
    from evalvitals.contract.m2 import StatsToolResultWire
    r = StatsToolResultWire(
        tool="signal_label_assoc", ok=True, n_signal=2, n_control=6,
        n_measured=8, n_imputed_absent=0,
    )
    assert r.is_mostly_imputed() is False
