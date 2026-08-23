"""What a real, cheap judge actually does — and the pipeline surviving it.

Every bug the first live vLLM run exposed was invisible to this suite, and the
reason was the fixtures rather than the coverage: the scripted judge always
returned a ``test`` field, every fake case carried an ``id``, and no fake model
declared TOOL_CALLS. Each of those is generous in a way a real run is not.

So the fixtures here are deliberately shabby, in the exact ways a cheap model on
low effort was observed to be shabby:

* it proposes a mechanism and no way to be wrong about it (no ``TEST:`` line),
* it leaves ``Hypothesis.id`` empty, because nothing in the loop fills it,
* the model under test declares tool calling, because every chat model served
  over an OpenAI-compatible endpoint does.

The assertions are about the pipeline staying COHERENT under that, not about it
producing a good diagnosis. A degraded judge should yield a degraded verdict
that still says what it is — not a broken join, and not an untestable claim
dressed up as a testable one.
"""

from __future__ import annotations

import pytest

from evalvitals.core.capability import Capability
from evalvitals.core.case import CaseBatch, FailureCase, Inputs, Label
from evalvitals.eval_agent import DiagnosisAgent, RunContext, VLDiagnoseLoop
from evalvitals.eval_agent.stages.protocol import ExperimentProtocol
from tests.conftest import FakeModel
from tests.test_eval_agent.test_vl_diagnose import ScriptedModel

pytest.importorskip("pydantic")

from evalvitals.contract import (  # noqa: E402
    SCHEMA_VERSION, DiagnosisOutput, HypothesisTestOutput, ProbeOutput,
)

#: What the judge returned on the live run: a mechanism, a failure mode, and no
#: `test` key at all. The parser accepts it and `test_design` comes out "".
NO_TEST_RESPONSE = (
    '[{"hypothesis": "The model gives unstable answers across resamples.", '
    '"failure_mode": "self_consistency"}]'
)

#: The same claim WITH the commitment to how it could be wrong.
WITH_TEST_RESPONSE = (
    '[{"hypothesis": "The model gives unstable answers across resamples.", '
    '"failure_mode": "self_consistency", "test": "self_consistency.consistency"}]'
)


def _batch(n_fail: int = 4, n_pass: int = 4, **slots) -> CaseBatch:
    """A labelled batch. No explicit ids anywhere — matching how cases arrive
    from a manifest, and the reason nothing downstream can rely on one."""
    cases = [
        FailureCase(inputs=Inputs(prompt=f"question {i}", **slots),
                    expected="yes", label=Label.FAIL)
        for i in range(n_fail)
    ]
    cases += [
        FailureCase(inputs=Inputs(prompt=f"question {i}", **slots),
                    expected="yes", observed="yes", label=Label.PASS)
        for i in range(n_pass)
    ]
    return CaseBatch(cases)


def _endpoint_like_model() -> FakeModel:
    """A model shaped like the api backend's.

    ``compose(key, "api")`` reports ``caps=['generate', 'tool_calls']`` for every
    model, because the OpenAI chat schema has a tools field. Nothing about that
    says the run is an agent run.
    """
    return FakeModel(
        capabilities={Capability.GENERATE, Capability.LOGPROBS, Capability.TOOL_CALLS},
        modalities={"text"},
    )


def _run(tmp_path, judge_response: str, model=None, cases=None):
    with RunContext(tmp_path / "run") as ctx:
        VLDiagnoseLoop(
            model=model or _endpoint_like_model(),
            protocol=ExperimentProtocol(
                description="Does the model answer these consistently?",
                task_domain="reasoning",
            ),
            diagnosis_agent=DiagnosisAgent(judge=ScriptedModel([judge_response])),
            max_cycles=1, run_logger=ctx.logger,
        ).run(cases if cases is not None else _batch())
    return {p.name: p for p in (ctx.root / "contract").glob("*.json")}


def _latest(files, stage, wire):
    names = sorted(n for n in files if n.endswith(f".{stage}.json"))
    assert names, f"no {stage} payload; got {sorted(files)}"
    return wire.model_validate_json(files[names[-1]].read_text())


# ── a judge that proposes no test ────────────────────────────────────────────

def test_a_judge_that_proposes_no_test_still_produces_a_coherent_run(tmp_path):
    files = _run(tmp_path, NO_TEST_RESPONSE)
    assert not [n for n in files if n.endswith(".invalid.json")], sorted(files)

    m3 = _latest(files, "m3", DiagnosisOutput)
    assert m3.hypotheses, "the judge did propose something"
    h = m3.hypotheses[0]

    # The claim is reported as untestable rather than dressed in a directive
    # that would make it read as routable to every downstream reader.
    assert h.test_design == ""
    assert h.is_routable is False
    assert m3.untestable == [h.id]


def test_the_m3_m5_join_closes_when_no_id_was_ever_assigned(tmp_path):
    """Hypothesis.id is "" and nothing in the loop fills it.

    Each stage therefore derives one, and on the live run they derived
    DIFFERENTLY -- M3 said "h0", M5 said "unknown" -- so the verdict pointed at
    no claim. Downstream that is indistinguishable from "not adjudicated yet",
    which is the silent-join failure the contract exists to expose.
    """
    files = _run(tmp_path, NO_TEST_RESPONSE)
    m3 = _latest(files, "m3", DiagnosisOutput)
    m5 = _latest(files, "m5", HypothesisTestOutput)

    m3_ids = {h.id for h in m3.hypotheses}
    m5_ids = {r.hypothesis_id for r in m5.results}
    assert m5_ids, "M5 adjudicated nothing"
    assert m5_ids <= m3_ids, f"verdicts point at unknown claims: {m5_ids - m3_ids}"
    assert "unknown" not in m5_ids


def test_every_verdict_is_attributable_to_whether_the_claim_was_testable(tmp_path):
    """The join is what lets a reader qualify ANY verdict on an untestable claim.

    Not only an inconclusive one: see the characterisation test below, where the
    untestable claim comes back SUPPORTED. Whatever the status, a reader has to
    be able to ask "was this claim ever decidable" and get an answer, and the
    only thing that can answer it is matching the verdict back to its proposal.
    """
    files = _run(tmp_path, NO_TEST_RESPONSE)
    m3 = _latest(files, "m3", DiagnosisOutput)
    m5 = _latest(files, "m5", HypothesisTestOutput)

    untestable = set(m3.untestable)
    assert untestable, "this fixture is supposed to produce an untestable claim"
    attributable = [r for r in m5.results if r.hypothesis_id in untestable]
    assert attributable, "no verdict could be traced back to the claim it judges"
    for r in attributable:
        # The provenance of the evidence is on the wire too, so a reader can see
        # WHICH machinery produced the verdict, not just what it decided.
        assert r.evidence.source in ("m2_stats_results", "fallback_per_case", "none")


def test_omitting_the_test_design_is_currently_the_more_permissive_path(tmp_path):
    """Characterisation, not endorsement: this asymmetry looks inverted.

    Same batch, same analyzers, same statistics. The ONLY difference is whether
    the judge said in advance how its claim could be wrong:

        no TEST: line  -> SUPPORTED     (verdict_fallback searches every signal)
        a TEST: line   -> INCONCLUSIVE  (the named test did not run, and
                                         hypothesis_tester refuses substitutes)

    The strict half is deliberate and documented in hypothesis_tester: "An
    explicit M3 test design is a preregistration, not a hint." The consequence
    is what looks unintended -- a judge that skips the preregistration gets the
    EASIER route to SUPPORTED, so the design rewards exactly the laziness it is
    trying to discipline, and a cheap judge on low effort takes that route by
    default.

    Pinned here so the behaviour is visible and cannot change silently. Whether
    to close it is a decision about the pipeline's evidential standard, not a
    bug fix -- so this test asserts what the code does today, and will fail
    loudly the day someone changes it.
    """
    lax = _latest(_run(tmp_path / "lax", NO_TEST_RESPONSE), "m5", HypothesisTestOutput)
    strict = _latest(_run(tmp_path / "strict", WITH_TEST_RESPONSE), "m5", HypothesisTestOutput)

    assert [r.status.value for r in lax.results] == ["supported"]
    assert [r.status.value for r in strict.results] == ["inconclusive"]
    assert strict.results[0].evidence_grade.value == "none"


def test_a_judge_that_does_propose_a_test_is_routable(tmp_path):
    """The control: the same pipeline, one field better, and the claim is testable."""
    files = _run(tmp_path, WITH_TEST_RESPONSE)
    m3 = _latest(files, "m3", DiagnosisOutput)
    assert m3.hypotheses[0].is_routable is True
    assert m3.untestable == []


# ── a model that declares tool calling ───────────────────────────────────────

def test_declared_tool_calling_does_not_make_it_an_agent_run(tmp_path):
    """Every OpenAI-compatible chat model declares tools.

    Reading that as `is_agent` labelled a single-turn text run "llm+agent" and
    told the reader trajectories had been analysed when the batch had none.
    """
    files = _run(tmp_path, WITH_TEST_RESPONSE)
    sel = _latest(files, "m1", ProbeOutput).selection
    assert sel.is_agent is False
    assert sel.profile == "llm"
    assert sel.routed_on == ["text"]


def test_a_batch_that_does_carry_trajectories_is_an_agent_run(tmp_path):
    """The control: is_agent follows the DATA, so real trajectories set it."""
    from evalvitals.core.case import Step, StepRole, Trajectory

    cases = _batch()
    for i, case in enumerate(cases):
        case.trajectory = Trajectory(
            sample_id=case.id, goal=case.inputs.prompt,
            steps=[Step(idx=0, role=StepRole.PLANNER, content="plan"),
                   Step(idx=1, role=StepRole.ACTOR, content="answer")],
            outcome=case.label,
        )
    files = _run(tmp_path, WITH_TEST_RESPONSE, cases=cases)
    sel = _latest(files, "m1", ProbeOutput).selection
    assert sel.is_agent is True
    assert sel.profile == "llm+agent"


# ── a judge that fails outright ──────────────────────────────────────────────

def test_an_empty_judge_response_does_not_corrupt_the_contract(tmp_path):
    """A rate-limited judge returns "". The run should degrade, not lie."""
    files = _run(tmp_path, "")
    assert not [n for n in files if n.endswith(".invalid.json")], sorted(files)
    m3 = _latest(files, "m3", DiagnosisOutput)
    # Zero hypotheses is a completed, empty result -- not an error, and not a
    # reason to invent one.
    assert m3.hypotheses == []
    assert m3.status.state.value in ("empty", "succeeded")


def test_a_malformed_judge_response_does_not_corrupt_the_contract(tmp_path):
    files = _run(tmp_path, "I think the model is just bad at this, honestly.")
    assert not [n for n in files if n.endswith(".invalid.json")], sorted(files)
    _latest(files, "m3", DiagnosisOutput)   # decodes, whatever it contains


# ── M2 must actually ship its statistics ─────────────────────────────────────

def test_m2_serializes_the_typed_verdicts_not_the_legacy_summaries():
    """StatsAnalysisReport carries two similarly named lists with different shapes.

        stats_results       list[StatsToolResult]  -- the per-tool verdicts
        stats_tool_results  list[dict]             -- legacy JSON-safe summaries,
                                                      keyed by `name`, no `tool`

    Reading the second and filtering on `tool` dropped every result. A live run
    shipped M2 with zero statistics next to corrected_rejections.n_tested=16 --
    and "nothing was tested" is a legitimate state, so nothing downstream could
    tell the empty payload was a field mix-up.
    """
    from evalvitals.analysis.stats_agent import StatsAnalysisReport
    from evalvitals.analysis.stats_tools import StatsToolResult
    from evalvitals.contract.emit import from_stats_report

    report = StatsAnalysisReport(
        model_name="m",
        stats_results=[
            StatsToolResult(tool="signal_label_assoc", ok=True, effect=0.4, reject=True,
                            analysis_key="signal_label_assoc:self_consistency.consistency"),
            StatsToolResult(tool="mcnemar_evalue", ok=True, effect=0.1, reject=False),
        ],
        # Populated too, and first in the old lookup order -- the whole trap.
        stats_tool_results=[{"name": "scalar_summary", "metrics": {"n_scalar_metrics": 52}}],
        corrected_rejections={"method": "BH", "alpha": 0.05, "n_tested": 2,
                              "rejected_result_keys": []},
    )
    wire = from_stats_report(report, trace_id="t", cycle=0)
    assert [r.tool for r in wire.stats_results] == ["signal_label_assoc", "mcnemar_evalue"]
    assert wire.corrected_rejections.n_tested == 2


def test_m2_reports_partial_when_tested_results_do_not_reach_the_payload():
    """The failure that hid: n_tested > 0 with an empty results list.

    Both halves are individually valid, so only their combination is evidence of
    a plumbing fault -- and it has to be visible as one rather than as a stage
    that succeeded with nothing to say.
    """
    from evalvitals.analysis.stats_agent import StatsAnalysisReport
    from evalvitals.contract.emit import from_stats_report

    wire = from_stats_report(
        StatsAnalysisReport(
            model_name="m", stats_results=[],
            corrected_rejections={"method": "BH", "n_tested": 16, "rejected_result_keys": []},
        ),
        trace_id="t", cycle=0,
    )
    assert wire.status.state.value == "partial"
    assert "16 tests" in (wire.status.reason or "")

    # Genuinely nothing tested stays EMPTY -- the states must not collapse.
    quiet = from_stats_report(
        StatsAnalysisReport(model_name="m", stats_results=[],
                            corrected_rejections={"method": "none", "n_tested": 0}),
        trace_id="t", cycle=0,
    )
    assert quiet.status.state.value == "empty"


# ── the payload has to be readable by a person ───────────────────────────────

def test_a_model_is_named_not_repred():
    """The UI showed `<videollama2_model.MockAVModel object at 0x795d025e8590>`.

    Unreadable, and worse: the address changes every run, so two runs of the
    same model record two different identities and nothing compares across them.
    """
    from evalvitals.contract import ModelRef
    from evalvitals.contract.emit import model_name

    class MockAVModel:
        pass

    class Named:
        display_name = "VideoLLaMA2.1-7B-AV"

    class Composed:
        class spec:
            key = "qwen3.5-2b"

    assert model_name(MockAVModel()) == "MockAVModel"      # stable fallback
    assert model_name(Named()) == "VideoLLaMA2.1-7B-AV"    # what a producer should set
    assert model_name(Composed()) == "qwen3.5-2b"
    assert model_name("<foo.Bar object at 0x7f00>") == "unknown model"

    # And the contract refuses to record one, so this cannot regress quietly.
    with pytest.raises(ValueError):
        ModelRef(name="<videollama2_model.MockAVModel object at 0x795d025e8590>")


def test_m1_output_carries_the_model_identity(tmp_path):
    """A reader opens c0.m1.json with no ProbeInput beside it."""
    files = _run(tmp_path, WITH_TEST_RESPONSE)
    m1 = _latest(files, "m1", ProbeOutput)
    assert m1.model is not None
    assert m1.model.name and "object at 0x" not in m1.model.name
    assert m1.model.modalities == ["text"]


def test_every_statistical_result_carries_a_distinct_human_label():
    """A chart axis needs a name for the SUBJECT, unique across rows.

    `tool` is the procedure -- labelling with it put two bars both reading
    "Mcnemar evalue" on one chart with nothing to tell them apart. `config
    ['signal']` is the subject but is a machine name, and paired tools carry
    none at all.
    """
    from evalvitals.analysis.stats_agent import StatsAnalysisReport
    from evalvitals.analysis.stats_tools import StatsToolResult
    from evalvitals.contract.emit import from_stats_report, measured_label

    class _R:
        def __init__(self, tool, cfg): self.tool, self.config = tool, cfg

    # The analyzer's own short name wins over the identifier -- it is written
    # for a reader, and a bar chart's label column truncates anything longer.
    assert measured_label(_R("signal_label_assoc", {"signal": "answer_extraction_audit.output_chars"})) \
        == "Answer length"
    # An undocumented metric still gets a distinct label, just an unhelpful one:
    # jargon that admits it is jargon, never a paraphrase posing as an explanation.
    assert measured_label(_R("signal_label_assoc", {"signal": "contamination_score.guided_gain"})) \
        == "guided gain (contamination score)"
    assert measured_label(_R("mcnemar_evalue", {"strategy": "describe_first"})) \
        == "describe first vs baseline"
    # No subject named anywhere: say so, rather than borrowing the tool's name.
    assert measured_label(_R("mcnemar_evalue", {})).startswith("unnamed contrast")

    # Collisions are separated, because two identically labelled rows are two
    # rows a reader cannot distinguish.
    wire = from_stats_report(
        StatsAnalysisReport(
            model_name="m",
            stats_results=[StatsToolResult(tool="mcnemar_evalue", ok=True) for _ in range(3)],
        ),
        trace_id="t", cycle=0,
    )
    labels = [r.measured for r in wire.stats_results]
    assert len(set(labels)) == 3, labels


# ── M4's tier is an enum in the pipeline and a string on the wire ────────────

def test_a_fix_tier_enum_serialises_to_its_wire_spelling():
    """FixTier is an IntEnum: .value is an ordinal, .name is L3A_INTERNALS_READ.

    Neither is what the contract asks for. A live run passed the enum straight
    through and the whole M4 payload was rejected -- the repair results were
    lost to a spelling. `.label` is exactly the wire form and was already there.
    """
    from evalvitals.contract.emit import _tier
    from evalvitals.eval_agent.stages.fix_tiers import FixTier

    assert _tier(FixTier.L3A_INTERNALS_READ) == "L3a"
    assert _tier(FixTier.L0_RUNTIME_CONFIG) == "L0"
    assert _tier(FixTier.L4_PARAMETERS) == "L4"
    assert _tier("L2") == "L2"          # already a string
    assert _tier(None) == "L1"          # documented default


def test_every_tier_the_pipeline_has_is_representable():
    """L0 was missing from the contract, so a runtime-config repair -- the least
    invasive kind, and one FixTier defines -- could not be reported at all."""
    from evalvitals.contract import FixOutput
    from evalvitals.contract.emit import _tier
    from evalvitals.eval_agent.stages.fix_tiers import FixTier

    for tier in FixTier:
        FixOutput(
            schema_version=SCHEMA_VERSION, trace_id="t", produced_at="2026-08-23T00:00:00Z",
            status={"stage": "m4_fix", "state": "succeeded", "cycle": -1},
            max_tier=_tier(tier),
        )


def test_a_paired_contrast_is_named_by_its_arms():
    """Paired tools put their arms in `strategies`, a list.

    Checking only the singular `strategy` left the three strongest results of an
    audio-visual run -- the without_audio / without_video / describe_first
    contrasts, down to p=1.8e-29 -- all reading "unnamed contrast" and therefore
    indistinguishable from each other on a chart. These are the INTERVENTION
    grade evidence; they are the last rows that should be unreadable.
    """
    from evalvitals.contract.emit import measured_label

    class _R:
        tool = "mcnemar_evalue"
        def __init__(self, cfg): self.config = cfg

    # The reference arm reads second, so the label leads with what changed.
    assert measured_label(_R({"strategies": ["baseline", "without_audio"]})) \
        == "without audio vs baseline"
    assert measured_label(_R({"strategies": ["baseline", "describe_first"]})) \
        == "describe first vs baseline"
    # Neither arm is the baseline: keep the stated order.
    assert measured_label(_R({"strategies": ["describe_first", "sensitive"]})) \
        == "describe first vs sensitive"
    # Still honest when nothing names the subject.
    assert measured_label(_R({})).startswith("unnamed contrast")


# ── M4 must report the search, not only its winner ───────────────────────────

def test_the_selection_sweep_reaches_the_wire():
    """A run that swept seven candidates and confirmed one reported one row.

    FixOutcome carries `selection_attempted`; FixOutput had no field for it, so
    everything the search RULED OUT lived only in a markdown file. On the live
    audio-visual run that meant four L2 candidates were invisible and every
    reader concluded L2 had never been attempted.
    """
    from types import SimpleNamespace

    from evalvitals.contract.emit import from_fix_outcome

    outcome = SimpleNamespace(
        max_tier="L3a", routed=[], attempted=[], best=None, fixed=False,
        ebh_survivors=[], repair_rounds=1, recommendation=None, refine_signal=None,
        selected_on_explore="visual_grounding",
        selection_attempted=[
            {"name": "visual_grounding", "tier": "L1", "n_pairs": 60,
             "n_fixed": 8, "n_broken": 0, "effect": 0.1333, "verdict": "fixed"},
            {"name": "audio_evidence_then_answer", "tier": "L1", "n_pairs": 60,
             "n_fixed": 7, "n_broken": 8, "effect": -0.0167, "verdict": "unsafe"},
            {"name": "coded_pipeline", "tier": "L2", "n_pairs": 60,
             "n_fixed": 8, "n_broken": 3, "effect": 0.0833, "verdict": "partial"},
        ],
    )
    wire = from_fix_outcome(outcome, trace_id="t")

    assert [r.tier for r in wire.selection] == ["L1", "L1", "L2"]
    assert wire.selected_on_explore == "visual_grounding"
    # A candidate that made things worse is a result and must survive to the wire.
    unsafe = next(r for r in wire.selection if r.verdict == "unsafe")
    assert (unsafe.n_fixed, unsafe.n_broken) == (7, 8)
    # Selection rows carry no confirmation statistics, so they cannot be misread
    # as evidence: they were never validated on held-out cases.
    assert all(r.e_value is None and not r.reject for r in wire.selection)
    # And they stay out of the evidential list.
    assert wire.attempted == []
