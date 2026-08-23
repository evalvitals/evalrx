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
    DiagnosisOutput, HypothesisTestOutput, ProbeOutput,
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
