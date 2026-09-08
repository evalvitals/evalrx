"""The held-out cases must be recorded, or the strongest evidence has no cases.

With a confirm split, `run()` reassigns `data` to the explore half and then logs
`data`. The held-out half was never logged -- and that is the half M4 adjudicates
on and M5 validates its repair on.

Consequence, seen on a live audio-visual run: M5 reported "12 repaired, 1 broken"
and named thirteen case ids, and not one of them appeared in the report. Worse,
`FailureCase.id` defaults to a fresh uuid4, so those ids existed only inside that
process: unlogged means unrecoverable, permanently. The run's best-supported
claim became the one nobody can inspect a single case of.
"""

from __future__ import annotations

from evalrx.core.capability import Capability
from evalrx.core.case import CaseBatch, FailureCase, Inputs, Label
from evalrx.eval_agent import DiagnosisAgent, RunContext, VLDiagnoseLoop
from evalrx.eval_agent.stages.protocol import ExperimentProtocol
from evalrx.reporting.run_events import read_v2_events
from tests.conftest import FakeModel
from tests.test_eval_agent.test_vl_diagnose import ScriptedModel


def _batch(n: int = 20) -> CaseBatch:
    cases = []
    for i in range(n):
        cases.append(FailureCase(
            inputs=Inputs(prompt=f"question {i}"), expected="yes",
            observed="yes" if i % 2 else "no",
            label=Label.PASS if i % 2 else Label.FAIL,
        ))
    return CaseBatch(cases)


def _logged_ids(root) -> set[str]:
    ids = set()
    for e in read_v2_events(root):
        if e.get("event") == "case_record":
            ids.add(e["case"]["id"])
    return ids


def test_the_held_out_split_is_recorded_too(tmp_path):
    cases = _batch(20)
    all_ids = {c.id for c in cases}
    with RunContext(tmp_path / "run") as ctx:
        VLDiagnoseLoop(
            model=FakeModel(capabilities={Capability.GENERATE}, modalities={"text"}),
            protocol=ExperimentProtocol(description="does it answer?", task_domain="qa"),
            diagnosis_agent=DiagnosisAgent(judge=ScriptedModel([
                '[{"hypothesis":"it is unstable","failure_mode":"x","test":"attention.entropy"}]'
            ])),
            max_cycles=1, run_logger=ctx.logger,
            confirm_split=0.3,
        ).run(cases)

    logged = _logged_ids(ctx.root)
    missing = all_ids - logged
    assert not missing, (
        f"{len(missing)} of {len(all_ids)} cases were never logged. The held-out "
        "split is what M4 and M5 are measured on; ids are per-process uuids, so "
        "unlogged is unrecoverable."
    )


def test_logging_a_case_twice_does_not_duplicate_it(tmp_path):
    """run() now logs both splits; the logger must stay idempotent."""
    cases = _batch(8)
    with RunContext(tmp_path / "run") as ctx:
        ctx.logger.log_cases(cases)
        ctx.logger.log_cases(cases)
    ids = [
        e["case"]["id"] for e in read_v2_events(ctx.root) if e.get("event") == "case_record"
    ]
    assert len(ids) == len(set(ids)) == 8


def test_each_case_record_names_its_partition(tmp_path):
    """Logging every partition is not enough: the record has to say which.

    Without the tag a report sees one flat list and cannot tell the cases
    M1-M3 mined from the ones M4 and M5 were measured on -- the very
    distinction the split exists to make.
    """
    cases = _batch(20)
    with RunContext(tmp_path / "run") as ctx:
        VLDiagnoseLoop(
            model=FakeModel(capabilities={Capability.GENERATE}, modalities={"text"}),
            protocol=ExperimentProtocol(description="does it answer?", task_domain="qa"),
            diagnosis_agent=DiagnosisAgent(judge=ScriptedModel([
                '[{"hypothesis":"it is unstable","failure_mode":"x","test":"attention.entropy"}]'
            ])),
            max_cycles=1, run_logger=ctx.logger,
            confirm_split=0.3,
        ).run(cases)

    by_split: dict[str, set[str]] = {}
    run_start = None
    for e in read_v2_events(ctx.root):
        if e.get("event") == "run_start":
            run_start = e
        if e.get("event") == "case_record":
            by_split.setdefault(e.get("split", "<none>"), set()).add(e["case"]["id"])
    assert set(by_split) == {"explore", "confirm"}, by_split.keys()
    assert len(by_split["explore"]) == 14 and len(by_split["confirm"]) == 6
    assert not (by_split["explore"] & by_split["confirm"])
    # run_start's n_cases is the explore partition, and the record now says
    # how the batch was divided rather than leaving that to be inferred.
    assert run_start is not None
    assert run_start["n_cases"] == 14
    assert run_start["confirm_split"] == 0.3
