"""The held-out cases must be recorded, or the strongest evidence has no cases.

With a confirm split, `run()` reassigns `data` to the explore half and then logs
`data`. The held-out half was never logged -- and that is the half M5 adjudicates
on and M4 validates its repair on.

Consequence, seen on a live audio-visual run: M4 reported "12 repaired, 1 broken"
and named thirteen case ids, and not one of them appeared in the report. Worse,
`FailureCase.id` defaults to a fresh uuid4, so those ids existed only inside that
process: unlogged means unrecoverable, permanently. The run's best-supported
claim became the one nobody can inspect a single case of.
"""

from __future__ import annotations

import json

from evalvitals.core.capability import Capability
from evalvitals.core.case import CaseBatch, FailureCase, Inputs, Label
from evalvitals.eval_agent import DiagnosisAgent, RunContext, VLDiagnoseLoop
from evalvitals.eval_agent.stages.protocol import ExperimentProtocol
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
    for line in (root / "run_log.jsonl").read_text().splitlines():
        e = json.loads(line)
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
        "split is what M5 and M4 are measured on; ids are per-process uuids, so "
        "unlogged is unrecoverable."
    )


def test_logging_a_case_twice_does_not_duplicate_it(tmp_path):
    """run() now logs both splits; the logger must stay idempotent."""
    cases = _batch(8)
    with RunContext(tmp_path / "run") as ctx:
        ctx.logger.log_cases(cases)
        ctx.logger.log_cases(cases)
    ids = [
        json.loads(line)["case"]["id"]
        for line in (ctx.root / "run_log.jsonl").read_text().splitlines()
        if json.loads(line).get("event") == "case_record"
    ]
    assert len(ids) == len(set(ids)) == 8
