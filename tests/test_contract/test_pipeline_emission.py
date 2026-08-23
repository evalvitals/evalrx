"""M1-M5 emit contract-validated payloads for a real loop run.

The contract described the stage boundaries and nothing called it, so nothing
detected when the description and the pipeline disagreed. These tests close that
loop: a loop runs end to end against fake models, and every file it drops in
``<run>/contract/`` is re-validated from disk by the same wire models — which is
what a frontend does, so a green test here means a frontend can decode it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from evalvitals.core.capability import Capability
from evalvitals.core.case import CaseBatch, FailureCase, Inputs, Label
from evalvitals.eval_agent import DiagnosisAgent, RunContext, VLDiagnoseLoop
from evalvitals.eval_agent.stages.protocol import ExperimentProtocol
from tests.conftest import FakeModel
from tests.test_eval_agent.test_vl_diagnose import ScriptedModel

pytest.importorskip("pydantic")

from evalvitals.contract import (  # noqa: E402
    SCHEMA_VERSION, DiagnosisOutput, HypothesisTestOutput, ProbeOutput, StatsReportWire,
)

STAGE_MODELS = {
    "m1": ProbeOutput, "m2": StatsReportWire, "m3": DiagnosisOutput, "m5": HypothesisTestOutput,
}


def _batch(n_fail: int = 3, n_pass: int = 3, **slots) -> CaseBatch:
    cases = [
        FailureCase(id=f"f{i}", inputs=Inputs(prompt=f"question {i}", **slots),
                    expected="yes", label=Label.FAIL)
        for i in range(n_fail)
    ]
    cases += [
        FailureCase(id=f"p{i}", inputs=Inputs(prompt=f"question {i}", **slots),
                    expected="yes", observed="yes", label=Label.PASS)
        for i in range(n_pass)
    ]
    return CaseBatch(cases)


def _judge() -> DiagnosisAgent:
    return DiagnosisAgent(judge=ScriptedModel([
        '[{"hypothesis": "The model ignores the modality under test.", '
        '"failure_mode": "attention", "test": "attention.entropy"}]'
    ]))


def _run(tmp_path, model, cases) -> "tuple[dict, RunContext]":
    protocol = ExperimentProtocol(
        description="Does the model use the modality it was given?",
        task_domain="perception",
    )
    with RunContext(tmp_path / "run") as ctx:
        loop = VLDiagnoseLoop(
            model=model, protocol=protocol, diagnosis_agent=_judge(),
            max_cycles=1, run_logger=ctx.logger,
        )
        loop.run(cases)
    files = {p.name: p for p in (ctx.root / "contract").glob("*.json")}
    return files, ctx


# ── every stage emits, and every payload round-trips ──────────────────────────

def test_every_stage_emits_a_payload_that_revalidates(tmp_path):
    model = FakeModel(capabilities={Capability.GENERATE, Capability.ATTENTION,
                                    Capability.LOGPROBS},
                      modalities={"text", "image"})
    files, _ = _run(tmp_path, model, _batch(image="scene.png"))

    for stage, wire in STAGE_MODELS.items():
        name = f"c0.{stage}.json"
        assert name in files, f"M{stage[-1]} emitted nothing; got {sorted(files)}"
        # Decoded from disk by the contract itself — the frontend's exact path.
        payload = wire.model_validate_json(files[name].read_text())
        assert payload.schema_version == SCHEMA_VERSION
        assert payload.status.stage == stage
        assert payload.trace_id


def test_no_stage_wrote_an_invalid_marker(tmp_path):
    model = FakeModel(capabilities={Capability.GENERATE, Capability.LOGPROBS},
                      modalities={"text", "image"})
    with_run = tmp_path / "run" / "contract"
    _run(tmp_path, model, _batch(image="scene.png"))
    invalid = sorted(p.name for p in with_run.glob("*.invalid.json"))
    assert not invalid, f"stages failed validation: {invalid}"


# ── the modality record is what makes the payload readable per model family ───

@pytest.mark.parametrize(
    "modalities,slots,expected_routed,expected_profile",
    [
        ({"text"},                          {},                                    ["text"],                  "llm"),
        ({"text", "image"},                 {"image": "a.png"},                    ["image", "text"],         "vlm"),
        ({"text", "audio"},                 {"audio": "a.wav"},                    ["audio", "text"],         "alm"),
        ({"text", "image", "audio"},        {"audio": "a.wav", "image": "a.png"},  ["audio", "image", "text"], "avlm"),
        # The case the old per-kind enum could not express: an omni model
        # evaluated on audio must be routed as audio, not as everything it can do.
        ({"text", "image", "audio", "video"}, {"audio": "a.wav"},                  ["audio", "text"],         "alm"),
    ],
)
def test_m1_records_which_modality_the_run_was_routed_on(
    tmp_path, modalities, slots, expected_routed, expected_profile,
):
    model = FakeModel(capabilities={Capability.GENERATE, Capability.LOGPROBS},
                      modalities=set(modalities))
    files, _ = _run(tmp_path, model, _batch(**slots))
    sel = ProbeOutput.model_validate_json(files["c0.m1.json"].read_text()).selection

    assert sel.model_modalities == sorted(modalities)
    assert sel.routed_on == expected_routed
    assert sel.profile == expected_profile
    # Declared and routed diverge exactly when the batch narrows the model.
    assert set(sel.probed_modalities) <= set(sel.model_modalities) | {"text"}


def test_index_lists_what_was_written(tmp_path):
    model = FakeModel(capabilities={Capability.GENERATE}, modalities={"text"})
    files, ctx = _run(tmp_path, model, _batch())
    from evalvitals.contract.emit import ContractEmitter

    emitter = ContractEmitter(ctx.root, "t")
    emitter.written = [p for n, p in files.items()]
    index = json.loads(emitter.index().read_text())
    assert index["schema_version"] == SCHEMA_VERSION
    assert index["stages"]


# ── the emitter is an observer: it must never take a run down ─────────────────

def test_a_broken_payload_is_recorded_not_raised(tmp_path):
    from evalvitals.contract.emit import ContractEmitter

    emitter = ContractEmitter(tmp_path, "trace")

    def explode():
        raise ValueError("this analyzer emitted nonsense")

    assert emitter.emit("c0.m1", explode) is None
    recorded = json.loads((tmp_path / "contract" / "c0.m1.invalid.json").read_text())
    assert "nonsense" in recorded["error"]
    assert emitter.errors == [("c0.m1", "this analyzer emitted nonsense")]


def test_strict_mode_raises_for_ci(tmp_path):
    from evalvitals.contract.emit import ContractEmitter

    emitter = ContractEmitter(tmp_path, "trace", strict=True)
    with pytest.raises(ValueError):
        emitter.emit("c0.m1", lambda: (_ for _ in ()).throw(ValueError("boom")))


# ── generated artifacts must stay in step with the Python source ──────────────

def test_generated_artifacts_are_current(tmp_path):
    """The committed schema/TypeScript match the models they were generated from.

    Without this, a field added in Python and not regenerated is invisible to the
    frontend, and the failure surfaces as a runtime `undefined` in a browser
    rather than a red build.
    """
    from evalvitals.contract.export import _frontend_targets, export

    fresh = export(tmp_path, frontend=False)
    committed = Path("docs/contract")
    for produced in fresh:
        targets = [committed / produced.name]
        if produced.name == "contract.d.ts":
            targets += _frontend_targets()
        for target in targets:
            assert target.exists(), (
                f"{target} is missing; run "
                f"`python -m evalvitals.contract.export --out docs/contract`"
            )
            assert target.read_text() == produced.read_text(), (
                f"{target} is stale; run "
                f"`python -m evalvitals.contract.export --out docs/contract`"
            )


def test_report_data_carries_the_contract_payloads(tmp_path):
    """The dashboard's data model exposes what the pipeline emitted."""
    from evalvitals.reporting.dynamic import build_report_data

    model = FakeModel(capabilities={Capability.GENERATE, Capability.LOGPROBS},
                      modalities={"text", "audio"})
    _, ctx = _run(tmp_path, model, _batch(audio="clip.wav"))
    data = build_report_data(ctx.root)

    contract = data["contract"]
    assert "c0.m1" in contract, sorted(contract)
    selection = contract["c0.m1"]["selection"]
    assert selection["routed_on"] == ["audio", "text"]
    # The frontend renders from this without re-deriving the shape.
    assert contract["c0.m3"]["hypotheses"]
