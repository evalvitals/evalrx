"""StrategyProbe.detect_kind — trajectory-carrying data flips a VLM to AGENT."""

from __future__ import annotations

import evalvitals.analyzers  # noqa: F401  (populate the analyzer registry)
from evalvitals.core.capability import Capability
from evalvitals.core.case import CaseBatch, FailureCase, Inputs, Step, StepRole, Trajectory
from evalvitals.core.model import Model
from evalvitals.eval_agent.stages.probe import ModelKind, StrategyProbe


class _FakeVLM(Model):
    capabilities = frozenset({Capability.GENERATE, Capability.TOOL_CALLS})
    modalities = frozenset({"text", "image"})

    def generate(self, inputs, **kw):
        return ""

    def forward(self, inputs, capture, spec=None):
        raise NotImplementedError


def _batch(with_trajectory: bool) -> CaseBatch:
    trajectory = None
    if with_trajectory:
        trajectory = Trajectory(
            sample_id="s",
            goal="g",
            steps=[Step(idx=0, role=StepRole.USER, content="g")],
        )
    return CaseBatch([FailureCase(inputs=Inputs(prompt="g"), trajectory=trajectory)])


def test_vlm_without_trajectories_stays_vlm():
    probe = StrategyProbe()
    assert probe.detect_kind(_FakeVLM()) is ModelKind.VLM
    assert probe.detect_kind(_FakeVLM(), _batch(False)) is ModelKind.VLM


def test_trajectory_data_flips_to_agent():
    assert StrategyProbe().detect_kind(_FakeVLM(), _batch(True)) is ModelKind.AGENT


def test_select_with_trajectory_data_leads_with_agent_analyzers():
    ranked = StrategyProbe().select(_FakeVLM(), data=_batch(True))
    assert ranked[0] == "loop_detect"
    assert ranked.index("ignored_obs") == 1


def test_select_without_data_keeps_vlm_ranking():
    ranked = StrategyProbe().select(_FakeVLM(), data=_batch(False))
    assert ranked and ranked[0] != "loop_detect"


def test_non_iterable_data_is_ignored():
    assert StrategyProbe().detect_kind(_FakeVLM(), data=42) is ModelKind.VLM
