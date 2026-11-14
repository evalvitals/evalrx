"""RunLogger.log_probe <-> Result.image_overlays() wiring.

The bare attention heatmaps RunLogger saves have no spatial reference to the
actual image; Result subclasses (e.g. RelativeAttentionResult) can define an
``image_overlays(fig_dir, stem_prefix) -> list[Path]`` hook to additionally
save heatmap-on-image visualisations, which must flow into log_probe's
returned figure list the same way the bare heatmaps do — that returned list
is what gets forwarded to a multimodal judge (see stats_agent.py).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest


@dataclass
class _FakeResultWithOverlays:
    analyzer: str = "fake_analyzer"
    model: str = "FakeModel()"
    findings: dict = field(default_factory=lambda: {"n_cases_analyzed": 1})
    artifacts: dict = field(default_factory=dict)
    cases: Any = None

    def image_overlays(self, fig_dir, stem_prefix) -> "list[Path]":
        path = Path(fig_dir) / f"{stem_prefix}_fake_overlay.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"\x89PNG\r\n\x1a\n")
        return [path]


@dataclass
class _FakeResultOverlaysRaise:
    analyzer: str = "broken_analyzer"
    model: str = "FakeModel()"
    findings: dict = field(default_factory=dict)
    artifacts: dict = field(default_factory=dict)
    cases: Any = None

    def image_overlays(self, fig_dir, stem_prefix) -> "list[Path]":
        raise RuntimeError("boom")


@dataclass
class _FakeResultNoOverlays:
    analyzer: str = "plain_analyzer"
    model: str = "FakeModel()"
    findings: dict = field(default_factory=dict)
    artifacts: dict = field(default_factory=dict)
    cases: Any = None


def test_log_probe_returns_overlay_png_from_result_hook(tmp_path):
    from evalrx.eval_agent.run_logger import RunLogger

    logger = RunLogger(run_dir=tmp_path / "run1")
    pngs = logger.log_probe(0, {"fake_analyzer": _FakeResultWithOverlays()})
    assert len(pngs) == 1
    assert pngs[0].name == "c0_fake_analyzer_fake_overlay.png"
    assert pngs[0].exists()
    logger.close()


def test_log_probe_survives_image_overlays_raising(tmp_path):
    from evalrx.eval_agent.run_logger import RunLogger

    logger = RunLogger(run_dir=tmp_path / "run1")
    pngs = logger.log_probe(0, {"broken_analyzer": _FakeResultOverlaysRaise()})
    assert pngs == []
    logger.close()


def test_log_probe_without_image_overlays_hook_is_unaffected(tmp_path):
    from evalrx.eval_agent.run_logger import RunLogger

    logger = RunLogger(run_dir=tmp_path / "run1")
    pngs = logger.log_probe(0, {"plain_analyzer": _FakeResultNoOverlays()})
    assert pngs == []
    logger.close()


def test_log_probe_combines_overlay_and_npy_heatmap_pngs(tmp_path):
    """Overlay PNGs and the existing bare-heatmap PNGs (from numeric artifacts)
    must both end up in the returned figure list — the judge should see both."""
    pytest.importorskip("matplotlib")
    import numpy as np

    from evalrx.eval_agent.run_logger import RunLogger

    @dataclass
    class _Mixed:
        analyzer: str = "mixed_analyzer"
        model: str = "FakeModel()"
        findings: dict = field(default_factory=dict)
        artifacts: dict = field(
            default_factory=lambda: {"attention_map": np.random.rand(4, 4)}
        )
        cases: Any = None

        def image_overlays(self, fig_dir, stem_prefix) -> "list[Path]":
            path = Path(fig_dir) / f"{stem_prefix}_overlay.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"\x89PNG\r\n\x1a\n")
            return [path]

    logger = RunLogger(run_dir=tmp_path / "run1")
    pngs = logger.log_probe(0, {"mixed_analyzer": _Mixed()})
    names = {p.name for p in pngs}
    assert "c0_mixed_analyzer_overlay.png" in names
    assert "c0_mixed_analyzer_attention_map.png" in names
    logger.close()
