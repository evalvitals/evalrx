"""Batch RelativeAttentionAnalyzer: per-case focus signals + FAIL/PASS group maps.

P1 of the depth roadmap: attention analyzers must emit per-case findings so the
M2 stats layer can correlate attention behaviour with PASS/FAIL labels, plus
group-mean / difference spatial maps for visual comparison.
"""

from __future__ import annotations

import pytest
import torch

from evalvitals.analyzers.attention.relative_attn import RelativeAttentionAnalyzer
from evalvitals.core.capability import Capability
from evalvitals.core.case import CaseBatch, FailureCase, Inputs, Label
from evalvitals.core.model import Model, Trace

_SEQ = 8
_IMG_SLICE = slice(2, 6)  # 4 image tokens → spatial (2, 2)


class AttnVLM(Model):
    """Fake VLM with controllable last-row attention over 4 image tokens.

    Prompts containing "FOCUS" peak on the first image patch; any other
    task prompt (and the generic baseline) is uniform — so FOCUS cases get a
    high max relative weight (~4x) and flat cases ~1x.
    """

    capabilities = frozenset({Capability.GENERATE, Capability.ATTENTION})
    modalities = frozenset({"text", "image"})

    def generate(self, inputs, **kwargs):
        return "x"

    def forward(self, inputs, capture, spec=None) -> Trace:
        prompt = str(getattr(inputs, "prompt", inputs))
        row = torch.full((_SEQ,), 0.01)
        if "FOCUS" in prompt:
            row[_IMG_SLICE] = torch.tensor([0.4, 0.05, 0.05, 0.05])
        else:  # generic baseline and "flat" task prompts
            row[_IMG_SLICE] = torch.full((4,), 0.1)
        layer = torch.zeros(2, _SEQ, _SEQ)
        layer[:, -1, :] = row
        mask = torch.zeros(_SEQ, dtype=torch.bool)
        mask[_IMG_SLICE] = True
        return Trace(
            tokens=["t"] * _SEQ,
            token_ids=list(range(_SEQ)),
            provided={Capability.ATTENTION},
            attentions=[layer.clone() for _ in range(3)],
            extras={"image_token_mask": mask, "image_spatial_shape": (2, 2)},
        )


class NoMaskVLM(AttnVLM):
    def forward(self, inputs, capture, spec=None) -> Trace:
        trace = super().forward(inputs, capture, spec)
        trace.extras = {}
        return trace


def _labeled_batch() -> CaseBatch:
    return CaseBatch([
        FailureCase(id="p0", inputs=Inputs(prompt="FOCUS on the lesion", image="<img>"),
                    label=Label.PASS),
        FailureCase(id="p1", inputs=Inputs(prompt="FOCUS on the organ", image="<img>"),
                    label=Label.PASS),
        FailureCase(id="f0", inputs=Inputs(prompt="is there a finding", image="<img>"),
                    label=Label.FAIL),
        FailureCase(id="f1", inputs=Inputs(prompt="is there a lesion", image="<img>"),
                    label=Label.FAIL),
    ])


def test_per_case_signals_cover_batch():
    res = RelativeAttentionAnalyzer().run(AttnVLM(), _labeled_batch())
    pc = res.findings["per_case"]
    assert [e["id"] for e in pc] == ["p0", "p1", "f0", "f1"]
    assert res.findings["n_cases_analyzed"] == 4
    by_id = {e["id"]: e for e in pc}
    # FOCUS (pass) cases peak ~4x; flat (fail) cases ~1x.
    assert by_id["p0"]["max_relative_weight"] > 3.5
    assert by_id["f0"]["max_relative_weight"] < 1.5
    assert 0 < by_id["p0"]["focus_share"] <= 1


def test_group_means_and_diff_map():
    res = RelativeAttentionAnalyzer().run(AttnVLM(), _labeled_batch())
    assert res.findings["fail_mean_max_relative_weight"] < 1.5
    assert res.findings["pass_mean_max_relative_weight"] > 3.5
    diff = res.artifacts["diff_map_fail_minus_pass"]
    assert diff.shape == (2, 2)
    # Fail group is flat (~1.0); pass group peaks (~4.0) on the first patch.
    assert diff.flat[0] < -2.0
    assert res.artifacts["fail_mean_map"].shape == (2, 2)
    assert res.artifacts["pass_mean_map"].shape == (2, 2)


def test_backward_compatible_artifacts_and_findings():
    res = RelativeAttentionAnalyzer().run(AttnVLM(), _labeled_batch())
    # First-case artifacts + findings keys preserved for existing consumers.
    assert res.artifacts["attn_map"].shape == (4,)
    assert res.artifacts["spatial_map"].shape == (2, 2)
    for key in ("n_image_tokens", "n_layers", "layer_used", "map_shape", "top_patches"):
        assert key in res.findings


def test_unlabeled_batch_has_no_group_maps():
    cases = CaseBatch([
        FailureCase(id="u0", inputs=Inputs(prompt="FOCUS here", image="<img>")),
    ])
    res = RelativeAttentionAnalyzer().run(AttnVLM(), cases)
    assert "diff_map_fail_minus_pass" not in res.artifacts
    assert "fail_mean_max_relative_weight" not in res.findings


def test_all_cases_failing_raises_actionable_error():
    import pytest

    with pytest.raises(ValueError, match="image_token_mask"):
        RelativeAttentionAnalyzer().run(NoMaskVLM(), _labeled_batch())


def _labeled_batch_with_real_images() -> CaseBatch:
    """Same shape as _labeled_batch but with real PIL images (overlay needs .convert)."""
    pytest.importorskip("PIL")
    from PIL import Image

    img = Image.new("RGB", (32, 32), color=(100, 110, 120))
    return CaseBatch([
        FailureCase(id="p0", inputs=Inputs(prompt="FOCUS on the lesion", image=img),
                    label=Label.PASS),
        FailureCase(id="f0", inputs=Inputs(prompt="is there a finding", image=img),
                    label=Label.FAIL),
    ])


def test_overlay_anchors_each_map_on_its_recorded_case():
    pytest.importorskip("PIL")
    pytest.importorskip("matplotlib")  # overlay() colormaps the heatmap
    from PIL import Image

    res = RelativeAttentionAnalyzer().run(AttnVLM(), _labeled_batch_with_real_images())
    for key in ("spatial_map", "fail_mean_map", "pass_mean_map", "diff_map_fail_minus_pass"):
        case_id = res.findings[f"{key}_case_id"]
        assert case_id in {"p0", "f0"}
        img = res.overlay(key)
        assert isinstance(img, Image.Image)
        assert img.size == (32, 32)
        assert img.mode == "RGB"


def test_overlay_none_when_image_is_an_unresolvable_string():
    # Existing tests use a placeholder string for `image` (not a real path/URL)
    # — overlay must not raise, just return None.
    res = RelativeAttentionAnalyzer().run(AttnVLM(), _labeled_batch())
    assert res.overlay("spatial_map") is None


def test_overlay_resolves_a_real_file_path_string(tmp_path):
    """Inputs.image may be a lazy path the backend resolves (see Inputs'
    docstring — TextVQASizeDataset stores exactly this); overlay() must
    decode it the same way the model's own forward pass does."""
    pytest.importorskip("PIL")
    pytest.importorskip("transformers")
    pytest.importorskip("matplotlib")  # overlay() colormaps the heatmap
    from PIL import Image

    img_path = tmp_path / "case.png"
    Image.new("RGB", (32, 32), color=(100, 110, 120)).save(img_path)
    cases = CaseBatch([
        FailureCase(id="p0", inputs=Inputs(prompt="FOCUS on the lesion", image=str(img_path)),
                    label=Label.PASS),
        FailureCase(id="f0", inputs=Inputs(prompt="is there a finding", image=str(img_path)),
                    label=Label.FAIL),
    ])
    res = RelativeAttentionAnalyzer().run(AttnVLM(), cases)
    img = res.overlay("spatial_map")
    assert isinstance(img, Image.Image)
    assert img.size == (32, 32)


def test_overlay_none_for_unknown_or_1d_map():
    res = RelativeAttentionAnalyzer().run(AttnVLM(), _labeled_batch_with_real_images())
    assert res.overlay("does_not_exist") is None


def test_save_overlay_writes_a_real_png(tmp_path):
    pytest.importorskip("matplotlib")  # save_overlay() colormaps the heatmap
    res = RelativeAttentionAnalyzer().run(AttnVLM(), _labeled_batch_with_real_images())
    path = tmp_path / "overlay.png"
    assert res.save_overlay("spatial_map", path) is True
    assert path.exists()


def test_image_overlays_writes_one_file_per_available_map(tmp_path):
    pytest.importorskip("matplotlib")  # image_overlays() colormaps each heatmap
    res = RelativeAttentionAnalyzer().run(AttnVLM(), _labeled_batch_with_real_images())
    paths = res.image_overlays(tmp_path, "c0_relative_attention")
    names = {p.name for p in paths}
    assert names == {
        "c0_relative_attention_spatial_map_overlay.png",
        "c0_relative_attention_fail_mean_map_overlay.png",
        "c0_relative_attention_pass_mean_map_overlay.png",
        "c0_relative_attention_diff_map_fail_minus_pass_overlay.png",
    }
    assert all(p.exists() for p in paths)


def test_image_overlays_is_empty_and_does_not_raise_without_real_images():
    res = RelativeAttentionAnalyzer().run(AttnVLM(), _labeled_batch())
    assert res.image_overlays("/tmp", "c0_relative_attention") == []


def test_stats_layer_picks_up_attention_signals():
    from evalvitals.analysis.stats_tools import build_stats_input, describe_data

    batch = _labeled_batch()
    res = RelativeAttentionAnalyzer().run(AttnVLM(), batch)
    inp = build_stats_input({"relative_attention": res}, batch)
    assert "relative_attention.max_relative_weight" in inp.per_case
    # Continuous signal → eligible for median-split association and rank corr.
    assert "relative_attention.max_relative_weight" in describe_data(inp)["continuous_signals"]


# ── WS3: richer per-case scalars + the per-case map stack ────────────────────

_RICH_FEATURES = ("attention_entropy", "top1_share", "center_offset", "edge_mass")


def test_richer_per_case_features_present():
    res = RelativeAttentionAnalyzer().run(AttnVLM(), _labeled_batch())
    for e in res.findings["per_case"]:
        for k in _RICH_FEATURES:
            assert k in e, f"missing {k}"
            assert 0.0 <= e[k] <= 1.0
    # FOCUS (pass) cases spike on one patch → lower entropy than the flat fail
    # cases, and higher single-patch dominance.
    by_id = {e["id"]: e for e in res.findings["per_case"]}
    assert by_id["p0"]["attention_entropy"] < by_id["f0"]["attention_entropy"]
    assert by_id["p0"]["top1_share"] > by_id["f0"]["top1_share"]


def test_per_case_map_stack_artifact():
    import numpy as np

    res = RelativeAttentionAnalyzer().run(AttnVLM(), _labeled_batch())
    maps = res.artifacts["per_case_maps"]
    assert set(maps) == {"p0", "p1", "f0", "f1"}
    for cid, m in maps.items():
        assert m.dtype == np.float16
        assert m.shape == (2, 2)


def test_richer_features_are_continuous_signals_not_leaks():
    from evalvitals.analysis.stats_tools import build_stats_input

    batch = _labeled_batch()
    res = RelativeAttentionAnalyzer().run(AttnVLM(), batch)
    inp = build_stats_input({"relative_attention": res}, batch)
    for k in _RICH_FEATURES:
        key = f"relative_attention.{k}"
        # New features enter the tested family — none is misfiled as a label leak.
        assert key in inp.per_case, f"{key} missing from tested family"
        assert key not in inp.sanity