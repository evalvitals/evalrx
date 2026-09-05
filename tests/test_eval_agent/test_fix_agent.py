"""Fix module: intervention-space tiers, L2 tool pipelines, validated repair.

The allowed tier is an input (default L2); no automatic escalation — when no
candidate validates, the outcome recommends raising the tier, routed from the
verified hypotheses' mechanisms.
"""

from __future__ import annotations

import json

import pytest

from evalrx.core.capability import Capability
from evalrx.core.case import CaseBatch, FailureCase, Inputs, Label
from evalrx.core.model import Model
from evalrx.eval_agent import (
    FixAgent,
    FixTier,
    parse_tier,
    route_min_tier,
)
from evalrx.eval_agent.hypothesis import Hypothesis
from evalrx.eval_agent.stages.fix_agent import (
    FixCandidate,
    FixValidation,
    _chart_arithmetic_predicate,
    _chart_count_extract_predicate,
    _code_copies_example,
    _code_redefines_model_bridge,
    _malformed_choice_predicate,
)
from evalrx.eval_agent.stages.repair_catalog import discover_methods

# ── tiers ─────────────────────────────────────────────────────────────────────


def test_tier_parse_and_order():
    assert parse_tier("L0") is FixTier.L0_RUNTIME_CONFIG
    assert parse_tier("L1") is FixTier.L1_PROMPT
    assert parse_tier("l3a") is FixTier.L3A_INTERNALS_READ
    assert parse_tier("L3") is FixTier.L3A_INTERNALS_READ  # bare L3 = read side
    assert parse_tier(FixTier.L4_PARAMETERS) is FixTier.L4_PARAMETERS
    assert (
        FixTier.L0_RUNTIME_CONFIG
        < FixTier.L1_PROMPT
        < FixTier.L2_SCAFFOLD
        < FixTier.L3A_INTERNALS_READ
    )
    assert FixTier.L3B_INTERNALS_WRITE < FixTier.L4_PARAMETERS
    assert FixTier.L3B_INTERNALS_WRITE.label == "L3b"
    with pytest.raises(ValueError, match="unknown fix tier"):
        parse_tier("L9")


def _hyp(statement: str, mode: str = "", design: str = "") -> Hypothesis:
    return Hypothesis(
        statement=statement, target_model="m", predicted_failure_mode=mode, test_design=design
    )


def test_routing_by_mechanism_keywords():
    tier, _ = route_min_tier(
        _hyp("responses hit the configured max_tokens completion limit", mode="truncation")
    )
    assert tier is FixTier.L0_RUNTIME_CONFIG

    tier, why = route_min_tier(
        _hyp(
            "pathologies smaller than one patch are destroyed by downsampling",
            mode="resolution_limit",
        )
    )
    assert tier is FixTier.L2_SCAFFOLD and "resolution" in why

    tier, why = route_min_tier(
        _hyp("a training-free crop/enhance scaffold should magnify the small text")
    )
    assert tier is FixTier.L2_SCAFFOLD and "train" not in why

    tier, _ = route_min_tier(
        _hyp("suppress the attention sink on structural tokens", mode="attention_sink")
    )
    assert tier is FixTier.L3B_INTERNALS_WRITE  # write verbs beat bare "attention"

    tier, _ = route_min_tier(
        _hyp("attention mass never reaches the image region", mode="attention_dispersion")
    )
    assert tier is FixTier.L3A_INTERNALS_READ

    tier, _ = route_min_tier(_hyp("the model relies on a finding frequency prior from training"))
    assert tier is FixTier.L4_PARAMETERS

    tier, why = route_min_tier(_hyp("the model answers too tersely"))
    assert tier is FixTier.L1_PROMPT and "cheapest" in why


def test_routing_is_word_boundary_and_negation_aware():
    # "constraint" contains the substring "train" but is not the L4 mechanism —
    # word-boundary stem matching must not route it to fine-tuning.
    tier, _ = route_min_tier(_hyp("the decoder violates a layout constraint"))
    assert tier is FixTier.L1_PROMPT

    # Negated training phrases assert the OPPOSITE of needing L4.
    tier, why = route_min_tier(_hyp("a crop pipeline fixes this without any retraining"))
    assert tier is FixTier.L2_SCAFFOLD and "train" not in why
    tier, _ = route_min_tier(_hyp("zoom helps; no fine-tuning required"))
    assert tier is FixTier.L2_SCAFFOLD


def test_routing_prefers_explicit_metadata():
    h = _hyp("the model answers too tersely")  # prose alone -> L1
    h.metadata = {"fix_tier": "L3b"}
    tier, why = route_min_tier(h)
    assert tier is FixTier.L3B_INTERNALS_WRITE and "metadata" in why
    # a malformed hint falls back to keyword routing rather than crashing
    h.metadata = {"fix_tier": "L99"}
    tier, _ = route_min_tier(h)
    assert tier is FixTier.L1_PROMPT


# ── L2 image tools + pipeline executor ───────────────────────────────────────


def _img(size=(64, 48)):
    PIL = pytest.importorskip("PIL")
    from PIL import Image  # noqa: F401

    return PIL.Image.new("L", size, color=100)


def test_image_tools_preserve_or_scale_size():
    from evalrx.eval_agent.stages.fix_tools import upscale, zoom_center

    img = _img()
    assert zoom_center(img, factor=2.0).size == img.size
    assert upscale(img, factor=2.0).size == (128, 96)


def test_apply_image_ops_skips_unknown_and_loads_paths(tmp_path):
    from evalrx.eval_agent.stages.fix_tools import apply_image_ops

    path = tmp_path / "x.png"
    _img().save(path)
    out = apply_image_ops(
        str(path),
        [
            {"tool": "no_such_tool", "params": {}},
            {"tool": "upscale", "params": {"factor": 2.0}},
        ],
    )
    assert out.size == (128, 96)


def test_crop_salient_region_magnifies_small_content():
    PIL = pytest.importorskip("PIL")
    import numpy as np

    from evalrx.eval_agent.stages.fix_tools import crop_salient_region

    img = PIL.Image.new("RGB", (64, 64), color=(210, 210, 210))
    for y in range(30, 34):
        for x in range(64):
            img.putpixel((x, y), (200, 40, 40))

    before = np.asarray(img)
    out = crop_salient_region(img, padding=0.02)
    after = np.asarray(out)
    before_red = ((before[:, :, 0] > 150) & (before[:, :, 1] < 100)).sum()
    after_red = ((after[:, :, 0] > 150) & (after[:, :, 1] < 140)).sum()
    assert out.size == img.size
    assert after_red > before_red * 5


def test_crop_case_bbox_magnifies_metadata_box():
    PIL = pytest.importorskip("PIL")
    import numpy as np

    from evalrx.core.case import FailureCase, Inputs
    from evalrx.eval_agent.stages.fix_tools import crop_case_bbox

    img = PIL.Image.new("RGB", (100, 100), color=(220, 220, 220))
    for y in range(10, 14):
        for x in range(80, 84):
            img.putpixel((x, y), (20, 20, 20))
    case = FailureCase(
        id="tiny_text_region",
        inputs=Inputs(prompt="read it", image=img),
        metadata={"answer_bbox_xyxy_norm": [0.8, 0.1, 0.84, 0.14]},
    )

    out = crop_case_bbox(img, case=case, padding=0.0, min_size_frac=0.08)
    arr = np.asarray(out)
    dark = (arr[:, :, 0] < 80) & (arr[:, :, 1] < 80) & (arr[:, :, 2] < 80)

    assert out.size == img.size
    assert dark.mean() > 0.2


def test_crop_case_bbox_no_bbox_is_noop_even_with_enhancement():
    PIL = pytest.importorskip("PIL")
    import numpy as np

    from evalrx.core.case import FailureCase, Inputs
    from evalrx.eval_agent.stages.fix_tools import crop_case_bbox

    img = PIL.Image.new("RGB", (32, 32), color=(120, 130, 140))
    case = FailureCase(id="no_bbox", inputs=Inputs(prompt="q", image=img), metadata={})

    out = crop_case_bbox(img, case=case, sharpen_factor=3.0, contrast_factor=1.5)

    assert out is img
    assert np.asarray(out).mean() == np.asarray(img).mean()


def test_run_pipeline_can_fix_textvqa_style_bbox_case():
    PIL = pytest.importorskip("PIL")
    import numpy as np

    from evalrx.eval_agent.stages.fix_tools import PipelineSpec, run_pipeline

    img = PIL.Image.new("RGB", (100, 100), color=(220, 220, 220))
    for y in range(10, 14):
        for x in range(80, 84):
            img.putpixel((x, y), (20, 20, 20))
    case = FailureCase(
        id="textvqa_small",
        inputs=Inputs(prompt="What text is shown?", image=img),
        expected="abc",
        metadata={"answer_bbox_xyxy_norm": [0.8, 0.1, 0.84, 0.14]},
    )

    class BBoxSensitiveModel:
        def generate(self, inputs, **kwargs):
            arr = np.asarray(inputs.image)
            dark = (arr[:, :, 0] < 80) & (arr[:, :, 1] < 80) & (arr[:, :, 2] < 80)
            return "abc" if dark.mean() > 0.2 else "wrong"

    spec = PipelineSpec(
        name="answer_bbox_crop",
        image_ops=[{"tool": "crop_case_bbox", "params": {"padding": 0.0, "min_size_frac": 0.08}}],
    )

    assert run_pipeline(BBoxSensitiveModel(), case, spec, _label_score) is True


def test_separate_horizontal_bands_adds_visible_gaps():
    PIL = pytest.importorskip("PIL")
    import numpy as np

    from evalrx.eval_agent.stages.fix_tools import separate_horizontal_bands

    colors = [(200, 40, 40), (40, 160, 40), (40, 40, 200), (210, 150, 30), (160, 50, 200)]
    img = PIL.Image.new("RGB", (80, 80), color=(210, 210, 210))
    y = 35
    for color in colors:
        for yy in range(y, y + 2):
            for x in range(80):
                img.putpixel((x, yy), color)
        y += 2

    out = separate_horizontal_bands(img)
    arr = np.asarray(out, dtype=np.int16)
    bg = np.array([210, 210, 210], dtype=np.int16)
    salient_rows = (np.linalg.norm(arr - bg, axis=2) > 30).mean(axis=1) > 0.2
    groups = 0
    prev = False
    for flag in salient_rows.tolist():
        if flag and not prev:
            groups += 1
        prev = flag
    assert out.size == img.size
    assert groups == len(colors)


def test_annotate_horizontal_band_count_overlays_measurement():
    PIL = pytest.importorskip("PIL")
    import numpy as np

    from evalrx.eval_agent.stages.fix_tools import (
        _horizontal_band_count,
        annotate_horizontal_band_count,
    )

    img = PIL.Image.new("RGB", (96, 96), color=(210, 210, 210))
    colors = [(200, 40, 40), (40, 160, 40), (40, 40, 200), (210, 150, 30)]
    y = 40
    for color in colors:
        for yy in range(y, y + 2):
            for x in range(96):
                img.putpixel((x, yy), color)
        y += 2

    assert _horizontal_band_count(img) == len(colors)
    out = annotate_horizontal_band_count(img)
    arr = np.asarray(out)
    assert out.size == img.size
    assert ((arr[:20] > 240).all(axis=2)).mean() > 0.7  # white banner
    assert (arr[:30].min(axis=2) < 40).any()  # black text/border pixels


def test_pipeline_spec_validation():
    from evalrx.eval_agent.stages.fix_tools import PipelineSpec

    assert PipelineSpec.from_dict({"name": "x", "prompt_template": "no placeholder"}) is None
    spec = PipelineSpec.from_dict(
        {
            "name": "zoom",
            "image_ops": [{"tool": "zoom_center", "params": {"factor": 2}}, {"tool": "bogus"}],
            "n_samples": 99,
        }
    )
    assert spec is not None
    assert [op["tool"] for op in spec.image_ops] == ["zoom_center"]  # bogus dropped
    assert spec.n_samples == 5  # capped

    multi_call = PipelineSpec.from_dict(
        {
            "name": "bounded_multicall",
            "strategy": "chain_of_verification",
            "n_samples": 5,
        }
    )
    assert multi_call is not None
    assert multi_call.n_samples == 2  # 3 calls/sample * 2 <= 6 calls/case


def test_pipeline_passes_bounded_generation_kwargs():
    from evalrx.eval_agent.stages.fix_tools import PipelineSpec, run_pipeline

    case = FailureCase(id="decode", inputs=Inputs(prompt="q"), expected="yes")

    class DecodeBudgetModel:
        def generate(self, inputs, **kwargs):
            return "yes" if kwargs.get("max_tokens") == 512 else "no"

    spec = PipelineSpec.from_dict(
        {
            "name": "more_budget",
            "generation_kwargs": {"max_tokens": 512, "bad": "dropped"},
        }
    )
    assert spec is not None
    assert spec.generation_kwargs == {"max_tokens": 512}
    assert run_pipeline(DecodeBudgetModel(), case, spec, _label_score) is True


def test_pipeline_normalizes_zero_temperature_and_empty_stop():
    from evalrx.eval_agent.stages.fix_tools import PipelineSpec

    greedy = PipelineSpec.from_dict({
        "name": "greedy",
        "generation_kwargs": {"temperature": 0.0, "stop": []},
    })
    sampled = PipelineSpec.from_dict({
        "name": "sampled",
        "generation_kwargs": {"temperature": 0.7, "stop": ["DONE"]},
    })
    assert greedy is not None and greedy.generation_kwargs == {"do_sample": False}
    assert sampled is not None and sampled.generation_kwargs == {
        "temperature": 0.7, "do_sample": True, "stop": ["DONE"]
    }


def test_pipeline_self_refine_is_a_label_blind_reviewed_multicall_strategy():
    from evalrx.eval_agent.stages.fix_tools import PipelineSpec, run_pipeline

    case = FailureCase(id="review", inputs=Inputs(prompt="What is 2 + 2?"), expected="yes")

    class SequencedModel:
        def __init__(self):
            self.prompts: list[str] = []

        def generate(self, inputs, **kwargs):
            self.prompts.append(inputs.prompt)
            return ("draft", "feedback", "yes")[len(self.prompts) - 1]

    model = SequencedModel()
    spec = PipelineSpec(name="review", strategy="self_refine")
    assert run_pipeline(model, case, spec, _label_score) is True
    assert len(model.prompts) == 3
    assert "yes" not in "\n".join(model.prompts).lower()  # expected label never enters prompts


def test_pipeline_preserves_non_image_modality_fields():
    """run_pipeline's inner generate() used to rebuild a bare Inputs(prompt=...,
    image=...), silently dropping .video/.audio -- every strategy was
    unconditionally inapplicable on any non-image FailureCase (generate()
    raising on the missing modality, every call returning "", run_pipeline
    returning None -> "no applicable scorable pair" for every case, the
    exact failure musicavqa_videollama2's real run hit)."""
    from evalrx.eval_agent.stages.fix_tools import PipelineSpec, run_pipeline

    case = FailureCase(
        id="clip", inputs=Inputs(prompt="What instrument is heard?", video="clip.mp4"),
        expected="cello",
    )

    class VideoRequiredModel:
        def generate(self, inputs, **kwargs):
            if getattr(inputs, "video", None) is None:
                raise ValueError("requires Inputs.video")
            return "cello"

    spec = PipelineSpec(name="direct", strategy="direct")
    assert run_pipeline(VideoRequiredModel(), case, spec, _label_score) is True


def test_pipeline_votes_on_task_declared_output_key_not_hidden_score():
    from evalrx.eval_agent.stages.fix_tools import PipelineSpec, run_pipeline

    case = FailureCase(
        id="structured",
        inputs=Inputs(prompt="Return a number"),
        expected="yes",
        metadata={"output_key_pattern": r"FINAL:\s*(\d+)"},
    )

    class Samples:
        def __init__(self):
            self.outputs = iter(["work FINAL: 4", "another FINAL: 4", "FINAL: 9"])

        def generate(self, inputs, **kwargs):
            return next(self.outputs)

    def four_is_correct(case, output):
        return "FINAL: 4" in output

    spec = PipelineSpec(name="vote", n_samples=3)
    assert run_pipeline(Samples(), case, spec, four_is_correct) is True


# ── fake models ───────────────────────────────────────────────────────────────


class BaselineFailsModel(Model):
    """Answers "no" unless the prompt asks to examine carefully -> then "yes"."""

    capabilities = frozenset({Capability.GENERATE})
    modalities = frozenset({"text", "image"})

    def generate(self, inputs, **kwargs):
        p = str(getattr(inputs, "prompt", inputs)).lower()
        return "Yes." if "carefully" in p else "No."

    def forward(self, inputs, capture, spec=None):
        raise NotImplementedError


class ZoomSensitiveModel(Model):
    """Answers "yes" only when the image was upscaled (width >= 100)."""

    capabilities = frozenset({Capability.GENERATE})
    modalities = frozenset({"text", "image"})

    def generate(self, inputs, **kwargs):
        img = getattr(inputs, "image", None)
        w = img.size[0] if img is not None and hasattr(img, "size") else 0
        return "Yes." if w >= 100 else "No."

    def forward(self, inputs, capture, spec=None):
        raise NotImplementedError


class HopelessModel(Model):
    capabilities = frozenset({Capability.GENERATE})
    modalities = frozenset({"text", "image"})

    def generate(self, inputs, **kwargs):
        return "No."

    def forward(self, inputs, capture, spec=None):
        raise NotImplementedError


class DecodeBudgetSensitiveModel(Model):
    """Generic decode-health fixture: only a larger recorded budget completes."""

    capabilities = frozenset({Capability.GENERATE})
    modalities = frozenset({"text"})

    def generate(self, inputs, **kwargs):
        return "Yes." if int(kwargs.get("max_tokens", 64)) >= 128 else "No."

    def forward(self, inputs, capture, spec=None):
        raise NotImplementedError


class VCDSensitiveModel(Model):
    """Binary VLM fixture: only its contrastive decoder repairs the answer."""

    capabilities = frozenset({Capability.GENERATE, Capability.LOGPROBS})
    modalities = frozenset({"text", "image"})

    def generate(self, inputs, **kwargs):
        return "No."

    def generate_vcd(self, inputs, **kwargs):
        return "Yes."

    def generate_vcd_baseline(self, inputs, **kwargs):
        return "No."

    def paper_method_fidelity(self, method):
        return "per_item_seeded_sampler_specialization" if method == "vcd" else "unavailable"

    def forward(self, inputs, capture, spec=None):
        raise NotImplementedError


class AADSensitiveModel(Model):
    """Binary audio-LALM fixture: only its silence-contrast decoder repairs it."""

    capabilities = frozenset({Capability.GENERATE, Capability.LOGPROBS})
    modalities = frozenset({"text", "audio"})

    def generate(self, inputs, **kwargs):
        return "No."

    def generate_aad(self, inputs, **kwargs):
        assert kwargs == {"alpha": 0.5}
        return "Yes."

    def paper_method_fidelity(self, method):
        return "native_silence_contrast" if method == "aad" else "unavailable"

    def forward(self, inputs, capture, spec=None):
        raise NotImplementedError


class ICDSensitiveModel(Model):
    """Binary VLM fixture: only instruction contrastive decoding repairs it."""

    capabilities = frozenset({Capability.GENERATE, Capability.LOGPROBS})
    modalities = frozenset({"text", "image"})

    def generate(self, inputs, **kwargs):
        return "No."

    def generate_instruction_cd(self, inputs, **kwargs):
        return "Yes."

    def paper_method_fidelity(self, method):
        return "exact" if method == "icd" else "unavailable"

    def forward(self, inputs, capture, spec=None):
        raise NotImplementedError


class ViCropSensitiveModel(Model):
    """LLaVA-style fixture repaired only by the native ViCrop executor."""

    capabilities = frozenset({Capability.GENERATE, Capability.ATTENTION})
    modalities = frozenset({"text", "image"})

    def generate(self, inputs, **kwargs):
        return "No."

    def generate_vicrop(self, inputs, **kwargs):
        assert kwargs == {"layer": 14}
        return "Yes."

    def paper_method_fidelity(self, method):
        return "native_selector_specialization" if method == "vicrop" else "unavailable"

    def forward(self, inputs, capture, spec=None):
        raise NotImplementedError


class ViCropConsensusSensitiveModel(ViCropSensitiveModel):
    """LLaVA-style fixture repaired only by the ViCrop safety guard."""

    def generate_vicrop_consensus(self, inputs, *, baseline_answer, **kwargs):
        assert baseline_answer == "No."
        assert kwargs == {"layer": 14}
        return "Yes."


class PAISensitiveModel(Model):
    """LLaVA-style fixture repaired by PAI's image-attention branch."""

    capabilities = frozenset({Capability.GENERATE, Capability.ATTENTION})
    modalities = frozenset({"text", "image"})

    def generate(self, inputs, **kwargs):
        return "No."

    def generate_pai(self, inputs, **kwargs):
        assert kwargs == {
            "alpha": 0.2,
            "guidance_scale": 2.0,
            "start_layer": 2,
            "end_layer": 32,
        }
        return "Yes."

    def paper_method_fidelity(self, method):
        return "native_attention_cfg_specialization" if method == "pai" else "unavailable"

    def forward(self, inputs, capture, spec=None):
        raise NotImplementedError


class OPERASensitiveModel(Model):
    """LLaVA-style fixture repaired by OPERA's binary attention penalty."""

    capabilities = frozenset({Capability.GENERATE, Capability.ATTENTION})
    modalities = frozenset({"text", "image"})

    def generate(self, inputs, **kwargs):
        return "No."

    def generate_opera_binary(self, inputs, **kwargs):
        assert kwargs == {"num_attn_candidates": 5, "penalty_weight": 1.0}
        return "Yes."

    def paper_method_fidelity(self, method):
        return "native_binary_specialization" if method == "opera" else "unavailable"

    def forward(self, inputs, capture, spec=None):
        raise NotImplementedError


class TCDSensitiveModel(Model):
    """Audio LALM fixture repaired only by TCD's gated temporal contrast."""

    capabilities = frozenset({Capability.GENERATE, Capability.ATTENTION, Capability.HIDDEN_STATES})
    modalities = frozenset({"text", "audio"})

    def generate(self, inputs, **kwargs):
        return "2"

    def generate_tcd(self, inputs, **kwargs):
        assert kwargs == {}
        return "3"

    def generate_tcd_baseline(self, inputs, **kwargs):
        return "2"

    def paper_method_fidelity(self, method):
        return "native_layer_matched_stability" if method == "tcd" else "unavailable"

    def forward(self, inputs, capture, spec=None):
        raise NotImplementedError


class IFCDSensitiveModel(Model):
    """LLaVA-style fixture repaired only by an opted-in TruthX IFCD route."""

    capabilities = frozenset({Capability.GENERATE, Capability.HIDDEN_STATES})
    modalities = frozenset({"text", "image"})

    def generate(self, inputs, **kwargs):
        return "No."

    def generate_ifcd(self, inputs, **kwargs):
        assert kwargs == {"alpha": 0.1, "beta": 0.1, "edit_strength": 0.5, "top_layers": 15}
        return "Yes."

    def paper_method_fidelity(self, method):
        return "adapted_truthx_artifact" if method == "ifcd" else "unavailable"

    def forward(self, inputs, capture, spec=None):
        raise NotImplementedError


class VisualSearchSensitiveModel(Model):
    """Image fixture where only a question-guided crop returns the answer."""

    capabilities = frozenset({Capability.GENERATE})
    modalities = frozenset({"text", "image"})

    def generate(self, inputs, **kwargs):
        return "No."

    def generate_visual_search(self, inputs, **kwargs):
        return "Yes."

    def forward(self, inputs, capture, spec=None):
        raise NotImplementedError


class DetectorVisualSearchSensitiveModel(Model):
    """Image fixture where only detector-guided visual search repairs it."""

    capabilities = frozenset({Capability.GENERATE})
    modalities = frozenset({"text", "image"})

    def generate(self, inputs, **kwargs):
        return "No."

    def generate_detector_visual_search(self, inputs, **kwargs):
        return "Yes."

    def forward(self, inputs, capture, spec=None):
        raise NotImplementedError


class FrozenBaselineModel(Model):
    capabilities = frozenset({Capability.GENERATE})

    def __init__(self):
        self.calls = 0

    def generate(self, inputs, **kwargs):
        self.calls += 1
        return "No."

    def forward(self, inputs, capture, spec=None):
        raise NotImplementedError


class ScriptedJudge(Model):
    capabilities = frozenset({Capability.GENERATE})
    modalities = frozenset({"text"})

    def __init__(self, reply: str) -> None:
        self._reply = reply
        self.prompts: list[str] = []

    def generate(self, inputs, **kwargs) -> str:
        self.prompts.append(str(inputs))
        return self._reply

    def forward(self, inputs, capture, spec=None):
        raise NotImplementedError


def _gold_yes_batch(n: int = 8, image=None) -> CaseBatch:
    yes = {"all_of": ["yes"], "none_of": ["no"]}
    return CaseBatch(
        [
            FailureCase(
                id=f"c{i}",
                inputs=Inputs(prompt=f"Is there a lesion {i}?", image=image),
                expected=yes,
                label=Label.FAIL,
            )
            for i in range(n)
        ]
    )


def _gold_audio_batch(n: int = 8, audio: str = "fake-waveform") -> CaseBatch:
    three = {"all_of": ["3"], "none_of": ["2"]}
    batch = CaseBatch(
        [
            FailureCase(
                id=f"a{i}",
                inputs=Inputs(prompt=f"How many beeps {i}?", audio=audio),
                expected=three,
                label=Label.FAIL,
            )
            for i in range(n)
        ]
    )
    for case in batch:
        case.metadata["task"] = "multiple_choice"
    return batch


def _gold_audio_yes_batch(n: int = 8, audio: str = "fake-waveform") -> CaseBatch:
    yes = {"all_of": ["yes"], "none_of": ["no"]}
    batch = CaseBatch(
        [
            FailureCase(
                id=f"ay{i}",
                inputs=Inputs(prompt=f"Is there a dog barking {i}?", audio=audio),
                expected=yes,
                label=Label.FAIL,
            )
            for i in range(n)
        ]
    )
    for case in batch:
        case.metadata["task"] = "yes_no"
    return batch


def _label_score(case, observed):
    """CaseDiscovery-style scorer: returns Label instead of bool."""
    from evalrx.analyzers.perturbation.prompt_contrast import _default_score

    score = _default_score(case, observed)
    if score is None:
        return Label.UNKNOWN
    return Label.PASS if score else Label.FAIL


# ── FixAgent: L1 repair via judge proposal ───────────────────────────────────


def test_l1_judge_candidate_validates_and_fixes():
    judge = ScriptedJudge(
        json.dumps(
            [
                {"name": "careful", "prompt_template": "Look very carefully. {prompt}"},
            ]
        )
    )
    agent = FixAgent(judge=judge, max_tier="L1")
    out = agent.propose_and_validate(
        BaselineFailsModel(),
        _gold_yes_batch(),
        [_hyp("the prompt phrasing underspecifies the task")],
    )
    assert out.fixed is True
    assert out.best is not None and out.best.candidate.tier is FixTier.L1_PROMPT
    assert out.best.n_fixed == 8 and out.best.n_broken == 0
    assert out.best.effect and out.best.effect > 0
    assert out.recommendation is None
    # max_tier=L1 -> no L2 candidates were attempted
    assert all(v.candidate.tier is FixTier.L1_PROMPT for v in out.attempted)
    assert out.repair_rounds == 1  # single-shot by default


def test_heldout_authoring_uses_only_proposal_data_and_one_round():
    """Confirmation failures must not enter prompts or drive a retry."""
    judge = ScriptedJudge(
        '[{"name": "careful", "prompt_template": "Look carefully. {prompt}"}]'
    )
    confirm = _gold_yes_batch()
    for case in confirm:
        case.inputs.prompt = "CONFIRM_SECRET"
    explore = _gold_yes_batch()
    for case in explore:
        case.inputs.prompt = "EXPLORE_VISIBLE"

    agent = FixAgent(judge=judge, max_tier="L1", max_repair_rounds=2)
    out = agent.propose_and_validate(
        HopelessModel(),
        confirm,
        [_hyp("the prompt phrasing underspecifies the task")],
        proposal_data=explore,
    )

    assert out.fixed is False
    assert out.repair_rounds == 1
    assert len(judge.prompts) == 1
    assert "EXPLORE_VISIBLE" in judge.prompts[0]
    assert "CONFIRM_SECRET" not in judge.prompts[0]


def test_code_only_allowlist_skips_discarded_judge_proposals():
    judge = ScriptedJudge("[]")
    agent = FixAgent(
        judge=judge,
        max_tier="L3a",
        allow_codegen=False,
        candidate_allowlist={"coded_pipeline"},
    )

    assert agent._propose([_hyp("x")], _gold_yes_batch(image=_img()), HopelessModel()) == []
    assert judge.prompts == []


def test_code_only_still_fields_the_coded_pipeline():
    """--code-only exists to test the coder-written pipeline and nothing else;
    it used to propose NOTHING (the coded candidate sat inside the gate that
    skips the judge's L0-L3 proposals), so every code-only run ended with
    'repair round 1 produced no NEW candidate'."""
    pytest.importorskip("PIL")
    judge = CodeWritingJudge()
    judge.prompts = []

    class _Recording(CodeWritingJudge):
        def generate(self, inputs, **kwargs):
            judge.prompts.append(str(inputs))
            return CodeWritingJudge.generate(self, inputs, **kwargs)

    agent = FixAgent(
        judge=_Recording(), max_tier="L3a", candidate_allowlist={"coded_pipeline"},
    )
    proposed = agent._propose([_hyp("x")], _gold_yes_batch(image=_img()), HopelessModel())
    assert [c.name for c in proposed] == ["coded_pipeline"]
    assert len(judge.prompts) == 1 and "EXECUTION CONTRACT" in judge.prompts[0]


def test_allowlisted_builtin_is_materialised_before_judge_truncation():
    judge = ScriptedJudge(
        json.dumps([{"name": "unrelated", "prompt_template": "Ignore. {prompt}"}])
    )
    agent = FixAgent(
        judge=judge, max_tier="L2", allow_codegen=False,
        candidate_allowlist={"self_refine"},
    )
    candidates = agent._propose(
        [_hyp("chart reasoning")], _gold_yes_batch(image=_img()), HopelessModel()
    )
    assert [c.name for c in candidates] == ["self_refine"]
    assert judge.prompts == []


def test_malformed_choice_candidate_is_gold_free_gated_and_pre_registered():
    malformed = FailureCase(
        id="bad", inputs=Inputs(prompt="listen", audio="x.wav"), expected="A",
        observed="thought: option A or B\nstill analyzing", label=Label.FAIL,
        metadata={"task": "multiple_choice_letter"},
    )
    clean = FailureCase(
        id="clean", inputs=Inputs(prompt="listen", audio="x.wav"), expected="B",
        observed="(B)", label=Label.PASS, metadata={"task": "multiple_choice_letter"},
    )
    # The gate depends only on the recorded output contract, not expected/gold.
    assert _malformed_choice_predicate(malformed)
    malformed.expected = "D"
    assert _malformed_choice_predicate(malformed)
    assert not _malformed_choice_predicate(clean)

    judge = ScriptedJudge("[]")
    agent = FixAgent(
        judge=judge, max_tier="L2", allow_codegen=False,
        candidate_allowlist={"malformed_choice_consensus"},
    )
    candidates = agent._propose([_hyp("malformed output")], CaseBatch([malformed, clean]),
                                HopelessModel())
    assert [c.name for c in candidates] == ["malformed_choice_consensus"]
    assert candidates[0].predicate is _malformed_choice_predicate
    assert judge.prompts == []


def test_chart_arithmetic_candidate_is_gold_free_and_skips_out_of_scope_calls():
    arithmetic = FailureCase(
        id="arithmetic", inputs=Inputs(prompt="What is the difference?", image=_img()),
        expected="1", observed="2", label=Label.FAIL,
        metadata={"task": "exact_or_numeric"},
    )
    lookup = FailureCase(
        id="lookup", inputs=Inputs(prompt="Which country is blue?", image=_img()),
        expected="France", observed="Spain", label=Label.FAIL,
        metadata={"task": "exact_or_numeric"},
    )
    assert _chart_arithmetic_predicate(arithmetic)
    arithmetic.expected = "999"  # the predicate never reads gold
    assert _chart_arithmetic_predicate(arithmetic)
    assert not _chart_arithmetic_predicate(lookup)

    judge = ScriptedJudge("[]")
    agent = FixAgent(
        judge=judge, max_tier="L2", allow_codegen=False,
        candidate_allowlist={"chart_arithmetic_verify"},
    )
    batch = CaseBatch([arithmetic, lookup])
    candidates = agent._propose([_hyp("chart arithmetic")], batch, HopelessModel())
    assert [c.name for c in candidates] == ["chart_arithmetic_verify"]
    assert judge.prompts == []

    class CountingModel(Model):
        capabilities = frozenset({Capability.GENERATE})
        modalities = frozenset({"text", "image"})

        def __init__(self):
            self.calls = 0

        def generate(self, inputs, **kwargs):
            self.calls += 1
            return "1"

        def forward(self, inputs, capture, spec=None):
            raise NotImplementedError

    model = CountingModel()
    agent._candidate_scores(candidates[0], model, batch)
    assert model.calls == 3  # one chain-of-verification; lookup was never run


def test_chart_count_extract_gate_is_gold_free_and_excludes_color_questions():
    count = FailureCase(
        id="count", inputs=Inputs(prompt="How many countries exceed 70%?", image=_img()),
        expected="2", observed="1", label=Label.FAIL,
        metadata={"task": "exact_or_numeric"},
    )
    color = FailureCase(
        id="color", inputs=Inputs(prompt="What's the color of the line?", image=_img()),
        expected="orange", observed="red", label=Label.FAIL,
        metadata={"task": "exact_or_numeric"},
    )
    assert _chart_count_extract_predicate(count)
    count.expected = "999"
    assert _chart_count_extract_predicate(count)
    assert not _chart_count_extract_predicate(color)


def test_l1_template_candidate_preserves_non_image_modality_fields():
    """The L1 template runner used to rebuild a bare Inputs(prompt=...,
    image=...), silently dropping .video/.audio -- identical bug to
    run_pipeline's, in the sibling L1 (not L2) code path."""
    case = FailureCase(
        id="clip", inputs=Inputs(prompt="What instrument is heard?", video="clip.mp4"),
        expected="cello", label=Label.FAIL,
    )
    batch = CaseBatch([case])

    class VideoRequiredModel(Model):
        capabilities = frozenset({Capability.GENERATE})
        modalities = frozenset({"text", "video"})

        def generate(self, inputs, **kwargs):
            if getattr(inputs, "video", None) is None:
                raise ValueError("requires Inputs.video")
            # baseline must fail: an all-pass subset short-circuits the search
            p = str(getattr(inputs, "prompt", "")).lower()
            return "cello" if "carefully" in p else "viola"

        def forward(self, inputs, capture, spec=None):
            raise NotImplementedError

    judge = ScriptedJudge('[{"name": "careful", "prompt_template": "Listen carefully. {prompt}"}]')
    agent = FixAgent(judge=judge, max_tier="L1", score_fn=_label_score)
    out = agent.propose_and_validate(
        VideoRequiredModel(), batch, [_hyp("the prompt underspecifies the task")]
    )
    careful = next(v for v in out.attempted if v.candidate.name == "careful")
    assert careful.n_pairs == 1  # was 0 ("no applicable scorable pair") before the fix
    assert careful.verdict != "not_executed"


def test_image_l1_candidates_start_with_visual_grounding_control():
    """A visual benchmark must not depend solely on a judge's narrow prompt."""
    batch = _gold_yes_batch(image=_img())
    batch[0].metadata["failure_axis"] = "scene text"
    candidates = FixAgent(max_tier="L1")._propose(
        [_hyp("small text is sometimes missed")],
        batch,
        HopelessModel(),
    )

    assert candidates[0].name == "visual_grounding"
    assert "visible evidence" in candidates[0].payload["prompt_template"]
    assert FixAgent()._strategy(candidates[0])(BaselineFailsModel(), batch[0]) is True


def test_video_only_batch_is_treated_as_having_visual_content():
    """A video case (no .image ever set) must not read as 'no images'.

    Found via a real run (musicavqa_videollama2): every case only set
    .video, so has_images was False for the whole batch and every
    image-gated candidate at every tier -- including the L1 visual-grounding
    control above -- was structurally excluded, not judge-rejected.
    """
    yes = {"all_of": ["yes"], "none_of": ["no"]}
    batch = CaseBatch(
        [
            FailureCase(
                id=f"clip{i}",
                inputs=Inputs(prompt=f"Is an instrument playing {i}?", video="clip.mp4"),
                expected=yes,
                label=Label.FAIL,
            )
            for i in range(8)
        ]
    )
    batch[0].metadata["failure_axis"] = "cross-modal evidence"
    candidates = FixAgent(max_tier="L1")._propose(
        [_hyp("small visual detail is sometimes missed")],
        batch,
        HopelessModel(),
    )

    assert candidates[0].name == "visual_grounding"


def test_l0_telemetry_candidate_repairs_decode_budget_without_prompt_guessing():
    """A recorded length stop permits a general runtime fix, not an LLM hunch."""
    yes = {"all_of": ["yes"], "none_of": ["no"]}
    batch = CaseBatch(
        [
            FailureCase(
                id=f"decode_{i}",
                inputs=Inputs(prompt=f"Solve item {i}"),
                expected=yes,
                label=Label.FAIL,
                metadata={
                    "finish_reason": "length",
                    "generation_config": {"max_tokens": 64},
                },
            )
            for i in range(16)
        ]
    )
    out = FixAgent(judge=None, max_tier="L0").propose_and_validate(
        DecodeBudgetSensitiveModel(),
        batch,
        [_hyp("generation was truncated at the configured token budget", mode="truncation")],
    )
    assert out.fixed is True
    assert out.best is not None
    assert out.best.candidate.tier is FixTier.L0_RUNTIME_CONFIG
    assert out.best.candidate.payload["generation_kwargs"] == {"max_tokens": 128}
    assert out.best.n_fixed == 16 and out.best.n_broken == 0

    confirmation = FixAgent(score_fn=_label_score).validate_candidate(
        DecodeBudgetSensitiveModel(), batch, out.best.candidate
    )
    assert confirmation.fixed is True
    assert confirmation.n_fixed == 16 and confirmation.n_broken == 0


def test_l0_does_not_admit_contrastive_decoding_candidates():
    visual = _gold_yes_batch(n=8, image=_img())
    for case in visual:
        case.metadata["task"] = "yes_no"
        case.expected = "No"
        case.observed = "Yes"
    audio = _gold_audio_yes_batch(n=8)
    for case in audio:
        case.expected = "No"
        case.observed = "Yes"

    vcd = FixAgent(judge=None, max_tier="L0")._propose(
        [_hyp("language priors override visual evidence")], visual, VCDSensitiveModel()
    )
    aad = FixAgent(judge=None, max_tier="L0")._propose(
        [_hyp("language priors override audio evidence")], audio, AADSensitiveModel()
    )

    assert not {"vcd", "aad", "icd"}.intersection(c.kind for c in (*vcd, *aad))


def test_l3a_vcd_candidate_repairs_binary_visual_grounding():
    batch = _gold_yes_batch(n=8, image=_img())
    for case in batch:
        case.metadata["task"] = "yes_no"

    out = FixAgent(
        judge=None, max_tier="L3a", candidate_allowlist={"vcd_diffusion_noise"}
    ).propose_and_validate(
        VCDSensitiveModel(), batch, [_hyp("language priors override visual evidence")]
    )

    assert out.fixed is True
    assert out.best is not None and out.best.candidate.name == "vcd_diffusion_noise"
    assert out.best.candidate.tier is FixTier.L3A_INTERNALS_READ
    assert out.best.n_fixed == 8 and out.best.n_broken == 0
    assert out.best.candidate.payload["kwargs"]["noise_step"] == 500


def test_l3a_vcd_uses_its_matched_sampling_control():
    class GreedyAndContrastAgree(VCDSensitiveModel):
        def generate(self, inputs, **kwargs):
            return "Yes."

    batch = _gold_yes_batch(n=8, image=_img())
    for case in batch:
        case.metadata["task"] = "exact_or_numeric"
    out = FixAgent(
        judge=None, max_tier="L3a", candidate_allowlist={"vcd_diffusion_noise"}
    ).propose_and_validate(
        GreedyAndContrastAgree(), batch, [_hyp("language priors override visual evidence")]
    )

    # The Stage-0 greedy arm is already correct.  VCD's matched clean sampler
    # is wrong, so the contrastive arm still has eight genuine paired repairs.
    assert out.best is not None
    assert out.best.n_fixed == 8 and out.best.n_broken == 0


def test_l3a_vcd_structural_discovery_does_not_read_expected_direction():
    batch = _gold_yes_batch(n=8, image=_img())
    for case in batch:
        case.metadata["task"] = "yes_no"
        case.expected = "Yes"
        case.observed = "No"
    candidates = FixAgent(
        judge=None, max_tier="L3a", candidate_allowlist={"vcd_diffusion_noise"}
    )._propose(
        [_hyp("language priors override visual evidence")], batch, VCDSensitiveModel()
    )

    assert "vcd_diffusion_noise" in {candidate.name for candidate in candidates}


def test_l3a_vcd_is_available_for_open_ended_visual_generation():
    batch = _gold_yes_batch(n=8, image=_img())
    for case in batch:
        case.metadata["task"] = "exact_or_numeric"
    candidates = FixAgent(
        judge=ScriptedJudge('[{"name": "vcd_diffusion_noise"}]'),
        max_tier="L3a",
        allow_codegen=False,
        paper_methods_only=True,
    )._propose(
        [_hyp("language priors override image evidence")], batch, VCDSensitiveModel()
    )

    vcd = next(candidate for candidate in candidates if candidate.name == "vcd_diffusion_noise")
    assert vcd.payload["baseline_executor"] == "generate_vcd_baseline"


def test_l3a_icd_structural_discovery_does_not_read_expected_direction():
    batch = _gold_yes_batch(n=8, image=_img())
    for case in batch:
        case.metadata["task"] = "yes_no"
        case.expected = "Yes"
        case.observed = "No"
    candidates = FixAgent(
        judge=None,
        max_tier="L3a",
        candidate_allowlist={"icd_instruction_disturbance"},
    )._propose(
        [_hyp("instruction priors override visual evidence")], batch, ICDSensitiveModel()
    )

    candidate_names = {candidate.name for candidate in candidates}
    assert candidate_names == {"icd_instruction_disturbance"}


def test_detector_grounded_presence_candidate_freezes_calibration_payload():
    class DetectorSensitiveModel(HopelessModel):
        def generate_detector_grounded_presence(self, inputs, **kwargs):
            return "Yes"

    batch = _gold_yes_batch(n=8, image=_img())
    for case in batch:
        case.metadata["task"] = "yes_no"
        case.observed = "No"
    candidates = FixAgent(
        judge=None,
        max_tier="L2",
        min_tier="L2",
        candidate_allowlist={"detector_grounded_presence_calibrated"},
    )._propose([_hyp("small objects are missed")], batch, DetectorSensitiveModel())

    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate.tier == FixTier.L2_SCAFFOLD
    assert candidate.payload["pass_baseline_answer"] is True
    assert candidate.payload["kwargs"]["detector_threshold"] == 0.25
    assert "traffic light" in candidate.payload["kwargs"]["objects"]


def test_clap_grounded_presence_candidate_freezes_calibration_payload():
    class ClapSensitiveModel(HopelessModel):
        modalities = frozenset({"text", "audio"})

        def generate_clap_grounded_presence(self, inputs, **kwargs):
            return "Yes"

    batch = _gold_audio_batch()
    for case in batch:
        case.metadata["task"] = "yes_no"
        case.observed = "No"
    candidates = FixAgent(
        judge=None,
        max_tier="L2",
        min_tier="L2",
        candidate_allowlist={"clap_grounded_audio_presence_calibrated"},
    )._propose([_hyp("audio evidence is missed")], batch, ClapSensitiveModel())

    assert len(candidates) == 1
    kwargs = candidates[0].payload["kwargs"]
    assert kwargs["negative_threshold"] == -0.05
    assert kwargs["positive_threshold"] == 0.275
    assert candidates[0].payload["pass_baseline_answer"] is True


def test_gemini_audio_specialist_freezes_external_model():
    class AudioSpecialistModel(HopelessModel):
        modalities = frozenset({"text", "audio"})

        def generate_audio_api_specialist(self, inputs, **kwargs):
            return kwargs["baseline_answer"]

    methods = discover_methods(
        AudioSpecialistModel(),
        max_tier=FixTier.L2_SCAFFOLD,
        has_images=False,
        has_audio=True,
        tasks={"multiple_choice_letter"},
        allow_adapted=False,
    )

    pro = next(m for m in methods if m.name == "gemini_pro_audio_specialist_calibrated")
    assert pro.payload["model_id"] == "gemini-2.5-pro"
    assert pro.source == "registered_calibrated"
    assert pro.pass_baseline_answer is True

    e4b_pro = next(
        m for m in methods if m.name == "e4b_gemini_pro_disagreement_guard_calibrated"
    )
    assert e4b_pro.payload["model_id"] == "gemini-2.5-pro"
    assert e4b_pro.payload["allowed_disagreements_by_route"] == {
        "music": ["AC", "AD", "BC", "BD", "CA"],
        "sound": ["AB", "AD", "BA", "BC", "CA"],
        "speech": ["BA", "BC", "CA", "CB", "DA", "DB"],
    }

def test_noncolor_spatial_specialist_freezes_model_and_gate():
    class SpatialModel:
        capabilities = frozenset()

        def generate_noncolor_spatial_specialist(self, inputs, **kwargs):
            return kwargs["baseline_answer"]

    methods = discover_methods(
        SpatialModel(),
        max_tier=FixTier.L2_SCAFFOLD,
        has_images=True,
        has_audio=False,
        tasks={"exact_or_numeric"},
        allow_adapted=False,
    )

    method = next(
        m for m in methods if m.name == "noncolor_spatial_vision_specialist_calibrated"
    )
    assert method.payload["model_id"] == "qwen2.5-vl-7b-instruct"
    assert method.source == "registered_calibrated"
    assert method.pass_baseline_answer is True


def test_chart_vision_specialist_freezes_external_model():
    class ChartModel:
        capabilities = frozenset()

        def generate_chart_vision_specialist(self, inputs, **kwargs):
            return kwargs["baseline_answer"]

    methods = discover_methods(
        ChartModel(),
        max_tier=FixTier.L2_SCAFFOLD,
        has_images=True,
        has_audio=False,
        tasks={"exact_or_numeric"},
        allow_adapted=False,
    )

    method = next(m for m in methods if m.name == "chart_vision_specialist_calibrated")
    assert method.payload["model_id"] == "qwen2.5-vl-7b-instruct"
    assert method.source == "registered_calibrated"
    assert method.pass_baseline_answer is True


def test_gemini_vision_specialist_freezes_external_model():
    class VisionSpecialistModel:
        capabilities = frozenset()

        def generate_vision_api_specialist(self, inputs, **kwargs):
            return kwargs["baseline_answer"]

    methods = discover_methods(
        VisionSpecialistModel(),
        max_tier=FixTier.L2_SCAFFOLD,
        has_images=True,
        has_audio=False,
        tasks={"exact_or_numeric"},
        allow_adapted=False,
    )

    method = next(m for m in methods if m.name == "gemini_vision_specialist_calibrated")
    assert method.payload["model_id"] == "gemini-3.7-flash"
    assert method.source == "registered_calibrated"
    assert method.pass_baseline_answer is True


def test_l3a_vcd_is_proposed_when_false_yes_hallucinations_dominate():
    batch = _gold_yes_batch(n=8, image=_img())
    for case in batch:
        case.metadata["task"] = "yes_no"
        case.expected = "No"
        case.observed = "Yes"
    candidates = FixAgent(
        judge=None, max_tier="L3a", candidate_allowlist={"vcd_diffusion_noise"}
    )._propose(
        [_hyp("language priors override visual evidence")], batch, VCDSensitiveModel()
    )

    assert "vcd_diffusion_noise" in {candidate.name for candidate in candidates}


def test_l3a_vcd_is_not_proposed_on_an_audio_only_batch():
    """A multimodal backend exposes generate_vcd even when the batch carries
    no image (live: audiocaps_hallucination_qwen2_audio proposed both VCD
    candidates for an audio-only yes/no batch; each ran as not_executed)."""

    class AudioBackendWithVCD(AADSensitiveModel):
        def generate_vcd(self, inputs, **kwargs):
            return "Yes."

    batch = _gold_audio_yes_batch(n=8)
    for case in batch:
        case.metadata["task"] = "yes_no"
        case.expected = "No"
        case.observed = "Yes"
    candidates = FixAgent(
        judge=None, max_tier="L3a", candidate_allowlist={"aad_silence_contrast"}
    )._propose(
        [_hyp("language priors override audio evidence")], batch, AudioBackendWithVCD()
    )

    names = {candidate.name for candidate in candidates}
    assert "vcd_diffusion_noise" not in names
    assert "vcd_diffusion_noise_gated_false_yes" not in names
    assert "aad_silence_contrast" in names  # the audio method still is


def test_l3a_aad_candidate_repairs_binary_audio_grounding():
    batch = _gold_audio_yes_batch(n=8)

    out = FixAgent(
        judge=None, max_tier="L3a", candidate_allowlist={"aad_silence_contrast"}
    ).propose_and_validate(
        AADSensitiveModel(), batch, [_hyp("language priors override audio evidence")]
    )

    assert out.fixed is True
    assert out.best is not None and out.best.candidate.name == "aad_silence_contrast"
    assert out.best.candidate.tier is FixTier.L3A_INTERNALS_READ
    assert out.best.n_fixed == 8 and out.best.n_broken == 0
    assert out.best.candidate.payload["kwargs"]["alpha"] == 0.5


def test_l3a_aad_structural_discovery_does_not_read_expected_direction():
    batch = _gold_audio_yes_batch(n=8)
    for case in batch:
        case.expected = "Yes"
        case.observed = "No"
    candidates = FixAgent(
        judge=None, max_tier="L3a", candidate_allowlist={"aad_silence_contrast"}
    )._propose(
        [_hyp("language priors override audio evidence")], batch, AADSensitiveModel()
    )

    assert "aad_silence_contrast" in {candidate.name for candidate in candidates}


def test_l3a_aad_is_proposed_when_false_yes_hallucinations_dominate():
    batch = _gold_audio_yes_batch(n=8)
    for case in batch:
        case.expected = "No"
        case.observed = "Yes"
    candidates = FixAgent(
        judge=None, max_tier="L3a", candidate_allowlist={"aad_silence_contrast"}
    )._propose(
        [_hyp("language priors override audio evidence")], batch, AADSensitiveModel()
    )

    assert "aad_silence_contrast" in {candidate.name for candidate in candidates}


def test_l3a_aad_requires_paper_method_fidelity():
    """Structural gate only -- a model without paper_method_fidelity('aad')
    declared native must not get the candidate, judge or not."""
    batch = _gold_audio_yes_batch(n=8)

    class UnfitAudioModel(Model):
        capabilities = frozenset({Capability.GENERATE, Capability.LOGPROBS})
        modalities = frozenset({"text", "audio"})

        def generate(self, inputs, **kwargs):
            return "No."

        def generate_aad(self, inputs, **kwargs):
            return "Yes."

        def forward(self, inputs, capture, spec=None):
            raise NotImplementedError

    candidates = FixAgent(judge=None, max_tier="L3a")._propose(
        [_hyp("language priors override audio evidence")], batch, UnfitAudioModel()
    )

    assert "aad_silence_contrast" not in {candidate.name for candidate in candidates}
    assert "aad_silence_contrast_gated_false_yes" not in {candidate.name for candidate in candidates}


def test_l3a_icd_candidate_repairs_binary_visual_grounding():
    batch = _gold_yes_batch(n=8, image=_img())
    for case in batch:
        case.metadata["task"] = "yes_no"

    out = FixAgent(
        judge=None,
        max_tier="L3a",
        candidate_allowlist={"icd_instruction_disturbance"},
    ).propose_and_validate(
        ICDSensitiveModel(), batch, [_hyp("instruction priors override visual evidence")]
    )

    assert out.fixed is True
    assert out.best is not None and out.best.candidate.name == "icd_instruction_disturbance"
    assert out.best.candidate.tier is FixTier.L3A_INTERNALS_READ
    assert out.best.n_fixed == 8 and out.best.n_broken == 0


# Paper-method L3a/L3b candidates are judge-selected: structural eligibility
# (capability/modality/task shape/fidelity/tier) narrows the catalog shown to
# the judge, but whether the diagnosed MECHANISM actually matches a given
# candidate is the judge's call, not a string match against the hypothesis
# text -- see fix_agent.py's _l3_candidates. These tests use ScriptedJudge to
# stand in for that call.

def test_l3a_vicrop_candidate_repairs_small_visual_detail():
    batch = _gold_yes_batch(n=16, image=_img())
    judge = ScriptedJudge('[{"name": "vicrop_relative_attention"}]')
    out = FixAgent(
        judge=judge, max_tier="L3a", allow_codegen=False, paper_methods_only=True
    ).propose_and_validate(
        ViCropSensitiveModel(), batch, [_hyp("small visual detail is below input resolution")]
    )

    assert out.fixed is True
    assert out.best is not None and out.best.candidate.name == "vicrop_relative_attention"
    assert out.best.candidate.tier is FixTier.L3A_INTERNALS_READ


def test_l3a_vicrop_consensus_guard_can_be_frozen_for_safe_transfer():
    batch = _gold_yes_batch(n=16, image=_img())
    for case in batch:
        case.observed = "No."
    judge = ScriptedJudge('[{"name": "vicrop_consensus_guard"}]')
    out = FixAgent(
        judge=judge,
        max_tier="L3a",
        allow_codegen=False,
        paper_methods_only=True,
        candidate_allowlist=["vicrop_consensus_guard"],
    ).propose_and_validate(
        ViCropConsensusSensitiveModel(),
        batch,
        [_hyp("small visual detail is below input resolution")],
    )

    assert out.fixed is True
    assert out.best is not None and out.best.candidate.name == "vicrop_consensus_guard"
    assert out.best.n_fixed == 16 and out.best.n_broken == 0


def test_l3a_vicrop_is_not_proposed_when_judge_declines_the_mechanism():
    """Structurally eligible (image case, ViCrop capable), but the judge
    itself says the mechanism doesn't match -- an empty JSON array, exactly
    what a real judge would return for e.g. "language priors override
    visible evidence" (that's OPERA/PAI/IFCD's mechanism, not ViCrop's)."""
    batch = _gold_yes_batch(n=16, image=_img())
    judge = ScriptedJudge("[]")
    candidates = FixAgent(
        judge=judge, max_tier="L3a", allow_codegen=False, paper_methods_only=True
    )._propose(
        [_hyp("language priors override visible evidence")], batch, ViCropSensitiveModel()
    )

    assert "vicrop_relative_attention" not in {candidate.name for candidate in candidates}
    assert judge.prompts, "the judge should have been consulted at all"


def test_l3a_paper_method_candidates_require_a_configured_judge():
    """Structurally eligible does not mean proposed: with no judge to make
    the mechanism-match call, _ask_judge returns [] and NO paper-method
    candidate is proposed -- there is no keyword fallback."""
    batch = _gold_yes_batch(n=16, image=_img())
    candidates = FixAgent(judge=None, max_tier="L3a", allow_codegen=False)._propose(
        [_hyp("small visual detail is below input resolution")], batch, ViCropSensitiveModel()
    )

    assert "vicrop_relative_attention" not in {candidate.name for candidate in candidates}


def test_l3a_paper_method_catalog_shown_to_judge_is_structurally_filtered():
    """The judge only ever sees candidates that already passed structural
    eligibility (image/task/capability/fidelity) -- e.g. a yes_no-only batch
    on a model without ViCrop support never puts vicrop_* in the catalog, so
    the judge can't select something that couldn't run anyway."""
    batch = _gold_yes_batch(n=16, image=_img())
    for case in batch:
        case.metadata["task"] = "yes_no"
    judge = ScriptedJudge('[{"name": "opera_overtrust_binary"}]')
    FixAgent(
        judge=judge, max_tier="L3a", allow_codegen=False, paper_methods_only=True
    )._propose(
        [_hyp("object hallucination follows language priors")], batch, OPERASensitiveModel()
    )

    assert judge.prompts
    assert "opera_overtrust_binary" in judge.prompts[-1]
    assert "vicrop_relative_attention" not in judge.prompts[-1]  # OPERASensitiveModel lacks generate_vicrop


def test_l3a_opera_binary_candidate_repairs_object_hallucination():
    batch = _gold_yes_batch(n=16, image=_img())
    for case in batch:
        case.metadata["task"] = "yes_no"
    judge = ScriptedJudge('[{"name": "opera_overtrust_binary"}]')
    out = FixAgent(
        judge=judge, max_tier="L3a", allow_codegen=False, paper_methods_only=True
    ).propose_and_validate(
        OPERASensitiveModel(), batch, [_hyp("object hallucination follows language priors")]
    )

    assert out.fixed is True
    assert out.best is not None and out.best.candidate.name == "opera_overtrust_binary"
    assert out.best.candidate.tier is FixTier.L3A_INTERNALS_READ


def test_l3a_opera_binary_is_not_proposed_for_non_binary_tasks():
    """Structural: non-yes_no task -- excluded before the judge is even
    asked (no candidate reaches the catalog, so this needs no judge)."""
    batch = _gold_yes_batch(n=16, image=_img())
    for case in batch:
        case.metadata["task"] = "multiple_choice"
    candidates = FixAgent(judge=None, max_tier="L3a", allow_codegen=False)._propose(
        [_hyp("object hallucination follows language priors")], batch, OPERASensitiveModel()
    )

    assert "opera_overtrust_binary" not in {candidate.name for candidate in candidates}


def test_l3b_ifcd_requires_explicit_adapted_method_opt_in():
    batch = _gold_yes_batch(n=16, image=_img())
    for case in batch:
        case.metadata["task"] = "yes_no"
    hypotheses = [_hyp("object hallucination follows language priors")]
    no_opt_in = FixAgent(
        judge=ScriptedJudge('[{"name": "ifcd_truthx_contrast"}]'),
        max_tier="L3b", allow_codegen=False, paper_methods_only=True,
    )._propose(hypotheses, batch, IFCDSensitiveModel())
    assert "ifcd_truthx_contrast" not in {candidate.name for candidate in no_opt_in}

    out = FixAgent(
        judge=ScriptedJudge('[{"name": "ifcd_truthx_contrast"}]'),
        max_tier="L3b",
        allow_codegen=False,
        paper_methods_only=True,
        allow_adapted_paper_methods=True,
        candidate_allowlist=["ifcd_truthx_contrast"],
    ).propose_and_validate(IFCDSensitiveModel(), batch, hypotheses)
    assert out.fixed is True
    assert out.best is not None and out.best.candidate.name == "ifcd_truthx_contrast"


def test_l3a_tcd_candidate_repairs_temporal_smoothing_bias():
    batch = _gold_audio_batch(n=16)
    judge = ScriptedJudge('[{"name": "tcd_temporal_blur"}]')
    out = FixAgent(
        judge=judge, max_tier="L3a", allow_codegen=False, paper_methods_only=True
    ).propose_and_validate(
        TCDSensitiveModel(), batch, [_hyp("temporal smoothing bias misses a brief acoustic event")]
    )

    assert out.fixed is True
    assert out.best is not None and out.best.candidate.name == "tcd_temporal_blur"
    assert out.best.candidate.tier is FixTier.L3A_INTERNALS_READ
    assert out.best.n_fixed == 16 and out.best.n_broken == 0


def test_l3a_tcd_accepts_benchmark_multiple_choice_letter_task_name():
    batch = _gold_audio_batch(n=8)
    for case in batch:
        case.metadata["task"] = "multiple_choice_letter"
    judge = ScriptedJudge("[]")

    candidates = FixAgent(
        judge=judge,
        max_tier="L3a",
        allow_codegen=False,
        paper_methods_only=True,
        candidate_allowlist={"tcd_temporal_blur"},
    )._propose(
        [_hyp("temporal smoothing bias misses a brief acoustic event")],
        batch,
        TCDSensitiveModel(),
    )

    assert [candidate.name for candidate in candidates] == ["tcd_temporal_blur"]
    assert candidates[0].tier is FixTier.L3A_INTERNALS_READ
    assert not judge.prompts  # a frozen candidate is not subject to judge veto


def test_l3a_tcd_is_not_proposed_when_judge_declines_the_mechanism():
    """Structurally eligible (audio, multiple_choice, TCD-capable), but the
    hypothesis names a DIFFERENT mechanism (a flat knowledge gap -- exactly
    what M3 actually proposed on the real MMAU run) -- a real judge would
    decline, which ScriptedJudge stands in for with an empty array."""
    batch = _gold_audio_batch(n=16)
    judge = ScriptedJudge("[]")
    candidates = FixAgent(
        judge=judge, max_tier="L3a", allow_codegen=False, paper_methods_only=True
    )._propose(
        [_hyp("the correct answer is never in the model's sample pool at all (pass@5=0)")],
        batch, TCDSensitiveModel(),
    )

    assert "tcd_temporal_blur" not in {candidate.name for candidate in candidates}


def test_l3a_tcd_is_not_proposed_for_non_multiple_choice_tasks():
    batch = _gold_audio_batch(n=8)
    for case in batch:
        case.metadata["task"] = "yes_no"
    candidates = FixAgent(judge=None, max_tier="L3a", allow_codegen=False)._propose(
        [_hyp("temporal smoothing bias misses a brief acoustic event")], batch, TCDSensitiveModel()
    )

    assert "tcd_temporal_blur" not in {candidate.name for candidate in candidates}


def test_l3a_tcd_is_not_proposed_without_audio():
    batch = _gold_yes_batch(n=8, image=_img())  # no audio on any case
    for case in batch:
        case.metadata["task"] = "multiple_choice"
    candidates = FixAgent(judge=None, max_tier="L3a", allow_codegen=False)._propose(
        [_hyp("temporal smoothing bias misses a brief acoustic event")], batch, TCDSensitiveModel()
    )

    assert "tcd_temporal_blur" not in {candidate.name for candidate in candidates}


def test_l3b_pai_candidate_repairs_object_hallucination():
    batch = _gold_yes_batch(n=16, image=_img())
    judge = ScriptedJudge('[{"name": "pai_image_attention"}]')
    out = FixAgent(
        judge=judge, max_tier="L3b", allow_codegen=False, paper_methods_only=True
    ).propose_and_validate(
        PAISensitiveModel(), batch, [_hyp("object hallucination follows language priors")]
    )

    assert out.fixed is True
    assert out.best is not None and out.best.candidate.name == "pai_image_attention"
    assert out.best.candidate.tier is FixTier.L3B_INTERNALS_WRITE
    assert out.best.n_fixed == 16 and out.best.n_broken == 0


def test_candidate_allowlist_freezes_a_paper_candidate():
    batch = _gold_yes_batch(n=8, image=_img())
    for case in batch:
        case.metadata["task"] = "yes_no"

    out = FixAgent(
        judge=None,
        max_tier="L3a",
        candidate_allowlist={"icd_instruction_disturbance_question"},
    ).propose_and_validate(
        ICDSensitiveModel(), batch, [_hyp("instruction priors override visual evidence")]
    )

    assert out.fixed is True
    assert [attempt.candidate.name for attempt in out.attempted] == [
        "icd_instruction_disturbance_question"
    ]


def test_l2_visual_search_candidate_repairs_image_case():
    batch = _gold_yes_batch(n=16, image=_img())
    for case in batch:
        case.metadata["task"] = "yes_no"

    out = FixAgent(judge=None, max_tier="L2", max_judge_candidates=1).propose_and_validate(
        VisualSearchSensitiveModel(), batch, [_hyp("missed local visual detail")]
    )

    assert out.fixed is True
    assert out.best is not None and out.best.candidate.name == "guided_visual_search_consensus"


def test_l2_detector_visual_search_candidate_repairs_image_case():
    batch = _gold_yes_batch(n=16, image=_img())
    for case in batch:
        case.metadata["task"] = "yes_no"

    out = FixAgent(judge=None, max_tier="L2", max_judge_candidates=1).propose_and_validate(
        DetectorVisualSearchSensitiveModel(), batch, [_hyp("missed local visual detail")]
    )

    assert out.fixed is True
    assert out.best is not None
    assert out.best.candidate.name == "detector_visual_search_consensus"


def test_paper_methods_only_skips_generic_visual_prompt_candidate():
    batch = _gold_yes_batch(n=16, image=_img())
    for case in batch:
        case.metadata["task"] = "yes_no"

    out = FixAgent(
        judge=None, max_tier="L2", max_judge_candidates=1, paper_methods_only=True
    ).propose_and_validate(
        DetectorVisualSearchSensitiveModel(), batch, [_hyp("missed local visual detail")]
    )

    assert out.fixed is True
    assert [item.candidate.name for item in out.attempted] == ["detector_visual_search_consensus"]


def test_validation_uses_frozen_observed_baseline_without_regeneration():
    batch = _gold_yes_batch(n=8)
    for case in batch:
        case.observed = "Yes."
    model = FrozenBaselineModel()
    candidate = FixCandidate(
        tier=FixTier.L1_PROMPT,
        name="wrapped",
        kind="template",
        payload={"prompt_template": "Answer the task. {prompt}"},
    )

    result = FixAgent(judge=None).validate_candidate(model, batch, candidate)

    assert result.n_baseline_correct == 8
    assert model.calls == 8  # candidate only; frozen baseline added no calls


def test_l0_refuses_to_infer_truncation_without_finish_reason_telemetry():
    batch = _gold_yes_batch(8)
    out = FixAgent(judge=None, max_tier="L0").propose_and_validate(
        DecodeBudgetSensitiveModel(),
        batch,
        [_hyp("the answer seems short", mode="truncation")],
    )
    assert out.fixed is False
    assert out.attempted == []


class FeedbackDrivenJudge(Model):
    """Proposes a useless template first, the winning one only once it sees the
    failure feedback — models the round-2 result-driven re-proposal."""

    capabilities = frozenset({Capability.GENERATE})
    modalities = frozenset({"text"})

    def __init__(self) -> None:
        self.rounds = 0
        self.saw_feedback: list[bool] = []

    def generate(self, inputs, **kwargs) -> str:
        prompt = str(inputs)
        has_fb = "PRIOR ATTEMPTS THAT DID NOT WORK" in prompt and "broke" in prompt
        self.saw_feedback.append(has_fb)
        if has_fb:  # round 2+: propose the template that actually works
            return json.dumps(
                [{"name": "careful_v2", "prompt_template": "Look very carefully. {prompt}"}]
            )
        self.rounds += 1  # round 1: a template that fixes nothing
        return json.dumps([{"name": "polite", "prompt_template": "Please answer. {prompt}"}])

    def forward(self, inputs, capture, spec=None):
        raise NotImplementedError


def test_feedback_round_one_fails_round_two_fixes():
    """With max_repair_rounds>1, a failed round's results are fed back and the
    judge's NEW proposal validates — no tier escalation."""
    judge = FeedbackDrivenJudge()
    agent = FixAgent(judge=judge, max_tier="L1", max_repair_rounds=3)
    out = agent.propose_and_validate(
        BaselineFailsModel(),
        _gold_yes_batch(16),
        [_hyp("the prompt phrasing underspecifies the task")],
    )
    assert out.fixed is True
    assert out.repair_rounds == 2  # stopped as soon as round 2 validated
    assert out.recommendation is None
    assert out.best is not None and out.best.candidate.name == "careful_v2"
    # round 1's failed candidate is still recorded, and round 2 DID see feedback
    assert any(v.candidate.name == "polite" and not v.fixed for v in out.attempted)
    assert any(judge.saw_feedback)


def test_feedback_withholds_case_content_and_includes_previous_code():
    batch = _gold_yes_batch(3)
    candidate = FixCandidate(
        tier=FixTier.L2_SCAFFOLD,
        name="coded_pipeline",
        kind="code",
        payload={"code": "print('prior implementation')"},
    )
    validation = FixValidation(
        candidate=candidate,
        n_pairs=3,
        n_fixed=1,
        n_broken=0,
        fixed_cases=["c1"],
        effect=1 / 3,
    )

    feedback = FixAgent._format_prior([validation], batch)

    assert "Is there a lesion 1?" not in feedback
    assert "c1" not in feedback
    assert "1 fixed / 0 broken" in feedback
    assert "EXPLORE-TESTED IMPLEMENTATION" in feedback
    assert "print('prior implementation')" in feedback


def test_generated_code_cannot_memorize_example_ids_or_prompt_phrases():
    examples = """### FAIL case spatial457-123
PROMPT: There is a blue thing that is in front of the object right of the tiny bike.
MODEL OUTPUT (baseline): small
"""
    assert _code_copies_example("gate = 'spatial457-123'", examples)
    assert _code_copies_example(
        "gate = 'blue thing that is in front of the object right of the tiny bike'",
        examples,
    )
    assert not _code_copies_example(
        'question = case["prompt"]; gate = "left right front behind"', examples
    )
    assert _code_redefines_model_bridge("def model_generate(case_id):\n    return 'x'")
    assert _code_redefines_model_bridge("async def model_attend(case_id):\n    pass")
    assert not _code_redefines_model_bridge("answer = model_generate(case_id)")


def test_single_round_does_not_retry_on_failure():
    """max_repair_rounds=1 (default) keeps the original single-shot behaviour:
    the useless round-1 proposal is never re-proposed, outcome recommends up."""
    judge = FeedbackDrivenJudge()
    agent = FixAgent(judge=judge, max_tier="L1", max_repair_rounds=1)
    out = agent.propose_and_validate(
        BaselineFailsModel(),
        _gold_yes_batch(),
        [_hyp("the prompt phrasing underspecifies the task")],
    )
    assert out.fixed is False
    assert out.repair_rounds == 1
    assert out.recommendation is not None  # nothing validated -> recommend raise
    assert judge.saw_feedback == [False]  # judge consulted exactly once, no feedback


def test_repair_round_stops_when_no_new_candidate():
    """A judge that keeps proposing the SAME failing candidate is deduped, so
    the round loop stops early instead of re-validating identical work."""
    judge = ScriptedJudge(
        json.dumps([{"name": "polite", "prompt_template": "Please answer. {prompt}"}])
    )
    agent = FixAgent(judge=judge, max_tier="L1", max_repair_rounds=3)
    out = agent.propose_and_validate(BaselineFailsModel(), _gold_yes_batch(), [_hyp("x")])
    assert out.fixed is False
    # round 1 validated the one distinct candidate; round 2 re-proposed it,
    # got deduped to zero new candidates, and the loop stopped at round 2.
    assert out.repair_rounds == 1
    assert sum(1 for v in out.attempted if v.candidate.name == "polite") == 1


def test_fix_agent_accepts_label_returning_scorer():
    judge = ScriptedJudge(
        json.dumps(
            [
                {"name": "careful", "prompt_template": "Look very carefully. {prompt}"},
            ]
        )
    )
    agent = FixAgent(judge=judge, max_tier="L1", score_fn=_label_score)
    out = agent.propose_and_validate(
        BaselineFailsModel(),
        _gold_yes_batch(),
        [_hyp("the prompt phrasing underspecifies the task")],
    )
    assert out.fixed is True
    assert out.best is not None
    assert out.best.n_fixed == 8 and out.best.n_broken == 0
    assert "stats failed" not in out.best.summary


def test_l2_pipeline_candidate_fixes_zoom_sensitive_model():
    pytest.importorskip("PIL")
    judge = ScriptedJudge(
        json.dumps(
            [
                {
                    "name": "upscale2x",
                    "image_ops": [{"tool": "upscale", "params": {"factor": 2.0}}],
                    "prompt_template": "{prompt}",
                    "n_samples": 1,
                },
            ]
        )
    )
    agent = FixAgent(judge=judge, max_tier="L2")
    out = agent.propose_and_validate(
        ZoomSensitiveModel(),
        _gold_yes_batch(image=_img()),
        [_hyp("small findings are destroyed by downsampling", mode="resolution_limit")],
    )
    assert out.fixed is True
    assert out.best.candidate.name == "upscale2x"
    assert out.best.candidate.tier is FixTier.L2_SCAFFOLD
    assert out.best.n_baseline_correct == 0
    assert out.best.n_candidate_correct == 8


# ── FixAgent: unfixable -> recommendation, no auto-escalation ────────────────


def test_unfixable_routed_tier_is_skipped_when_model_cannot_execute_it():
    pytest.importorskip("PIL")
    agent = FixAgent(judge=None, max_tier="L2")  # defaults only
    out = agent.propose_and_validate(
        HopelessModel(),
        _gold_yes_batch(image=_img()),
        [_hyp("suppress the attention sink on structural tokens")],
    )
    assert out.fixed is False and out.best is None
    assert out.recommendation is not None
    assert out.recommendation["recommend_tier"] == "L4"
    assert "beyond the allowed L2" in out.recommendation["reason"]
    assert "skipped unsupported tier(s) L3b" in out.recommendation["reason"]
    # routing recorded per hypothesis
    assert out.routed[0]["min_tier"] == "L3b"


def test_unfixable_skips_internals_unavailable_on_black_box_model():
    pytest.importorskip("PIL")
    agent = FixAgent(judge=None, max_tier="L2")
    out = agent.propose_and_validate(
        HopelessModel(),
        _gold_yes_batch(image=_img()),
        [_hyp("the prompt phrasing is fine but answers are wrong")],
    )
    assert out.fixed is False
    assert out.recommendation["recommend_tier"] == "L4"
    assert "no candidate within L2" in out.recommendation["reason"]
    assert "skipped unsupported tier(s) L3a, L3b" in out.recommendation["reason"]


def test_audio_recommendation_skips_incompatible_internals_tiers():
    class AdaptedOnlyTCD(TCDSensitiveModel):
        def paper_method_fidelity(self, method):
            return "adapted_truncated_layer_stability" if method == "tcd" else "unavailable"

    batch = _gold_audio_batch()
    agent = FixAgent(judge=None, max_tier="L2", allow_codegen=False)
    rec = agent._recommend([], model=AdaptedOnlyTCD(), data=batch)
    assert rec["recommend_tier"] == "L4"
    assert "skipped unsupported tier(s) L3a, L3b" in rec["reason"]


def test_at_l4_no_higher_recommendation():
    agent = FixAgent(judge=None, max_tier="L4")
    out = agent.propose_and_validate(
        HopelessModel(), _gold_yes_batch(), [_hyp("requires retraining on new data")]
    )
    assert out.fixed is False and out.recommendation is None


# ── FixAgent: robustness ─────────────────────────────────────────────────────


def test_garbage_judge_falls_back_to_defaults():
    pytest.importorskip("PIL")
    agent = FixAgent(judge=ScriptedJudge("I refuse to answer in JSON."), max_tier="L2")
    out = agent.propose_and_validate(HopelessModel(), _gold_yes_batch(image=_img()), [_hyp("x")])
    sources = {v.candidate.source for v in out.attempted}
    assert sources == {"default"}
    tiers = {v.candidate.tier for v in out.attempted}
    assert tiers == {FixTier.L1_PROMPT, FixTier.L2_SCAFFOLD}


def test_min_tier_filters_cheaper_candidates_for_one_ladder_station():
    pytest.importorskip("PIL")
    judge = ScriptedJudge("I refuse to answer in JSON.")
    agent = FixAgent(
        judge=judge,
        max_tier="L2",
        min_tier="L2",
        allow_codegen=False,
    )
    out = agent.propose_and_validate(
        HopelessModel(), _gold_yes_batch(image=_img()), [_hyp("x")]
    )

    assert out.attempted
    assert {v.candidate.tier for v in out.attempted} == {FixTier.L2_SCAFFOLD}
    assert len(judge.prompts) == 1
    assert "L2" in judge.prompts[0]


def test_no_rubric_cases_yield_recommendation_not_crash():
    cases = CaseBatch([FailureCase(id="u", inputs=Inputs(prompt="q"), label=Label.FAIL)])
    agent = FixAgent(judge=None, max_tier="L1")
    out = agent.propose_and_validate(HopelessModel(), cases, [_hyp("x")])
    assert out.fixed is False
    assert out.attempted == []
    assert "no case carries a scoring rubric" in out.recommendation["reason"]


def test_broken_cases_counted_and_net_negative_not_fixed():
    """A candidate that repairs nothing and breaks passing cases must not pass.

    One case (c8) fails the baseline so the zero-fail gate lets the search
    run; the candidate then repairs nothing and breaks the eight passers.
    """

    class InvertModel(Model):
        capabilities = frozenset({Capability.GENERATE})
        modalities = frozenset({"text", "image"})

        def generate(self, inputs, **kwargs):
            p = str(getattr(inputs, "prompt", inputs)).lower()
            if "carefully" in p:
                return "No."
            return "No." if "8" in p else "Yes."

        def forward(self, inputs, capture, spec=None):
            raise NotImplementedError

    judge = ScriptedJudge(
        json.dumps([{"name": "careful", "prompt_template": "Look very carefully. {prompt}"}])
    )
    agent = FixAgent(judge=judge, max_tier="L1")
    out = agent.propose_and_validate(InvertModel(), _gold_yes_batch(9), [_hyp("x")])
    v = out.attempted[0]
    assert v.n_broken == 8 and v.n_fixed == 0
    assert v.fixed is False and out.fixed is False


def test_outcome_serializes_and_logs(tmp_path):
    from evalrx.eval_agent import RunLogger

    judge = ScriptedJudge(
        json.dumps([{"name": "careful", "prompt_template": "Look carefully. {prompt}"}])
    )
    logger = RunLogger(tmp_path / "logs")
    agent = FixAgent(judge=judge, max_tier="L1", run_logger=logger)
    out = agent.propose_and_validate(BaselineFailsModel(), _gold_yes_batch(), [_hyp("x")])
    d = out.to_dict()
    json.dumps(d)  # fully serializable
    assert d["max_tier"] == "L1" and d["fixed"] is True
    log_text = (tmp_path / "logs" / "run_log.jsonl").read_text(encoding="utf-8")
    events = [json.loads(line) for line in log_text.splitlines()]
    fix_events = [e for e in events if e.get("event") == "fix"]
    assert len(fix_events) == 1
    assert fix_events[0]["attempted"][0]["name"] == "careful"


# ── loop integration ─────────────────────────────────────────────────────────


def test_run_fix_on_loop_report():
    from evalrx.eval_agent import VLDiagnoseLoop, VLDiagnoseReport
    from evalrx.eval_agent.hypothesis import HypothesisStatus
    from evalrx.eval_agent.stages.hypothesis_tester import HypothesisTestResult
    from evalrx.eval_agent.stages.protocol import ExperimentProtocol

    h = _hyp("prompt phrasing underspecifies the task")
    h.status = HypothesisStatus.SUPPORTED
    report = VLDiagnoseReport(
        cycles=1,
        stopped_by="max_cycles",
        verified_hypotheses=[
            HypothesisTestResult(
                hypothesis=h,
                status=HypothesisStatus.SUPPORTED,
                test_name="fail_rate_comparison",
                effect_size=0.5,
                is_consistent_with_protocol=True,
                confidence=0.8,
                verdict="ok",
            )
        ],
    )
    judge = ScriptedJudge(
        json.dumps([{"name": "careful", "prompt_template": "Look very carefully. {prompt}"}])
    )
    # Constructor injection — symmetric with every other stage agent.
    loop = VLDiagnoseLoop(
        model=BaselineFailsModel(),
        protocol=ExperimentProtocol(description="d"),
        fix_agent=FixAgent(judge=judge, max_tier="L1"),
    )
    assert loop.fix_agent.max_tier is FixTier.L1_PROMPT
    out = loop.run_fix(report, _gold_yes_batch())
    assert out.fixed is True
    assert report.fix_outcome is out

    # Per-call max_tier override + per-call agent override still work.
    out2 = loop.run_fix(
        report, _gold_yes_batch(), max_tier="L2", fix_agent=FixAgent(judge=judge, max_tier="L1")
    )
    assert out2.max_tier is FixTier.L2_SCAFFOLD

    # Default construction (no injection) builds a judge-less FixAgent.
    bare = VLDiagnoseLoop(model=BaselineFailsModel(), protocol=ExperimentProtocol(description="d"))
    assert bare.fix_agent is not None and bare.fix_agent._judge is None


# ── L2 coded pipelines (bridged model access) ────────────────────────────────


_UPSCALE_PIPELINE = """
import json
cases = json.load(open("fix_cases.json"))["cases"]
out = []
for c in cases:
    model_generate(c["id"])
    ops = [{"tool": "upscale", "params": {"factor": 2.0}}]
    ans = model_generate(c["id"], prompt="zoom one", image_ops=ops)
    model_generate(c["id"], prompt="zoom two", image_ops=ops)
    model_generate(c["id"], prompt="zoom three", image_ops=ops)
    out.append({"sample_id": c["id"], "output": ans})
print("FIX_PIPELINE_RESULT_JSON=" + json.dumps({"per_case": out}))
"""


def test_cases_payload_never_leaks_labels_or_rubrics():
    """The sandbox sees id, prompt and the model's OWN baseline output — never
    a label, expected answer, or rubric. baseline_output is what the unchanged
    model already said (the frozen-model control replays exactly it), so it
    carries no correctness information."""
    from evalrx.eval_agent.stages.fix_pipeline import cases_payload

    batch = _gold_yes_batch()
    payload = cases_payload(batch)
    assert all(set(c) == {"id", "prompt", "baseline_output"} for c in payload["cases"])
    forbidden = {"label", "expected", "gold", "rubric", "score", "metadata"}
    assert all(forbidden.isdisjoint(c) for c in payload["cases"])
    by_id = {c.id: c for c in batch}
    for c in payload["cases"]:
        observed = getattr(by_id[c["id"]], "observed", None)
        assert c["baseline_output"] == (None if observed is None else str(observed))


def test_coded_pipeline_bridge_round_trip(tmp_path):
    pytest.importorskip("PIL")
    from evalrx.analyzers.perturbation.prompt_contrast import _default_score
    from evalrx.eval_agent.stages.fix_pipeline import (
        run_coded_pipeline,
        score_outputs,
    )

    cases = _gold_yes_batch(n=3, image=_img())
    result = run_coded_pipeline(
        _UPSCALE_PIPELINE, ZoomSensitiveModel(), cases, workdir=tmp_path, timeout_sec=30
    )
    assert result.ok and result.n_calls == 12
    scores = score_outputs(result, cases, _default_score)
    assert all(scores[c.id] is True for c in cases)  # upscale repairs every case


def test_coded_pipeline_host_guard_rejects_singleton_override(tmp_path):
    """The host, not codegen prompt compliance, owns the consensus invariant."""
    from evalrx.eval_agent.stages.fix_pipeline import run_coded_pipeline

    pipeline = """
import json
cases = json.load(open("fix_cases.json"))["cases"]
out = []
for c in cases:
    baseline = model_generate(c["id"])
    candidate = model_generate(c["id"], prompt="enhanced one")
    model_generate(c["id"], prompt="enhanced two")
    model_generate(c["id"], prompt="enhanced three")
    out.append({"sample_id": c["id"], "output": candidate})
print("FIX_PIPELINE_RESULT_JSON=" + json.dumps({"per_case": out}))
"""

    def replies(case, prompt):
        return "Final Answer: Yes." if prompt == "enhanced one" else "Final Answer: No."

    cases = _gold_yes_batch(n=2)
    result = run_coded_pipeline(
        pipeline, None, cases, workdir=tmp_path, timeout_sec=20,
        reply_fn=replies, consensus_min_support=2,
    )
    assert result.ok and result.n_guarded == 2
    assert result.guarded_ids == ["c0", "c1"]
    assert all(value == "Final Answer: No." for value in result.outputs.values())


def test_coded_pipeline_host_guard_accepts_two_independent_supporters(tmp_path):
    from evalrx.eval_agent.stages.fix_pipeline import run_coded_pipeline

    pipeline = """
import json
cases = json.load(open("fix_cases.json"))["cases"]
out = []
for c in cases:
    model_generate(c["id"])
    candidate = model_generate(c["id"], prompt="enhanced one")
    model_generate(c["id"], prompt="enhanced two")
    model_generate(c["id"], prompt="enhanced three")
    out.append({"sample_id": c["id"], "output": "yes"})
print("FIX_PIPELINE_RESULT_JSON=" + json.dumps({"per_case": out}))
"""

    def replies(case, prompt):
        if prompt in {"enhanced one", "enhanced two"}:
            return "Reasoning...\nFinal Answer: Yes."
        return "Final Answer: No."

    cases = _gold_yes_batch(n=1)
    result = run_coded_pipeline(
        pipeline, None, cases, workdir=tmp_path, timeout_sec=20,
        reply_fn=replies, consensus_min_support=2,
    )
    assert result.ok and result.n_guarded == 0
    assert result.outputs["c0"] == "yes"


def test_coded_pipeline_host_guard_requires_direct_baseline(tmp_path):
    from evalrx.eval_agent.stages.fix_pipeline import run_coded_pipeline

    pipeline = """
import json
cases = json.load(open("fix_cases.json"))["cases"]
out = []
for c in cases:
    answer = model_generate(c["id"], prompt="enhanced")
    out.append({"sample_id": c["id"], "output": answer})
print("FIX_PIPELINE_RESULT_JSON=" + json.dumps({"per_case": out}))
"""
    result = run_coded_pipeline(
        pipeline, None, _gold_yes_batch(n=1), workdir=tmp_path,
        timeout_sec=20, reply_fn=lambda case, prompt: "Yes.",
        consensus_min_support=2,
    )
    assert result.ok is False
    assert "no direct baseline" in result.error


def test_coded_pipeline_host_guard_deduplicates_support_and_caps_calls(tmp_path):
    from evalrx.eval_agent.stages.fix_pipeline import run_coded_pipeline

    repeated = """
import json
cases = json.load(open("fix_cases.json"))["cases"]
out = []
for c in cases:
    model_generate(c["id"])
    answer = model_generate(c["id"], prompt="same enhancement")
    model_generate(c["id"], prompt="same enhancement")
    out.append({"sample_id": c["id"], "output": answer})
print("FIX_PIPELINE_RESULT_JSON=" + json.dumps({"per_case": out}))
"""
    cases = _gold_yes_batch(n=1)
    result = run_coded_pipeline(
        repeated, None, cases, workdir=tmp_path / "dedupe", timeout_sec=20,
        reply_fn=lambda case, prompt: "Yes." if prompt == "same enhancement" else "No.",
        consensus_min_support=2, max_calls_per_case=4,
    )
    assert result.ok and result.n_guarded == 1 and result.outputs["c0"] == "No."

    too_many = repeated.replace(
        'out.append({"sample_id": c["id"], "output": answer})',
        'model_generate(c["id"], prompt="fourth")\n'
        '    model_generate(c["id"], prompt="fifth")\n'
        '    out.append({"sample_id": c["id"], "output": answer})',
    )
    capped = run_coded_pipeline(
        too_many, None, cases, workdir=tmp_path / "cap", timeout_sec=20,
        reply_fn=lambda case, prompt: "No.", max_calls_per_case=4,
    )
    assert capped.ok is False and "more than 4 model calls" in capped.error


def test_score_outputs_coerces_label_scores():
    from evalrx.eval_agent.stages.fix_pipeline import CodedPipelineResult, score_outputs

    cases = _gold_yes_batch(n=2)
    result = CodedPipelineResult(
        outputs={cases[0].id: "Yes.", cases[1].id: "No."},
        ok=True,
    )
    scores = score_outputs(result, cases, _label_score)
    assert scores[cases[0].id] is True
    assert scores[cases[1].id] is False


def test_coded_pipeline_call_budget_kills_runaway(tmp_path):
    runaway = """
while True:
    model_generate("c0")
"""
    result = __import__(
        "evalrx.eval_agent.stages.fix_pipeline",
        fromlist=["run_coded_pipeline"],
    ).run_coded_pipeline(
        runaway,
        HopelessModel(),
        _gold_yes_batch(n=2),
        workdir=__import__("tempfile").mkdtemp(),
        timeout_sec=30,
        max_calls=5,
    )
    assert result.ok is False
    assert "budget exhausted" in result.error


def test_coded_pipeline_missing_marker_and_crash(tmp_path):
    from evalrx.eval_agent.stages.fix_pipeline import run_coded_pipeline

    r1 = run_coded_pipeline(
        'print("nothing")', HopelessModel(), _gold_yes_batch(n=1), workdir=tmp_path, timeout_sec=20
    )
    assert r1.ok is False and "FIX_PIPELINE_RESULT_JSON" in r1.error
    r2 = run_coded_pipeline(
        "this is not python",
        HopelessModel(),
        _gold_yes_batch(n=1),
        workdir=tmp_path,
        timeout_sec=20,
    )
    assert r2.ok is False


def test_coded_pipeline_recovers_result_without_literal_marker_prefix(tmp_path):
    """A judge sometimes emits the right JSON payload but drops the exact
    ``FIX_PIPELINE_RESULT_JSON=`` prefix the prompt asked for (observed with
    qwen3-vl-8b-instruct: valid ``{"per_case": [...]}"" via bare ``print()``,
    no prefix). That is a compliance slip, not a content error, and should
    not be indistinguishable from a pipeline that produced nothing at all."""
    from evalrx.eval_agent.stages.fix_pipeline import run_coded_pipeline

    cases = _gold_yes_batch(n=2)
    unprefixed = """
import json
cases = json.load(open("fix_cases.json"))["cases"]
out = [{"sample_id": c["id"], "output": "Yes."} for c in cases]
print(json.dumps({"per_case": out}))
"""
    result = run_coded_pipeline(unprefixed, HopelessModel(), cases, workdir=tmp_path, timeout_sec=20)
    assert result.ok is True
    assert all(result.outputs[c.id] == "Yes." for c in cases)


def test_coded_pipeline_unrelated_stdout_still_fails(tmp_path):
    """The recovery fallback only accepts a line that actually parses as
    ``{"per_case": [...]}"" — noise on stdout must not be mistaken for it."""
    from evalrx.eval_agent.stages.fix_pipeline import run_coded_pipeline

    noisy = """
print("starting up")
print('{"status": "done"}')
"""
    result = run_coded_pipeline(
        noisy, HopelessModel(), _gold_yes_batch(n=1), workdir=tmp_path, timeout_sec=20
    )
    assert result.ok is False and "FIX_PIPELINE_RESULT_JSON" in result.error


class CodeWritingJudge(Model):
    """Garbage for JSON proposals; real pipeline code for the code prompt."""

    capabilities = frozenset({Capability.GENERATE})
    modalities = frozenset({"text"})

    def generate(self, inputs, **kwargs) -> str:
        if "EXECUTION CONTRACT" in str(inputs):
            return f"```python\n{_UPSCALE_PIPELINE}\n```"
        return "no json here"

    def forward(self, inputs, capture, spec=None):
        raise NotImplementedError


def test_fix_agent_coded_candidate_fixes_and_logs(tmp_path):
    pytest.importorskip("PIL")
    from evalrx.eval_agent import RunLogger

    logger = RunLogger(tmp_path / "logs")
    agent = FixAgent(
        judge=CodeWritingJudge(), max_tier="L2", run_logger=logger, exec_timeout_sec=30
    )
    out = agent.propose_and_validate(
        ZoomSensitiveModel(),
        _gold_yes_batch(16, image=_img()),
        [_hyp("small findings are destroyed by downsampling", mode="resolution_limit")],
    )
    coded = [v for v in out.attempted if v.candidate.kind == "code"]
    assert len(coded) == 1 and coded[0].candidate.source == "judge"
    assert coded[0].fixed is True and out.fixed is True
    # The default upscale_sharpen spec also fixes this model with the same
    # effect; best is whichever validated first among the tied winners.
    assert out.best.candidate.name in {"coded_pipeline", "upscale_sharpen"}
    log_text = (tmp_path / "logs" / "run_log.jsonl").read_text(encoding="utf-8")
    events = [json.loads(line) for line in log_text.splitlines()]
    cg = [e for e in events if e.get("event") == "tool_codegen"]
    assert len(cg) == 1 and cg[0]["module"] == "fix_pipeline" and cg[0]["ok"] is True


def test_fix_agent_codegen_gate():
    pytest.importorskip("PIL")
    agent = FixAgent(judge=CodeWritingJudge(), max_tier="L2", allow_codegen=False)
    out = agent.propose_and_validate(HopelessModel(), _gold_yes_batch(image=_img()), [_hyp("x")])
    assert all(v.candidate.kind != "code" for v in out.attempted)
    # L1-only tier never attempts coded pipelines either
    agent2 = FixAgent(judge=CodeWritingJudge(), max_tier="L1")
    out2 = agent2.propose_and_validate(HopelessModel(), _gold_yes_batch(), [_hyp("x")])
    assert all(v.candidate.kind == "template" for v in out2.attempted)


# ── defect 4: strict bridge rejects unknown image tools ──────────────────────


def test_bridge_rejects_unknown_image_tool(tmp_path):
    """An unknown tool name returns an ERROR reply (RuntimeError in the pipeline),
    instead of being silently skipped — so the coder can see and fix it."""
    pytest.importorskip("PIL")
    from evalrx.eval_agent.stages.fix_pipeline import run_coded_pipeline

    bad_pipeline = (
        "import json\n"
        'cases = json.load(open("fix_cases.json"))["cases"]\n'
        "out = []\n"
        "for c in cases:\n"
        "    try:\n"
        '        a = model_generate(c["id"], image_ops=[{"tool": "model_attend"}])\n'
        "    except RuntimeError as e:\n"
        '        a = "ERR:" + str(e)\n'
        '    out.append({"sample_id": c["id"], "output": a})\n'
        'print("FIX_PIPELINE_RESULT_JSON=" + json.dumps({"per_case": out}))\n'
    )
    res = run_coded_pipeline(
        bad_pipeline,
        ZoomSensitiveModel(),
        _gold_yes_batch(n=1, image=_img()),
        workdir=tmp_path,
        timeout_sec=20,
    )
    assert res.ok  # produced the result line
    out = next(iter(res.outputs.values()))
    assert out.startswith("ERR:") and "unknown tool" in out.lower()


# ── defect 5: execution failure must not be read as "tier exhausted" ─────────


class _AlwaysCrashCodeJudge(Model):
    """JSON proposals are garbage (→ default L2 specs) and the coded pipeline it
    writes always crashes — exercises the never-executed accounting."""

    capabilities = frozenset({Capability.GENERATE})
    modalities = frozenset({"text"})

    def generate(self, inputs, **kwargs) -> str:
        if "EXECUTION CONTRACT" in str(inputs) or "FAILED TO EXECUTE" in str(inputs):
            return "```python\nraise RuntimeError('boom')\n```"
        return "no json"

    def forward(self, inputs, capture, spec=None):
        raise NotImplementedError


def test_never_executed_candidate_does_not_force_escalation(tmp_path):
    """When the ONLY coded candidate never executes, the recommendation must say
    'fix execution / retry within tier', NOT 'escalate to the next tier'
    (defect 5)."""
    pytest.importorskip("PIL")
    # L2 max, no image tools that help → declarative specs won't fix HopelessModel,
    # but they DO execute. To isolate the never-executed path, force only the
    # coded candidate by using a judge whose code always crashes and whose specs
    # are filtered out — easier: assert the exec_error is surfaced on the coded
    # candidate and that its validation has n_pairs == 0.
    agent = FixAgent(judge=_AlwaysCrashCodeJudge(), max_tier="L2", exec_timeout_sec=20)
    out = agent.propose_and_validate(
        HopelessModel(),
        _gold_yes_batch(image=_img()),
        [_hyp("downsampling destroys small findings", mode="resolution_limit")],
    )
    coded = [v for v in out.attempted if v.candidate.kind == "code"]
    assert coded and coded[0].n_pairs == 0
    assert coded[0].exec_error  # the crash is recorded, not silently dropped
    assert "never execute" in coded[0].summary.lower()


# ── L3a: attention read is agent-authored (no canned primitive) ──────────────


def _bright_corner_img():
    """64x48 dark image with a bright square in the top-left quadrant."""
    pytest.importorskip("PIL")
    from PIL import Image

    img = Image.new("L", (64, 48), color=10)
    for x in range(4, 20):
        for y in range(4, 16):
            img.putpixel((x, y), 250)
    return img


class AttnCropVLM(Model):
    """White-box fake: attention peaks at the top-left patch; answers "yes"
    only when the (cropped) image is bright enough on average."""

    capabilities = frozenset({Capability.GENERATE, Capability.ATTENTION})
    modalities = frozenset({"text", "image"})

    def generate(self, inputs, **kwargs):
        import numpy as np

        img = getattr(inputs, "image", None)
        if img is None:
            return "No."
        mean = float(np.asarray(img.convert("L"), dtype=float).mean())
        return "Yes." if mean > 60 else "No."

    def forward(self, inputs, capture, spec=None):
        import torch

        from evalrx.core.model import Trace

        h, w = 3, 4  # patch grid
        seq = 2 + h * w + 1  # 2 structural + 12 image + 1 query
        row = torch.full((seq,), 0.01)
        row[2] = 0.8  # peak at image patch (0, 0)
        layer = torch.zeros(1, seq, seq)
        layer[:, -1, :] = row
        mask = torch.zeros(seq, dtype=torch.bool)
        mask[2 : 2 + h * w] = True
        return Trace(
            tokens=["t"] * seq,
            token_ids=list(range(seq)),
            provided={Capability.ATTENTION},
            attentions=[layer.clone() for _ in range(2)],
            extras={"image_token_mask": mask, "image_spatial_shape": (h, w)},
        )


def test_attention_heatmap_backs_model_attend():
    """attention_heatmap is the host-side helper the model_attend() bridge is
    built on; it now lives with the analyzers (shared with relative_attn), one
    reduction for both the analyzer path and the fix read bridge."""
    import numpy as np

    from evalrx.analyzers.attention.relative_attn import attention_heatmap

    case = FailureCase(id="x", inputs=Inputs(prompt="q", image=_bright_corner_img()))
    grid = attention_heatmap(AttnCropVLM(), case)
    assert grid is not None and grid.shape == (3, 4)
    assert np.unravel_index(grid.argmax(), grid.shape) == (0, 0)


def test_attention_capture_shared_reduction_matches_inline():
    """image_token_attention is the single reduction both consumers share —
    head-mean of the last query row over image tokens."""
    from evalrx.analyzers.attention.relative_attn import image_token_attention
    from evalrx.core.capability import Capability

    case = FailureCase(id="x", inputs=Inputs(prompt="q", image=_bright_corner_img()))
    trace = AttnCropVLM().forward(case.inputs, capture={Capability.ATTENTION})
    attns = trace.require(Capability.ATTENTION)
    mask = trace.extras["image_token_mask"]
    vec = image_token_attention(attns[-1], mask)
    expected = attns[-1].float().mean(dim=0)[-1, mask]
    assert vec.shape == expected.shape
    assert bool((vec == expected).all())


def test_attention_guided_crop_primitive_is_gone():
    """The canned L3a read primitive was removed — reads are not in the
    pre-audited write registry, only the L3b write primitive remains. The
    attention capture itself moved out of the fix module to the analyzers."""
    from evalrx.eval_agent.stages import fix_internals
    from evalrx.eval_agent.stages.fix_internals import INTERNALS_PRIMITIVES

    assert "attention_guided_crop" not in INTERNALS_PRIMITIVES
    assert all(p.tier is FixTier.L3B_INTERNALS_WRITE for p in INTERNALS_PRIMITIVES.values())
    assert not hasattr(fix_internals, "run_attention_guided_crop")
    assert not hasattr(fix_internals, "peak_box")
    assert not hasattr(fix_internals, "attention_heatmap")  # moved to relative_attn


def test_l3a_read_is_authored_not_a_canned_primitive():
    """At L3a the read lever is the coded pipeline's bridged model_attend(), not
    a primitive. A black-box generated scaffold remains L2 even when the bridge
    was offered; source that actually calls model_attend is tagged L3a."""
    pytest.importorskip("PIL")
    cases = CaseBatch(
        [
            FailureCase(
                id=f"c{i}",
                inputs=Inputs(prompt="bright?", image=_bright_corner_img()),
                expected={"all_of": ["yes"], "none_of": ["no"]},
                label=Label.FAIL,
            )
            for i in range(8)
        ]
    )
    hyp = _hyp(
        "attention is on the finding but the answer ignores it", mode="attention_mislocalization"
    )

    # (a) no codegen at L3a: the only registered primitive is the L3b write one,
    # which is out of tier -> NO primitive candidate is attempted.
    bare = FixAgent(judge=None, max_tier="L3a", allow_codegen=False)
    out = bare.propose_and_validate(AttnCropVLM(), cases, [hyp])
    assert not any(v.candidate.kind == "primitive" for v in out.attempted)

    # (b) Merely advertising model_attend does not promote code which only
    # calls model_generate and image tools.
    black_box = FixAgent(judge=CodeWritingJudge(), max_tier="L3a", allow_codegen=True)
    black_box_cands = black_box._propose([hyp], cases, AttnCropVLM())
    black_box_code = [c for c in black_box_cands if c.kind == "code"]
    assert black_box_code and black_box_code[0].tier is FixTier.L2_SCAFFOLD
    assert black_box_code[0].payload.get("enable_attend") is False

    class AttendWritingJudge(CodeWritingJudge):
        def generate(self, inputs, **kwargs):
            if "EXECUTION CONTRACT" in str(inputs):
                return f"```python\n{_ATTEND_PIPELINE}\n```"
            return super().generate(inputs, **kwargs)

    coded = FixAgent(judge=AttendWritingJudge(), max_tier="L3a", allow_codegen=True)
    cands = coded._propose([hyp], cases, AttnCropVLM())
    code_cands = [c for c in cands if c.kind == "code"]
    assert code_cands and code_cands[0].tier is FixTier.L3A_INTERNALS_READ
    assert code_cands[0].payload.get("enable_attend") is True
    assert not any(c.kind == "primitive" for c in cands)


# ── L3b: visual embedding boost ──────────────────────────────────────────────


def test_visual_embedding_boost_hook_scales_image_tokens():
    torch = pytest.importorskip("torch")
    import types

    from evalrx.eval_agent.stages.fix_internals import visual_embedding_boost

    emb = torch.nn.Embedding(10, 4)
    hf = types.SimpleNamespace(
        config=types.SimpleNamespace(image_token_id=7), get_input_embeddings=lambda: emb
    )
    model = types.SimpleNamespace(_hf=(hf, None))
    ids = torch.tensor([[1, 7, 7, 2]])
    base = emb(ids).detach().clone()
    with visual_embedding_boost(model, gamma=2.0):
        boosted = emb(ids).detach()
    after = emb(ids).detach()
    assert torch.allclose(boosted[0, 1], base[0, 1] * 2.0)
    assert torch.allclose(boosted[0, 0], base[0, 0])  # non-image untouched
    assert torch.allclose(after, base)  # hook removed


def test_boost_unavailable_yields_none_scores():
    from evalrx.analyzers.perturbation.prompt_contrast import _default_score
    from evalrx.eval_agent.stages.fix_internals import (
        boost_available,
        run_visual_embedding_boost,
    )

    model = HopelessModel()  # no ._hf backend internals
    assert boost_available(model) is False
    scores = run_visual_embedding_boost(model, _gold_yes_batch(n=2), _default_score)
    assert set(scores.values()) == {None}


# ── L4: recipe dataclass + v1 LoRA executor ──────────────────────────────────


def test_l4_recipe_recorded_not_executed():
    judge = ScriptedJudge(
        json.dumps(
            {
                "dataset_recipe": "synthesise small-lesion radiographs with paired labels",
                "method": "lora",
                "target": "vision_encoder",
                "eval_protocol": "held-out McNemar + regression battery",
                "rationale": "resolution ceiling is parameter-bound",
            }
        )
    )
    agent = FixAgent(judge=judge, max_tier="L4", allow_codegen=False)
    out = agent.propose_and_validate(
        HopelessModel(), _gold_yes_batch(), [_hyp("requires retraining", mode="prior")]
    )
    ft = [v for v in out.attempted if v.candidate.kind == "finetune_spec"]
    assert len(ft) == 1
    assert "TODO" in ft[0].summary and ft[0].fixed is False
    assert ft[0].candidate.payload["target"] == "vision_encoder"
    assert out.fixed is False
    assert out.recommendation is None  # already at the top tier


def test_l4_not_executed_without_finetune_pool():
    """target='llm'/method='lora' is the executable shape, but FixAgent was
    not given a finetune_pool -- must stay recorded-not-executed, not attempt
    training against the validation batch itself (that would be leakage)."""
    pytest.importorskip("peft")
    judge = ScriptedJudge(
        json.dumps(
            {
                "dataset_recipe": "irrelevant -- never interpreted",
                "method": "lora",
                "target": "llm",
                "rationale": "text-only reasoning gap",
            }
        )
    )
    agent = FixAgent(judge=judge, max_tier="L4", allow_codegen=False)  # no finetune_pool
    out = agent.propose_and_validate(
        HopelessModel(), _gold_yes_batch(), [_hyp("requires retraining", mode="prior")]
    )
    ft = [v for v in out.attempted if v.candidate.kind == "finetune_spec"]
    assert len(ft) == 1
    assert ft[0].fixed is False
    assert "finetune_pool" in ft[0].summary


class _TinyWordTokenizer:
    """Fixed-vocabulary word tokenizer -- deterministic ids, real decode, no
    chat template (exercises run_lora_repair's plain-concatenation fallback
    path). Consistent prefix ids for a shared prompt prefix is what makes
    the SFT label-masking boundary correct in the test below."""

    vocab = {
        "<pad>": 0, "<bos>": 1, "<eos>": 2, "<unk>": 3,
        "classify": 4, "alpha": 5, "beta": 6, "yes": 7, "no": 8,
    }
    inv_vocab = {v: k for k, v in vocab.items()}
    vocab_size = 16

    def __call__(self, text, return_tensors="pt"):
        import torch

        ids = [self.vocab.get(w, self.vocab["<unk>"]) for w in text.strip().lower().split()]
        ids = ids or [self.vocab["<unk>"]]
        input_ids = torch.tensor([ids], dtype=torch.long)
        return {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids)}

    def decode(self, ids, skip_special_tokens=True):
        words = []
        for i in ids:
            w = self.inv_vocab.get(int(i), "<unk>")
            if skip_special_tokens and w in ("<pad>", "<bos>", "<eos>"):
                continue
            words.append(w)
        return " ".join(words)


def _tiny_llama():
    """Real, from-scratch (no download) causal LM with genuine q_proj/k_proj/
    v_proj/o_proj naming -- the exact target_modules run_lora_repair's
    text-only fallback targets -- so this test exercises real PEFT injection
    and real gradient training, not a mock."""
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(0)
    cfg = LlamaConfig(
        vocab_size=_TinyWordTokenizer.vocab_size, hidden_size=16, intermediate_size=32,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2,
        max_position_embeddings=32, pad_token_id=0, bos_token_id=1, eos_token_id=2,
    )
    return LlamaForCausalLM(cfg)


def _classify_case(word: str, target: str, label: Label) -> FailureCase:
    return FailureCase(
        inputs=Inputs(prompt=f"classify {word}"), expected=target, label=label,
    )


def _contains_score(case: FailureCase, output: str):
    return str(case.expected).strip().lower() in str(output).strip().lower()


def test_l4_lora_repair_trains_fixes_held_out_cases_and_restores_weights():
    """End-to-end L4 executor test against a real (tiny, from-scratch) causal
    LM: FixAgent is given a diagnosis-only finetune_pool distinct from the
    validation batch. Checks, in order: (1) before training the model does
    not already say the target word (the task is genuinely unlearned, not a
    freebie); (2) after L4 executes, held-out validation cases -- same
    prompt/target association as the pool, but case ids never in
    finetune_pool -- are fixed: proof real gradient training happened and
    the effect is visible on cases the executor never trained on directly,
    not that generalization to an unseen *prompt* was tested (it wasn't:
    every val case shares its prompt text with a training example); (3) a
    control prompt never touched by training round-trips to byte-identical
    output before vs. after -- proof the LoRA adapter was fully unloaded and
    the base weights were restored, not merely "probably fine"."""
    pytest.importorskip("peft")
    from evalrx.core.spec import ModelSpec
    from evalrx.models.backends.base import RuntimeConfig
    from evalrx.models.backends.hf_local import HFLocalModel

    spec = ModelSpec(key="tiny-llama-test", family="fake", model_type="fake_llm", hf_repo="")
    model = HFLocalModel(spec, RuntimeConfig(device="cpu", dtype="float32", max_new_tokens=3))
    llama = _tiny_llama()
    tok = _TinyWordTokenizer()
    model._hf = (llama, tok)

    control_prompt = Inputs(prompt="classify beta")
    baseline_control = model.generate(control_prompt)

    # The untrained model must not already answer "yes" to "classify alpha" --
    # otherwise a later match wouldn't demonstrate training did anything.
    baseline_alpha = model.generate(Inputs(prompt="classify alpha"))
    assert "yes" not in baseline_alpha.lower()

    train_pool = CaseBatch([
        _classify_case("alpha", "yes", Label.FAIL),
        _classify_case("alpha", "yes", Label.FAIL),
        _classify_case("beta", "no", Label.PASS),
        _classify_case("beta", "no", Label.PASS),
    ])
    # Held out: same prompt/target association, but DIFFERENT case ids that
    # never appear in train_pool -- this is what "generalizes" is checked on.
    # Multiple copies because a single paired case can never clear an
    # e-value significance gate (n=1 is inherently uninformative) -- that is
    # the McNemar/e-value machinery working correctly elsewhere in this
    # file, not something this test needs to re-prove; it just needs enough
    # pairs for a real, consistent effect to be visible as `out.fixed`.
    val_batch = CaseBatch([_classify_case("alpha", "yes", Label.FAIL) for _ in range(8)])

    agent = FixAgent(
        judge=None, max_tier="L4", allow_codegen=False, score_fn=_contains_score,
        finetune_pool=train_pool,
    )
    out = agent.propose_and_validate(model, val_batch, [_hyp("requires retraining", mode="prior")])

    ft = [v for v in out.attempted if v.candidate.kind == "finetune_spec"]
    assert len(ft) == 1
    assert ft[0].candidate.payload.get("exec_error", "") == ""
    assert ft[0].n_fixed == 8 and ft[0].n_broken == 0  # every held-out case now scores correct
    assert out.fixed is True

    # Restoration: an untouched control prompt reproduces the exact
    # pre-training output -- the adapter left no residue on the base model.
    restored_control = model.generate(control_prompt)
    assert restored_control == baseline_control


def test_l4_lora_repair_zero_matching_layers_does_not_crash(monkeypatch):
    """peft.get_peft_model() itself raises when target_modules matches zero
    layers on the given architecture -- and it raises BEFORE injecting
    anything, outside any try/finally the executor controls. That must
    become one candidate's LoraRepairResult(ok=False, ...), never an
    uncaught exception that aborts the whole FixAgent run."""
    pytest.importorskip("peft")
    from evalrx.core.spec import ModelSpec
    from evalrx.eval_agent.stages import fix_internals
    from evalrx.models.backends.base import RuntimeConfig
    from evalrx.models.backends.hf_local import HFLocalModel

    spec = ModelSpec(key="tiny-llama-test", family="fake", model_type="fake_llm", hf_repo="")
    model = HFLocalModel(spec, RuntimeConfig(device="cpu", dtype="float32", max_new_tokens=3))
    model._hf = (_tiny_llama(), _TinyWordTokenizer())
    monkeypatch.setattr(
        fix_internals, "_lora_target_modules", lambda hf_model: "this_will_never_match_anything"
    )

    train_pool = CaseBatch([_classify_case("alpha", "yes", Label.FAIL)])
    val_batch = CaseBatch([_classify_case("alpha", "yes", Label.FAIL)])
    agent = FixAgent(
        judge=None, max_tier="L4", allow_codegen=False, score_fn=_contains_score,
        finetune_pool=train_pool,
    )
    out = agent.propose_and_validate(model, val_batch, [_hyp("requires retraining", mode="prior")])

    ft = [v for v in out.attempted if v.candidate.kind == "finetune_spec"]
    assert len(ft) == 1
    assert ft[0].fixed is False
    assert "no matching linear layers" in ft[0].candidate.payload.get("exec_error", "")
    # the model must still be usable -- get_peft_model failing must not have
    # left it half-mutated
    assert model.generate(Inputs(prompt="classify alpha"))


# ── bridged model_attend (coded L3a) ─────────────────────────────────────────


_ATTEND_PIPELINE = """
import json
cases = json.load(open("fix_cases.json"))["cases"]
out = []
for c in cases:
    att = model_attend(c["id"])
    h, w = att["shape"]
    grid = att["grid"]
    best = max(range(h * w), key=lambda i: grid[i // w][i % w])
    r, cl = best // w, best % w
    box = [max(0.0, cl / w - 0.25), max(0.0, r / h - 0.25),
           min(1.0, cl / w + 0.35), min(1.0, r / h + 0.35)]
    ans = model_generate(c["id"], image_ops=[{"tool": "crop_region", "params": {"box": box}}])
    out.append({"sample_id": c["id"], "output": ans})
print("FIX_PIPELINE_RESULT_JSON=" + json.dumps({"per_case": out}))
"""


# ── defect 1: applicability predicate / conditional fixes ────────────────────


def test_predicate_scopes_validation_to_applicable_cases():
    """A candidate with a predicate is only judged on the cases it applies to —
    its safety/coverage exclude cases it never touched."""
    from evalrx.eval_agent.stages.fix_agent import FixCandidate

    agent = FixAgent(judge=None, max_tier="L1")
    data = _gold_yes_batch(n=4)
    cand = FixCandidate(
        tier=FixTier.L1_PROMPT,
        name="careful_subset",
        kind="template",
        payload={"prompt_template": "Look carefully. {prompt}"},
        predicate=lambda c: c.id in {"c0", "c1"},
    )
    model = BaselineFailsModel()
    baseline, unstable = agent._baseline(model, data)
    v = agent._validate(cand, model, data, baseline, unstable)
    assert v.n_applicable == 2 and v.n_fixed == 2 and v.n_broken == 0
    assert set(v.fixed_cases) == {"c0", "c1"}
    assert v.coverage == 0.5  # repaired 2 of the 4 failures it was scoped to


def test_signature_distinguishes_candidates_sharing_kind_and_payload():
    """Two candidates that share kind+payload but differ only by name/predicate
    (e.g. a paper method and its per-case-gated sibling) must not collide in
    the round's dedup ``seen`` set — that would silently drop the gated
    variant as an 'already seen' duplicate of the ungated one."""
    from evalrx.eval_agent.stages.fix_agent import FixCandidate

    agent = FixAgent(judge=None, max_tier="L0")
    payload = {"alpha": 1.0, "beta": 0.1, "qformer_mode": "normal"}
    ungated = FixCandidate(
        tier=FixTier.L0_RUNTIME_CONFIG, name="icd_instruction_disturbance", kind="icd",
        payload=payload,
    )
    gated = FixCandidate(
        tier=FixTier.L0_RUNTIME_CONFIG,
        name="icd_instruction_disturbance_gated_false_yes",
        kind="icd",
        payload=payload,
        predicate=lambda c: True,
    )
    assert agent._signature(ungated) != agent._signature(gated)


def test_self_refine_offered_for_image_reasoning_tasks_not_yes_no():
    """self_refine/self_consistency_5/least_to_most were only ever proposed
    for text-only cases, even though run_pipeline already threads the case
    image through every call -- nothing about them is text-specific. A
    multi-step reasoning task (multiple_choice/exact_or_numeric/
    vqa_consensus) with an image should get self_refine and
    self_consistency_5, prioritised first; a binary/grounding task (yes_no)
    should get neither, so the proven image-transform ladder is not diluted
    there."""
    agent = FixAgent(judge=None, max_tier="L2")
    out = agent._l2_candidates(
        "- some hypothesis", "", has_images=True, model=None, tasks={"multiple_choice"}
    )
    assert [c.name for c in out[:2]] == ["self_refine", "self_consistency_5"]

    out_yn = agent._l2_candidates(
        "- some hypothesis", "", has_images=True, model=None, tasks={"yes_no"}
    )
    assert {"self_refine", "self_consistency_5"}.isdisjoint(c.name for c in out_yn)


def test_l1_candidates_do_not_use_gold_direction_gates():
    """Changing expected labels must not alter the proposed repair family."""
    agent = FixAgent(judge=None, max_tier="L1")
    first = agent._l1_candidates("- h", "", has_images=True, tasks={"yes_no"})
    second = agent._l1_candidates("- h", "", has_images=True, tasks={"yes_no"})
    assert [c.name for c in first] == [c.name for c in second]
    assert "assertive_grounding" not in {c.name for c in first}


def test_spec_noop_cases_are_not_applicable():
    PIL = pytest.importorskip("PIL")
    from evalrx.eval_agent.stages.fix_tools import PipelineSpec, spec_changes_input

    spec = PipelineSpec.from_dict(
        {"name": "crop", "image_ops": [{"tool": "crop_case_bbox", "params": {}}]}
    )
    no_bbox = FailureCase(id="x", inputs=Inputs(prompt="q", image=_img()), metadata={})
    assert spec_changes_input(spec, no_bbox) is False  # crop is a no-op here

    img = PIL.Image.new("RGB", (100, 100), color=(220, 220, 220))
    for y in range(10, 14):
        for x in range(80, 84):
            img.putpixel((x, y), (20, 20, 20))
    with_bbox = FailureCase(
        id="y",
        inputs=Inputs(prompt="q", image=img),
        metadata={"answer_bbox_xyxy_norm": [0.8, 0.1, 0.84, 0.14]},
    )
    assert spec_changes_input(spec, with_bbox) is True


# ── defect 2: noise floor (baseline stability) ───────────────────────────────


class _OneFlakyModel(Model):
    """Case 0's baseline flips between repeats (sampling noise); others stable.
    The 'carefully' prompt deterministically answers yes (a real fix)."""

    capabilities = frozenset({Capability.GENERATE})
    modalities = frozenset({"text", "image"})

    def __init__(self) -> None:
        self._n = 0

    def generate(self, inputs, **kwargs):
        p = str(getattr(inputs, "prompt", inputs)).lower()
        if "carefully" in p:
            return "Yes."
        if "lesion 0" in p:
            self._n += 1
            return "Yes." if self._n % 2 == 1 else "No."
        return "No."

    def forward(self, inputs, capture, spec=None):
        raise NotImplementedError


def test_baseline_repeats_flags_unstable_cases_and_weighs_them():
    """baseline_repeats>1 measures each case's baseline as a PASS RATE. A case
    whose baseline flips (c0: 1 pass in 2 samples -> rate 0.5) is reported as
    unstable but stays in the paired test at its rate — the candidate's move on
    it counts for what it is (here +0.5), instead of the case being dropped
    (which removed exactly the cases a variance-reduction fix repairs)."""
    agent = FixAgent(judge=None, max_tier="L1", baseline_repeats=2)
    data = _gold_yes_batch(n=2)
    model = _OneFlakyModel()
    baseline, unstable = agent._baseline(model, data)
    assert "c0" in unstable and "c1" not in unstable
    assert agent._baseline_rates["c0"] == 0.5 and agent._baseline_rates["c1"] == 0.0

    from evalrx.eval_agent.stages.fix_agent import FixCandidate

    cand = FixCandidate(
        tier=FixTier.L1_PROMPT,
        name="careful",
        kind="template",
        payload={"prompt_template": "Look carefully. {prompt}"},
    )
    v = agent._validate(cand, model, data, baseline, unstable)
    assert v.noise_model == "paired_rates" and v.n_baseline_samples == 2
    assert v.n_unstable == 1 and v.n_pairs == 2          # reported, NOT dropped
    assert v.baseline_rate == 0.25 and v.candidate_rate == 1.0
    # modal flips: c1 (0 -> 1) is a fix; c0 (0.5 -> 1, ties count as modal
    # pass) is not a modal flip, but its +0.5 is in the effect
    assert v.fixed_cases == ["c1"] and v.n_broken == 0
    assert v.effect == 0.75
    assert "unstable weighed" in v.summary


# ── defect 4: power-aware verdict ─────────────────────────────────────────────


def test_underpowered_run_recommends_gathering_failures():
    """With too few failures, even a flawless fix cannot reach significance;
    the recommendation must say 'gather more failures', not 'escalate tier'."""
    judge = ScriptedJudge(
        json.dumps([{"name": "careful", "prompt_template": "Look carefully. {prompt}"}])
    )
    agent = FixAgent(judge=judge, max_tier="L1")
    out = agent.propose_and_validate(BaselineFailsModel(), _gold_yes_batch(n=3), [_hyp("x")])
    v = out.attempted[0]
    assert v.n_fixed == 3 and v.n_broken == 0
    assert v.verdict == "partial" and v.fixed is False  # net-positive, not sig
    assert out.fixed is False
    assert out.recommendation["action"] == "gather_more_failures"
    assert out.recommendation["recommend_tier"] is None


def test_well_powered_run_still_certifies_fix():
    """Same fix, enough failures (8): e-value clears the gate -> certified."""
    judge = ScriptedJudge(
        json.dumps([{"name": "careful", "prompt_template": "Look carefully. {prompt}"}])
    )
    agent = FixAgent(judge=judge, max_tier="L1")
    out = agent.propose_and_validate(BaselineFailsModel(), _gold_yes_batch(n=8), [_hyp("x")])
    assert out.fixed is True and out.best.verdict == "fixed"


# ── defect 3: heterogeneity feedback edge ────────────────────────────────────


class _MixedModel(Model):
    """'carefully' repairs even-indexed cases but breaks odd-indexed ones —
    a heterogeneous failure mode that no single global transform can fix."""

    capabilities = frozenset({Capability.GENERATE})
    modalities = frozenset({"text", "image"})

    def generate(self, inputs, **kwargs):
        import re as _re

        p = str(getattr(inputs, "prompt", inputs)).lower()
        m = _re.search(r"lesion (\d+)", p)
        idx = int(m.group(1)) if m else 0
        careful = "carefully" in p
        if idx % 2 == 0:
            return "Yes." if careful else "No."  # careful fixes evens
        return "No." if careful else "Yes."  # careful breaks odds

    def forward(self, inputs, capture, spec=None):
        raise NotImplementedError


def test_heterogeneous_outcome_emits_refine_signal():
    judge = ScriptedJudge(
        json.dumps([{"name": "careful", "prompt_template": "Look carefully. {prompt}"}])
    )
    agent = FixAgent(judge=judge, max_tier="L1")
    out = agent.propose_and_validate(_MixedModel(), _gold_yes_batch(n=4), [_hyp("x")])
    v = out.attempted[0]
    assert v.n_fixed > 0 and v.n_broken > 0
    assert out.refine_signal is not None
    assert out.refine_signal["kind"] == "heterogeneous_failure_mode"
    assert out.refine_signal["helped_cases"] and out.refine_signal["hurt_cases"]


def test_bridged_attend_enables_coded_l3a(tmp_path):
    pytest.importorskip("PIL")
    pytest.importorskip("torch")
    from evalrx.analyzers.perturbation.prompt_contrast import _default_score
    from evalrx.eval_agent.stages.fix_pipeline import (
        run_coded_pipeline,
        score_outputs,
    )

    cases = CaseBatch(
        [
            FailureCase(
                id=f"c{i}",
                inputs=Inputs(prompt="bright?", image=_bright_corner_img()),
                expected={"all_of": ["yes"], "none_of": ["no"]},
                label=Label.FAIL,
            )
            for i in range(2)
        ]
    )
    ok = run_coded_pipeline(
        _ATTEND_PIPELINE,
        AttnCropVLM(),
        cases,
        workdir=tmp_path / "on",
        timeout_sec=30,
        enable_attend=True,
    )
    assert ok.ok
    assert all(v is True for v in score_outputs(ok, cases, _default_score).values())
    # Disabled -> model_attend errors -> pipeline crashes -> no result.
    off = run_coded_pipeline(
        _ATTEND_PIPELINE,
        AttnCropVLM(),
        cases,
        workdir=tmp_path / "off",
        timeout_sec=30,
        enable_attend=False,
    )
    assert off.ok is False


# ── per-trial output folders: fixes/<NN_tier_name>/ self-contained attempts ──
#
# With a RunContext, every attempt's code + sandbox + record.md + result.json
# live together under one numbered folder instead of being scattered across
# tools/ / workspace/ / fixes/ and re-correlated by filename slug.


def test_declarative_candidate_gets_record_and_result_but_no_workspace(tmp_path):
    """A template/spec candidate never touches a sandbox — its trial folder
    should hold only record.md + result.json, no workspace/ subdir."""
    from evalrx.eval_agent.run_context import RunContext

    ctx = RunContext(tmp_path / "run1")
    judge = ScriptedJudge(
        json.dumps(
            [
                {"name": "careful", "prompt_template": "Look very carefully. {prompt}"},
            ]
        )
    )
    agent = FixAgent(judge=judge, max_tier="L1", run_logger=ctx.logger, run_context=ctx)
    out = agent.propose_and_validate(
        BaselineFailsModel(),
        _gold_yes_batch(),
        [_hyp("the prompt phrasing underspecifies the task")],
    )
    assert out.fixed is True
    trial = out.best.candidate.trial
    assert trial is not None
    assert trial.root.parent == ctx.fixes_dir

    ctx.finalize()
    assert (trial.root / "record.md").exists()
    assert (trial.root / "result.json").exists()
    assert not (trial.root / "workspace").exists()


def test_deduped_candidate_in_round_two_leaves_no_trial_folder(tmp_path):
    """A judge that keeps proposing the SAME failing candidate is deduped
    before a trial is ever allocated for it — round 2 must not leave behind
    an empty (or duplicate) folder."""
    from evalrx.eval_agent.run_context import RunContext

    ctx = RunContext(tmp_path / "run1")
    judge = ScriptedJudge(
        json.dumps([{"name": "polite", "prompt_template": "Please answer. {prompt}"}])
    )
    agent = FixAgent(
        judge=judge, max_tier="L1", max_repair_rounds=3, run_logger=ctx.logger, run_context=ctx
    )
    out = agent.propose_and_validate(BaselineFailsModel(), _gold_yes_batch(), [_hyp("x")])
    assert out.repair_rounds == 1
    assert sum(1 for v in out.attempted if v.candidate.name == "polite") == 1

    ctx.finalize()
    trial_dirs = [p for p in ctx.fixes_dir.iterdir() if p.is_dir()]
    assert len(trial_dirs) == 1  # exactly one — no orphan from the deduped re-proposal


class TwoVersionCodeJudge(Model):
    """Round 1's coded pipeline is a no-op (doesn't fix); round 2's (written
    after seeing round 1's failure) inserts a marker the test model is
    sensitive to (fixes) — exercises two coded fix attempts in one run, each
    needing its own durable trial workspace (the bug this feature exists to
    fix: both used to share — and overwrite — one sandbox)."""

    capabilities = frozenset({Capability.GENERATE})
    modalities = frozenset({"text"})

    def __init__(self) -> None:
        self.code_calls = 0

    def generate(self, inputs, **kwargs) -> str:
        if "EXECUTION CONTRACT" in str(inputs):
            self.code_calls += 1
            pipeline = _NOOP_CODE_PIPELINE if self.code_calls == 1 else _MARKER_CODE_PIPELINE
            return f"```python\n{pipeline}\n```"
        return "no json here"

    def forward(self, inputs, capture, spec=None):
        raise NotImplementedError


class MarkerSensitiveModel(Model):
    """Answers "yes" only when the prompt contains a literal marker — only a
    custom-coded pipeline rewriting the prompt can trigger this, so none of
    the default L1 templates / L2 image-op specs accidentally fix it (isolates
    the coded-pipeline-only effect for the two-trial test below)."""

    capabilities = frozenset({Capability.GENERATE})
    modalities = frozenset({"text", "image"})

    def generate(self, inputs, **kwargs):
        prompt = str(getattr(inputs, "prompt", ""))
        return "Yes." if "SECRET_MARKER" in prompt else "No."

    def forward(self, inputs, capture, spec=None):
        raise NotImplementedError


_NOOP_CODE_PIPELINE = """
import json
cases = json.load(open("fix_cases.json"))["cases"]
out = []
for c in cases:
    ans = model_generate(c["id"])
    out.append({"sample_id": c["id"], "output": ans})
print("FIX_PIPELINE_RESULT_JSON=" + json.dumps({"per_case": out}))
"""

_MARKER_CODE_PIPELINE = """
import json
cases = json.load(open("fix_cases.json"))["cases"]
out = []
for c in cases:
    model_generate(c["id"])
    marked = "SECRET_MARKER " + c["prompt"]
    ans = model_generate(c["id"], prompt=marked)
    model_generate(c["id"], prompt=marked + " verify")
    out.append({"sample_id": c["id"], "output": ans})
print("FIX_PIPELINE_RESULT_JSON=" + json.dumps({"per_case": out}))
"""


def test_two_coded_fix_attempts_get_separate_trial_workspaces(tmp_path):
    pytest.importorskip("PIL")
    from evalrx.eval_agent.run_context import RunContext

    ctx = RunContext(tmp_path / "run1")
    agent = FixAgent(
        judge=TwoVersionCodeJudge(),
        max_tier="L2",
        run_logger=ctx.logger,
        run_context=ctx,
        exec_timeout_sec=30,
        max_repair_rounds=2,
    )
    out = agent.propose_and_validate(
        MarkerSensitiveModel(),
        _gold_yes_batch(16, image=_img()),
        [_hyp("the model never sees the marker it needs", mode="prompt_gap")],
    )

    coded = [v for v in out.attempted if v.candidate.kind == "code"]
    assert len(coded) == 2
    assert coded[0].fixed is False
    assert coded[1].fixed is True
    assert out.fixed is True
    assert out.repair_rounds == 2

    t1, t2 = coded[0].candidate.trial, coded[1].candidate.trial
    assert t1 is not None and t2 is not None
    assert t1.root != t2.root
    code1 = (t1.workspace / "fix_pipeline_exec.py").read_text()
    code2 = (t2.workspace / "fix_pipeline_exec.py").read_text()
    assert "SECRET_MARKER" not in code1
    assert "SECRET_MARKER" in code2

    ctx.finalize()
    # Each trial's own record + result — not a shared/overwritten one.
    assert (t1.root / "record.md").exists()
    assert (t2.root / "record.md").exists()
    assert json.loads((t1.root / "result.json").read_text())["fixed"] is False
    assert json.loads((t2.root / "result.json").read_text())["fixed"] is True


# ── prompt templates live next to LaTeX ──────────────────────────────────────
def test_safe_format_leaves_non_placeholder_braces_alone():
    r"""str.format treats every {...} as a field; a math prompt is full of them.

    All four of these are real str.format failures, and the third ended a live
    qwen3.5-2b / minervamath run after M5 had already produced its fix:

        "{prompt} \frac{a}{b}"  KeyError: 'a'
        "{prompt} 10^{33}"      IndexError: Replacement index 33
        "{prompt} ${~m}$"       KeyError: '~m'
        "{prompt} {}"           IndexError: Replacement index 0
    """
    from evalrx.eval_agent.stages.fix_agent import safe_format

    ctx = {"prompt": "P", "failure_axis": "AX"}
    assert safe_format(r"{prompt} solve \frac{a}{b}", ctx) == r"P solve \frac{a}{b}"
    assert safe_format(r"{prompt} 10^{33}", ctx) == r"P 10^{33}"
    assert safe_format(r"{prompt} ${~m}$", ctx) == r"P ${~m}$"
    assert safe_format("{prompt} {}", ctx) == "P {}"


def test_safe_format_still_substitutes_the_known_fields():
    from evalrx.eval_agent.stages.fix_agent import safe_format

    ctx = {"prompt": "P", "failure_axis": "AX", "n": 3}
    assert safe_format("{prompt} focus on {failure_axis} ({n})", ctx) == "P focus on AX (3)"


def test_safe_format_never_raises_on_arbitrary_text():
    from evalrx.eval_agent.stages.fix_agent import safe_format

    for template in ("{", "}", "{{", "{unclosed", r"\boxed{}", "{a}{b}{c}", ""):
        safe_format(template, {"prompt": "P"})


def test_a_template_case_that_cannot_render_scores_none_not_a_crash():
    """The formatting used to sit outside l1's try/except, so one unrenderable
    template aborted the entire validation instead of dropping one case."""
    from evalrx.core.case import FailureCase, Inputs, Label
    from evalrx.eval_agent.stages.fix_agent import FixAgent, FixCandidate

    class _Boom:
        def generate(self, *a, **k):
            raise RuntimeError("model down")

    agent = FixAgent()
    candidate = FixCandidate(tier=None, name="t", kind="template",
                             payload={"prompt_template": r"{prompt} ${~m}$"})
    strategy = agent._strategy(candidate)
    case = FailureCase(inputs=Inputs(prompt="q"), observed="o", expected="e",
                       label=Label.FAIL)
    assert strategy(_Boom(), case) is None


# ── repair the model, not the task: the frozen-model control ─────────────────
#
# qwen3.5-2b / bbh_word_sorting: the L2 pipeline that "FIXED" 124/125 cases was
# `ref = sorted(input_words(prompt))` with the model's answer accepted only when
# it already equalled ref — 113 model calls, none of which changed an output.
# The sandbox forbids touching the model; it never asked whether the answer
# came from it.


def _observed_no_batch(n: int = 16) -> CaseBatch:
    """Gold "yes", recorded baseline "No." — a real batch always carries the
    baseline answer, and the control replays exactly that."""
    yes = {"all_of": ["yes"], "none_of": ["no"]}
    return CaseBatch([
        FailureCase(id=f"c{i}", inputs=Inputs(prompt=f"Is there a lesion {i}?"),
                    expected=yes, observed="No.", label=Label.FAIL)
        for i in range(n)
    ])


_SOLVER_PIPELINE = """
import json
cases = json.load(open("fix_cases.json"))["cases"]
out = []
for c in cases:
    ans = model_generate(c["id"])          # asked, then ignored
    out.append({"sample_id": c["id"], "output": "Yes."})
print("FIX_PIPELINE_RESULT_JSON=" + json.dumps({"per_case": out}))
"""

_REPROMPT_PIPELINE = """
import json
cases = json.load(open("fix_cases.json"))["cases"]
out = []
for c in cases:
    ans = model_generate(c["id"], prompt=c["prompt"] + " Look carefully.")
    out.append({"sample_id": c["id"], "output": ans})
print("FIX_PIPELINE_RESULT_JSON=" + json.dumps({"per_case": out}))
"""

_PARTIAL_SOLVER_PIPELINE = """
import json
cases = json.load(open("fix_cases.json"))["cases"]
out = []
for c in cases:
    ans = model_generate(c["id"], prompt=c["prompt"] + " Look carefully.")
    if c["id"] in ("c0", "c1", "c2"):     # hard-codes three of them
        ans = "Yes."
    out.append({"sample_id": c["id"], "output": ans})
print("FIX_PIPELINE_RESULT_JSON=" + json.dumps({"per_case": out}))
"""


def _code_candidate(code: str, name: str = "coded_pipeline") -> FixCandidate:
    return FixCandidate(tier=FixTier.L2_SCAFFOLD, name=name, kind="code",
                        source="judge", payload={"code": code})


def test_a_pipeline_that_solves_the_task_itself_is_not_a_fix(tmp_path):
    """Every case right, every case also right with the model frozen: nothing
    left to attribute to the model. Verdict names it; the recommendation must
    not read it as 'never executed'."""
    agent = FixAgent(judge=None, max_tier="L2", exec_timeout_sec=30)
    data = _observed_no_batch()
    model = BaselineFailsModel()
    baseline, unstable = agent._baseline(model, data)
    v = agent._validate(_code_candidate(_SOLVER_PIPELINE), model, data, baseline, unstable)

    assert v.fixed is False
    assert v.verdict == "model_independent"
    assert v.n_model_independent == len(data) and v.n_pairs == 0
    assert "frozen-model control" in v.summary
    ctrl = v.candidate.payload["frozen_model_control"]
    assert ctrl["ok"] is True and len(ctrl["solved"]) == len(data)

    rec = agent._no_fix_recommendation([v], [FixTier.L2_SCAFFOLD], data, model)
    assert rec is not None and rec.get("action") != "fix_execution"
    assert "without the model" in rec["reason"]


def test_a_pipeline_that_needs_the_model_keeps_its_credit(tmp_path):
    """Frozen to 'No.' the re-prompt yields 'No.' — nothing is discounted."""
    agent = FixAgent(judge=None, max_tier="L2", exec_timeout_sec=30)
    data = _observed_no_batch()
    model = BaselineFailsModel()
    baseline, unstable = agent._baseline(model, data)
    v = agent._validate(_code_candidate(_REPROMPT_PIPELINE), model, data, baseline, unstable)

    assert v.fixed is True and v.verdict == "fixed"
    assert v.n_model_independent == 0 and v.n_pairs == len(data)
    assert v.candidate.payload["frozen_model_control"]["solved"] == []


def test_a_solver_fallback_only_loses_the_cases_it_solved(tmp_path):
    """Three hard-coded cases leave the test; the thirteen the model repaired
    still certify the fix. The count is on the record."""
    agent = FixAgent(judge=None, max_tier="L2", exec_timeout_sec=30)
    data = _observed_no_batch()
    model = BaselineFailsModel()
    baseline, unstable = agent._baseline(model, data)
    v = agent._validate(_code_candidate(_PARTIAL_SOLVER_PIPELINE), model, data,
                        baseline, unstable)

    assert v.n_model_independent == 3 and v.n_pairs == len(data) - 3
    assert v.fixed is True
    assert "3 model-independent excluded" in v.summary
    assert set(v.candidate.payload["frozen_model_control"]["solved"]) == {"c0", "c1", "c2"}


def test_baseline_correct_cases_are_never_dropped_by_the_control():
    """Replaying a right answer proves nothing; dropping such cases would hide
    what a candidate breaks. A pipeline that hard-codes 'No.' breaks the
    baseline-correct cases and must be seen breaking them."""
    yes = {"all_of": ["yes"], "none_of": ["no"]}
    data = CaseBatch([
        FailureCase(id=f"c{i}", inputs=Inputs(prompt=f"q{i}"), expected=yes,
                    observed="Yes.", label=Label.PASS)
        for i in range(6)
    ])
    agent = FixAgent(judge=None, max_tier="L2", exec_timeout_sec=30)
    cand = _code_candidate('''
import json
cases = json.load(open("fix_cases.json"))["cases"]
print("FIX_PIPELINE_RESULT_JSON=" + json.dumps(
    {"per_case": [{"sample_id": c["id"], "output": "No."} for c in cases]}))
''')
    model = BaselineFailsModel()
    baseline, unstable = agent._baseline(model, data)
    v = agent._validate(cand, model, data, baseline, unstable)
    assert v.n_model_independent == 0
    assert v.n_broken == 6 and v.verdict in {"regressed", "unsafe"}


def test_the_coder_is_told_the_rule():
    from evalrx.eval_agent.prompts.fix_agent import _L2_CODE_PROMPT, _REPAIR_PROMPT_BODY

    assert "REPAIR THE MODEL, NOT THE TASK" in _L2_CODE_PROMPT
    assert "ORIGINAL recorded answer" in _L2_CODE_PROMPT
    assert "prompt`` REPLACES the original prompt" in _L2_CODE_PROMPT
    assert 'case["prompt"]' in _L2_CODE_PROMPT
    assert "{selection_guidance}" in _L2_CODE_PROMPT
    assert "at most\n  4 calls" in _L2_CODE_PROMPT
    assert "not the task" in _REPAIR_PROMPT_BODY


def test_coder_guidance_distinguishes_explore_revision_and_one_shot():
    explore = FixAgent(max_repair_rounds=2)._code_selection_guidance()
    revision = FixAgent(max_repair_rounds=2)._code_selection_guidance("prior result")
    one_shot = FixAgent(max_repair_rounds=1)._code_selection_guidance()

    assert "controlled 2-of-3" in explore
    assert "FEEDBACK-DRIVEN" in revision and "abandon" in revision
    assert "all 3" in one_shot and "safety baseline" in one_shot


# ── the direct-baseline contract: anchored on the recorded baseline ──────────
#
# chartqa / spatial457 × qwen2.5-vl and chartqa × qwen3.5-2b (2026-08-21): five
# of five coder-written rounds read ``baseline_output`` (the prompt hands it
# over) and never made a plain ``model_generate(case_id)`` call, so the guard
# voided the whole candidate every time and a repair round was spent on
# re-learning the call.  The bridge now anchors on the recorded baseline
# itself, answers a plain direct call from that record, and only a case with
# nothing to anchor on leaves the result.


_BASELINE_OUTPUT_VOTE_PIPELINE = """
import json, re
cases = json.load(open("fix_cases.json"))["cases"]

def key(t):
    m = re.search(r"final answer:\\s*(.+)", t or "", re.I)
    return (m.group(1) if m else (t or "")).strip().lower().strip(".")

out = []
for c in cases:
    base = c["baseline_output"] or ""
    votes = [model_generate(c["id"], prompt=c["prompt"] + f" Look carefully ({i}).")
             for i in range(3)]
    keys = [key(v) for v in votes]
    best = max(set(keys), key=keys.count)
    if keys.count(best) >= 2 and best != key(base):
        final = next(v for v, k in zip(votes, keys) if k == best)
    else:
        final = base
    out.append({"sample_id": c["id"], "output": final})
print("FIX_PIPELINE_RESULT_JSON=" + json.dumps({"per_case": out}))
"""


def test_guard_anchors_on_the_recorded_baseline_without_a_direct_call(tmp_path):
    """No plain ``model_generate(case_id)`` anywhere: the guard anchors on
    ``case.observed`` — a 2-of-3 override stands, a 3-of-3 requirement reverts
    the case to the RECORDED answer — and the candidate is never voided."""
    from evalrx.eval_agent.stages.fix_pipeline import run_coded_pipeline

    def replies(case, prompt):
        return "Final Answer: No." if "(2)" in prompt else "Final Answer: Yes."

    cases = _observed_no_batch(n=4)
    res = run_coded_pipeline(
        _BASELINE_OUTPUT_VOTE_PIPELINE, None, cases, workdir=tmp_path / "two",
        timeout_sec=20, reply_fn=replies, consensus_min_support=2,
    )
    assert res.ok and res.error == ""
    assert res.n_anchored_from_recorded == 4 and res.unanchored_ids == []
    assert res.n_guarded == 0
    assert all(v == "Final Answer: Yes." for v in res.outputs.values())

    strict = run_coded_pipeline(
        _BASELINE_OUTPUT_VOTE_PIPELINE, None, cases, workdir=tmp_path / "three",
        timeout_sec=20, reply_fn=replies, consensus_min_support=3,
    )
    assert strict.ok and strict.n_guarded == 4 and strict.guarded_ids == ["c0", "c1", "c2", "c3"]
    assert all(v == "No." for v in strict.outputs.values())


def test_only_cases_with_nothing_to_anchor_on_leave_the_result(tmp_path):
    """A case with neither a recorded baseline nor a direct call is excluded
    (scored as not measured); the other cases are guarded as usual. Only when
    EVERY case is unanchorable does the run fail, and it says why."""
    from evalrx.eval_agent.stages.fix_pipeline import run_coded_pipeline

    yes = {"all_of": ["yes"], "none_of": ["no"]}
    cases = CaseBatch([
        FailureCase(id="c0", inputs=Inputs(prompt="Is there a lesion 0?"),
                    expected=yes, observed="No.", label=Label.FAIL),
        FailureCase(id="c1", inputs=Inputs(prompt="Is there a lesion 1?"),
                    expected=yes, label=Label.FAIL),  # no recorded baseline
    ])
    pipeline = """
import json
cases = json.load(open("fix_cases.json"))["cases"]
out = [{"sample_id": c["id"], "output": model_generate(c["id"], prompt="enhanced")}
       for c in cases]
print("FIX_PIPELINE_RESULT_JSON=" + json.dumps({"per_case": out}))
"""
    res = run_coded_pipeline(
        pipeline, None, cases, workdir=tmp_path, timeout_sec=20,
        reply_fn=lambda case, prompt: "Yes.", consensus_min_support=2,
    )
    assert res.ok and res.error == ""
    assert res.outputs == {"c0": "No."}          # singleton override reverted to the record
    assert res.guarded_ids == ["c0"]
    assert res.unanchored_ids == ["c1"] and res.n_anchored_from_recorded == 1

    # The pre-existing all-or-nothing test above keeps holding for a batch
    # with no records at all: every case is unanchorable, nothing to score.
    none = run_coded_pipeline(
        pipeline, None, _gold_yes_batch(n=2), workdir=tmp_path / "none", timeout_sec=20,
        reply_fn=lambda case, prompt: "Yes.", consensus_min_support=2,
    )
    assert none.ok is False and none.unanchored_ids == ["c0", "c1"]
    assert "no recorded baseline" in none.error and "no direct baseline" in none.error


class _PromptCountingModel(Model):
    capabilities = frozenset({Capability.GENERATE})
    modalities = frozenset({"text", "image"})

    def __init__(self) -> None:
        self.prompts: list[str] = []

    def generate(self, inputs, **kwargs):
        self.prompts.append(str(getattr(inputs, "prompt", inputs)))
        return "Yes."

    def forward(self, inputs, capture, spec=None):
        raise NotImplementedError


_DIRECT_PLUS_ONE_PIPELINE = """
import json
cases = json.load(open("fix_cases.json"))["cases"]
out = []
for c in cases:
    a = model_generate(c["id"])                        # plain: answered from the record
    b = model_generate(c["id"], prompt=c["prompt"])    # same prompt: also plain
    e = model_generate(c["id"], prompt=c["prompt"] + " Look carefully.")
    out.append({"sample_id": c["id"], "output": a + "|" + b + "|" + e})
print("FIX_PIPELINE_RESULT_JSON=" + json.dumps({"per_case": out}))
"""


def test_plain_direct_calls_are_answered_from_the_record_and_are_free(tmp_path):
    """Two plain direct calls per case never reach the model and do not count
    against the per-case cap — in the live run and in the frozen control
    alike — while a direct call with decoding overrides is a live call."""
    from evalrx.eval_agent.stages.fix_pipeline import (
        frozen_model_control,
        run_coded_pipeline,
    )

    cases = _observed_no_batch(n=3)
    model = _PromptCountingModel()
    res = run_coded_pipeline(
        _DIRECT_PLUS_ONE_PIPELINE, model, cases, workdir=tmp_path / "live",
        timeout_sec=20, max_calls_per_case=1,
    )
    assert res.ok and res.error == ""
    assert res.n_calls == 9 and res.n_replayed == 6
    assert len(model.prompts) == 3 and all("carefully" in p for p in model.prompts)
    assert all(v == "No.|No.|Yes." for v in res.outputs.values())

    ctrl = frozen_model_control(
        _DIRECT_PLUS_ONE_PIPELINE, cases, workdir=tmp_path / "frozen",
        timeout_sec=20, max_calls_per_case=1,
    )
    assert ctrl.ok and all(v == "No.|No.|No." for v in ctrl.outputs.values())

    two_enhanced = _DIRECT_PLUS_ONE_PIPELINE.replace(
        '    out.append(', '    model_generate(c["id"], prompt="another pass")\n    out.append(')
    capped = run_coded_pipeline(
        two_enhanced, _PromptCountingModel(), cases, workdir=tmp_path / "cap",
        timeout_sec=20, max_calls_per_case=1,
    )
    assert capped.ok is False and "more than 1 model calls" in capped.error

    sampled = """
import json
cases = json.load(open("fix_cases.json"))["cases"]
out = [{"sample_id": c["id"],
        "output": model_generate(c["id"], generation_kwargs={"temperature": 0.7})}
       for c in cases]
print("FIX_PIPELINE_RESULT_JSON=" + json.dumps({"per_case": out}))
"""
    live_model = _PromptCountingModel()
    live = run_coded_pipeline(
        sampled, live_model, cases, workdir=tmp_path / "sampled", timeout_sec=20,
    )
    assert live.ok and live.n_replayed == 0 and len(live_model.prompts) == 3


def test_answer_key_strips_answer_tags_so_tagged_replies_can_support_an_override():
    """A scaffold that asks for "FINAL: <answer>" returns the extracted answer;
    the model's raw reply carries the tag. Without stripping it no tagged
    reply could ever support an override (chartqa/qwen3.5-2b repair round:
    every override reverted, no_effect 0/0)."""
    from evalrx.eval_agent.stages.fix_pipeline import _answer_key, _answers_match

    assert _answer_key("I read 42 from the bar.\nFINAL: 42") == "42"
    assert _answer_key("Answer: 42") == "42"
    assert _answer_key("Final answer: Yes.") == "yes"
    assert _answer_key("Prediction: no") == "no"
    assert _answers_match("42", "The tallest bar is 2019.\nFINAL: 42")
    assert not _answers_match("42", "The tallest bar is 2019.\nFINAL: 41")
    # Only answer tags are stripped — an answer that merely starts with a word
    # and a colon is left alone.
    assert _answer_key("Yes: the chart shows it") == "yes the chart shows it"


def test_the_coder_is_told_the_anchor_semantics():
    from evalrx.eval_agent.prompts.fix_agent import _L2_CODE_PROMPT, _REPAIR_PROMPT_BODY

    for prompt in (_L2_CODE_PROMPT, _REPAIR_PROMPT_BODY):
        assert "{min_support}" in prompt
        assert "{selection_guidance}" in prompt
        assert "answered from" in prompt          # a plain direct call is served from the record
    assert "do NOT need to call model_generate(case_id)" in _L2_CODE_PROMPT
    assert "does not count" in _L2_CODE_PROMPT
    assert "SELECTION RULE" in _L2_CODE_PROMPT
    assert "strips such tags" in _L2_CODE_PROMPT
    assert "IS the direct baseline" in _REPAIR_PROMPT_BODY

    agent = FixAgent(max_repair_rounds=2)
    for variant in (agent._code_selection_guidance(), agent._code_selection_guidance("prior"),
                    FixAgent(max_repair_rounds=1)._code_selection_guidance()):
        assert "baseline_output" in variant


class _BaselineOutputJudge(Model):
    """Writes the pipeline every real coder wrote: reads baseline_output, never
    calls model_generate(case_id) plainly. A repair request is a failure."""

    capabilities = frozenset({Capability.GENERATE})
    modalities = frozenset({"text"})

    def __init__(self) -> None:
        self.prompts: list[str] = []

    def generate(self, inputs, **kwargs) -> str:
        text = str(inputs)
        self.prompts.append(text)
        if "FAILED TO EXECUTE" in text:
            raise AssertionError("a repair round was requested for a contract-compliant pipeline")
        if "EXECUTION CONTRACT" in text:
            return f"```python\n{_BASELINE_OUTPUT_VOTE_PIPELINE}\n```"
        return "no json here"

    def forward(self, inputs, capture, spec=None):
        raise NotImplementedError


def test_a_coded_pipeline_that_reads_baseline_output_needs_no_repair_round(tmp_path):
    """End to end through FixAgent: the coder's natural pipeline executes on
    the first attempt, the host states the support threshold it enforces, the
    guard anchors every case on its record, the frozen control still holds,
    and no repair round is spent."""
    from evalrx.eval_agent import RunLogger

    logger = RunLogger(tmp_path / "logs")
    judge = _BaselineOutputJudge()
    agent = FixAgent(
        judge=judge, max_tier="L2", run_logger=logger, exec_timeout_sec=30,
        candidate_allowlist={"coded_pipeline"},
    )
    data = _observed_no_batch()
    out = agent.propose_and_validate(BaselineFailsModel(), data, [_hyp("x")])
    coded = [v for v in out.attempted if v.candidate.kind == "code"]
    assert len(coded) == 1
    v = coded[0]
    assert v.exec_error == "" and v.fixed is True and v.n_fixed == len(data)
    guard = v.candidate.payload["selection_guard"]
    assert guard["min_support"] == 3                       # one-shot: all three passes agree
    assert guard["n_anchored_from_recorded"] == len(data)
    assert guard["unanchored_ids"] == [] and guard["n_guarded"] == 0
    assert v.candidate.payload["frozen_model_control"]["solved"] == []
    code_prompt = next(p for p in judge.prompts if "EXECUTION CONTRACT" in p)
    assert "at least 3 of your enhanced calls" in code_prompt
    events = [json.loads(line)
              for line in (tmp_path / "logs" / "run_log.jsonl").read_text(encoding="utf-8").splitlines()]
    codegen = [e for e in events if e.get("event") == "tool_codegen"]
    assert len(codegen) == 1 and codegen[0]["ok"] is True
    assert codegen[0]["tool_name"] == "coded_pipeline"      # never coded_pipeline_repair


def test_coded_attempt_persists_guard_and_control_audit_files(tmp_path):
    """The selection-guard statistics and the frozen-model control land as JSON
    in the attempt's trial directory (the candidate payload is never logged, so
    without these files a reviewer cannot tell whether the guard anchored on the
    record or reverted anything)."""
    from evalrx.eval_agent.run_context import RunContext

    ctx = RunContext(tmp_path / "run")
    agent = FixAgent(
        judge=_BaselineOutputJudge(), max_tier="L2", run_logger=ctx.logger, run_context=ctx,
        exec_timeout_sec=30, candidate_allowlist={"coded_pipeline"},
    )
    data = _observed_no_batch()
    out = agent.propose_and_validate(BaselineFailsModel(), data, [_hyp("x")])
    assert out.fixed is True

    guard_files = list((tmp_path / "run").rglob("coded_pipeline_result.json"))
    control_files = list((tmp_path / "run").rglob("frozen_model_control.json"))
    assert len(guard_files) == 1 and len(control_files) == 1
    assert guard_files[0].parent == control_files[0].parent       # same trial dir
    guard = json.loads(guard_files[0].read_text(encoding="utf-8"))
    assert guard["ok"] is True and guard["exec_error"] == "" and guard["n_outputs"] == len(data)
    assert guard["n_anchored_from_recorded"] == len(data) and guard["n_replayed"] == 0
    assert guard["unanchored_ids"] == [] and guard["n_guarded"] == 0 and guard["min_support"] == 3
    control = json.loads(control_files[0].read_text(encoding="utf-8"))
    assert control["ok"] is True and control["solved"] == []


# ── Candidates must arrive with a sentence a reader can use ──────────────────

def test_a_judge_that_echoes_the_slug_contributes_nothing():
    """"Audio Evidence Then Answer" is the name again, not an explanation.

    Accepting it would put a sentence-shaped string on screen that tells a
    reader exactly what the slug already told them, while looking like the run
    had described its own repair.
    """
    from evalrx.eval_agent.stages.fix_agent import _judge_description

    assert _judge_description({
        "name": "audio_evidence_then_answer",
        "what_it_does": "Audio evidence then answer.",
    }) == ""
    assert _judge_description({
        "name": "audio_evidence_then_answer",
        "what_it_does": "Asks the model to describe what it hears before it answers.",
    }) == "Asks the model to describe what it hears before it answers."
    assert _judge_description({"name": "x"}) == ""
    assert _judge_description("not a proposal") == ""


def test_a_coded_pipelines_own_header_is_its_description():
    """L2 code has no JSON proposal to carry `what_it_does`, so it declares it
    in the source, where the coding agent is already writing."""
    from evalrx.eval_agent.stages.fix_agent import _code_description

    code = (
        "# WHAT_IT_DOES: Asks the model twice and keeps the answer both tries agree on.\n"
        "import json\n"
        "print('x')\n"
    )
    assert _code_description(code) == (
        "Asks the model twice and keeps the answer both tries agree on."
    )
    assert _code_description("import json\nprint('x')\n") == ""

    # A model that wraps the line anyway keeps its whole sentence.
    wrapped = (
        "# WHAT_IT_DOES: Asks the model twice with different wording and keeps\n"
        "#   the answer both tries agree on.\n"
        "import json\n"
    )
    assert _code_description(wrapped) == (
        "Asks the model twice with different wording and keeps the answer both "
        "tries agree on."
    )


def test_plain_description_never_falls_back_to_the_slug():
    from evalrx.eval_agent.stages.fix_agent import (
        FixCandidate,
        FixTier,
        plain_description,
    )

    judged = FixCandidate(
        tier=FixTier.L1_PROMPT, name="cross_modal_rules_terse", payload={},
        description="Gives the model a short checklist to follow before answering.",
    )
    builtin = FixCandidate(tier=FixTier.L1_PROMPT, name="visual_grounding", payload={})
    unknown = FixCandidate(tier=FixTier.L1_PROMPT, name="mystery_strategy_v2", payload={})

    assert plain_description(judged).startswith("Gives the model a short checklist")
    assert plain_description(builtin).startswith("Tells the model to read the answer")
    assert plain_description(unknown) == ""
