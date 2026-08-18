"""Fix module: intervention-space tiers, L2 tool pipelines, validated repair.

The allowed tier is an input (default L2); no automatic escalation — when no
candidate validates, the outcome recommends raising the tier, routed from the
verified hypotheses' mechanisms.
"""

from __future__ import annotations

import json

import pytest

from evalvitals.core.capability import Capability
from evalvitals.core.case import CaseBatch, FailureCase, Inputs, Label
from evalvitals.core.model import Model
from evalvitals.eval_agent import (
    FixAgent,
    FixTier,
    parse_tier,
    route_min_tier,
)
from evalvitals.eval_agent.hypothesis import Hypothesis
from evalvitals.eval_agent.stages.fix_agent import FixCandidate

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
    from evalvitals.eval_agent.stages.fix_tools import upscale, zoom_center

    img = _img()
    assert zoom_center(img, factor=2.0).size == img.size
    assert upscale(img, factor=2.0).size == (128, 96)


def test_apply_image_ops_skips_unknown_and_loads_paths(tmp_path):
    from evalvitals.eval_agent.stages.fix_tools import apply_image_ops

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

    from evalvitals.eval_agent.stages.fix_tools import crop_salient_region

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

    from evalvitals.core.case import FailureCase, Inputs
    from evalvitals.eval_agent.stages.fix_tools import crop_case_bbox

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

    from evalvitals.core.case import FailureCase, Inputs
    from evalvitals.eval_agent.stages.fix_tools import crop_case_bbox

    img = PIL.Image.new("RGB", (32, 32), color=(120, 130, 140))
    case = FailureCase(id="no_bbox", inputs=Inputs(prompt="q", image=img), metadata={})

    out = crop_case_bbox(img, case=case, sharpen_factor=3.0, contrast_factor=1.5)

    assert out is img
    assert np.asarray(out).mean() == np.asarray(img).mean()


def test_run_pipeline_can_fix_textvqa_style_bbox_case():
    PIL = pytest.importorskip("PIL")
    import numpy as np

    from evalvitals.eval_agent.stages.fix_tools import PipelineSpec, run_pipeline

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

    from evalvitals.eval_agent.stages.fix_tools import separate_horizontal_bands

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

    from evalvitals.eval_agent.stages.fix_tools import (
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
    from evalvitals.eval_agent.stages.fix_tools import PipelineSpec

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


def test_pipeline_passes_bounded_generation_kwargs():
    from evalvitals.eval_agent.stages.fix_tools import PipelineSpec, run_pipeline

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


def test_pipeline_self_refine_is_a_label_blind_reviewed_multicall_strategy():
    from evalvitals.eval_agent.stages.fix_tools import PipelineSpec, run_pipeline

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
    from evalvitals.eval_agent.stages.fix_tools import PipelineSpec, run_pipeline

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
    from evalvitals.eval_agent.stages.fix_tools import PipelineSpec, run_pipeline

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
    from evalvitals.analyzers.perturbation.prompt_contrast import _default_score

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
            return "cello"

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


def test_l0_vcd_candidate_repairs_binary_visual_grounding():
    batch = _gold_yes_batch(n=8, image=_img())
    for case in batch:
        case.metadata["task"] = "yes_no"

    out = FixAgent(judge=None, max_tier="L0").propose_and_validate(
        VCDSensitiveModel(), batch, [_hyp("language priors override visual evidence")]
    )

    assert out.fixed is True
    assert out.best is not None and out.best.candidate.name == "vcd_diffusion_noise"
    assert out.best.n_fixed == 8 and out.best.n_broken == 0
    assert out.best.candidate.payload["noise_step"] == 999


def test_l0_vcd_is_not_proposed_when_false_negatives_dominate_binary_diagnosis():
    batch = _gold_yes_batch(n=8, image=_img())
    for case in batch:
        case.metadata["task"] = "yes_no"
        case.expected = "Yes"
        case.observed = "No"
    candidates = FixAgent(judge=None, max_tier="L0")._propose(
        [_hyp("language priors override visual evidence")], batch, VCDSensitiveModel()
    )

    assert "vcd_diffusion_noise" not in {candidate.name for candidate in candidates}


def test_l0_icd_is_not_proposed_when_false_negatives_dominate_binary_diagnosis():
    batch = _gold_yes_batch(n=8, image=_img())
    for case in batch:
        case.metadata["task"] = "yes_no"
        case.expected = "Yes"
        case.observed = "No"
    candidates = FixAgent(judge=None, max_tier="L0")._propose(
        [_hyp("instruction priors override visual evidence")], batch, ICDSensitiveModel()
    )

    candidate_names = {candidate.name for candidate in candidates}
    assert "icd_instruction_disturbance" not in candidate_names
    assert "icd_instruction_disturbance_question" not in candidate_names


def test_l0_vcd_is_proposed_when_false_yes_hallucinations_dominate():
    batch = _gold_yes_batch(n=8, image=_img())
    for case in batch:
        case.metadata["task"] = "yes_no"
        case.expected = "No"
        case.observed = "Yes"
    candidates = FixAgent(judge=None, max_tier="L0")._propose(
        [_hyp("language priors override visual evidence")], batch, VCDSensitiveModel()
    )

    assert "vcd_diffusion_noise" in {candidate.name for candidate in candidates}


def test_l0_aad_candidate_repairs_binary_audio_grounding():
    batch = _gold_audio_yes_batch(n=8)

    out = FixAgent(judge=None, max_tier="L0").propose_and_validate(
        AADSensitiveModel(), batch, [_hyp("language priors override audio evidence")]
    )

    assert out.fixed is True
    assert out.best is not None and out.best.candidate.name == "aad_silence_contrast"
    assert out.best.n_fixed == 8 and out.best.n_broken == 0
    assert out.best.candidate.payload["alpha"] == 0.5


def test_l0_aad_is_not_proposed_when_false_negatives_dominate_binary_diagnosis():
    batch = _gold_audio_yes_batch(n=8)
    for case in batch:
        case.expected = "Yes"
        case.observed = "No"
    candidates = FixAgent(judge=None, max_tier="L0")._propose(
        [_hyp("language priors override audio evidence")], batch, AADSensitiveModel()
    )

    assert "aad_silence_contrast" not in {candidate.name for candidate in candidates}


def test_l0_aad_is_proposed_when_false_yes_hallucinations_dominate():
    batch = _gold_audio_yes_batch(n=8)
    for case in batch:
        case.expected = "No"
        case.observed = "Yes"
    candidates = FixAgent(judge=None, max_tier="L0")._propose(
        [_hyp("language priors override audio evidence")], batch, AADSensitiveModel()
    )

    assert "aad_silence_contrast" in {candidate.name for candidate in candidates}


def test_l0_aad_requires_paper_method_fidelity():
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

    candidates = FixAgent(judge=None, max_tier="L0")._propose(
        [_hyp("language priors override audio evidence")], batch, UnfitAudioModel()
    )

    assert "aad_silence_contrast" not in {candidate.name for candidate in candidates}
    assert "aad_silence_contrast_gated_false_yes" not in {candidate.name for candidate in candidates}


def test_l0_icd_candidate_repairs_binary_visual_grounding():
    batch = _gold_yes_batch(n=8, image=_img())
    for case in batch:
        case.metadata["task"] = "yes_no"

    out = FixAgent(judge=None, max_tier="L0").propose_and_validate(
        ICDSensitiveModel(), batch, [_hyp("instruction priors override visual evidence")]
    )

    assert out.fixed is True
    assert out.best is not None and out.best.candidate.name == "icd_instruction_disturbance"
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
        max_tier="L0",
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


def test_no_rubric_cases_yield_recommendation_not_crash():
    cases = CaseBatch([FailureCase(id="u", inputs=Inputs(prompt="q"), label=Label.FAIL)])
    agent = FixAgent(judge=None, max_tier="L1")
    out = agent.propose_and_validate(HopelessModel(), cases, [_hyp("x")])
    assert out.fixed is False
    assert out.attempted == []
    assert "no case carries a scoring rubric" in out.recommendation["reason"]


def test_broken_cases_counted_and_net_negative_not_fixed():
    """A candidate that repairs nothing and breaks passing cases must not pass."""

    class InvertModel(Model):
        capabilities = frozenset({Capability.GENERATE})
        modalities = frozenset({"text", "image"})

        def generate(self, inputs, **kwargs):
            p = str(getattr(inputs, "prompt", inputs)).lower()
            return "No." if "carefully" in p else "Yes."

        def forward(self, inputs, capture, spec=None):
            raise NotImplementedError

    judge = ScriptedJudge(
        json.dumps([{"name": "careful", "prompt_template": "Look very carefully. {prompt}"}])
    )
    agent = FixAgent(judge=judge, max_tier="L1")
    out = agent.propose_and_validate(InvertModel(), _gold_yes_batch(), [_hyp("x")])
    v = out.attempted[0]
    assert v.n_broken == 8 and v.n_fixed == 0
    assert v.fixed is False and out.fixed is False


def test_outcome_serializes_and_logs(tmp_path):
    from evalvitals.eval_agent import RunLogger

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
    from evalvitals.eval_agent import VLDiagnoseLoop, VLDiagnoseReport
    from evalvitals.eval_agent.hypothesis import HypothesisStatus
    from evalvitals.eval_agent.stages.hypothesis_tester import HypothesisTestResult
    from evalvitals.eval_agent.stages.protocol import ExperimentProtocol

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
    ans = model_generate(c["id"], image_ops=[{"tool": "upscale", "params": {"factor": 2.0}}])
    out.append({"sample_id": c["id"], "output": ans})
print("FIX_PIPELINE_RESULT_JSON=" + json.dumps({"per_case": out}))
"""


def test_cases_payload_never_leaks_labels_or_rubrics():
    from evalvitals.eval_agent.stages.fix_pipeline import cases_payload

    payload = cases_payload(_gold_yes_batch())
    assert all(set(c) == {"id", "prompt"} for c in payload["cases"])


def test_coded_pipeline_bridge_round_trip(tmp_path):
    pytest.importorskip("PIL")
    from evalvitals.analyzers.perturbation.prompt_contrast import _default_score
    from evalvitals.eval_agent.stages.fix_pipeline import (
        run_coded_pipeline,
        score_outputs,
    )

    cases = _gold_yes_batch(n=3, image=_img())
    result = run_coded_pipeline(
        _UPSCALE_PIPELINE, ZoomSensitiveModel(), cases, workdir=tmp_path, timeout_sec=30
    )
    assert result.ok and result.n_calls == 3
    scores = score_outputs(result, cases, _default_score)
    assert all(scores[c.id] is True for c in cases)  # upscale repairs every case


def test_score_outputs_coerces_label_scores():
    from evalvitals.eval_agent.stages.fix_pipeline import CodedPipelineResult, score_outputs

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
        "evalvitals.eval_agent.stages.fix_pipeline",
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
    from evalvitals.eval_agent.stages.fix_pipeline import run_coded_pipeline

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
    from evalvitals.eval_agent.stages.fix_pipeline import run_coded_pipeline

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
    from evalvitals.eval_agent.stages.fix_pipeline import run_coded_pipeline

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
    from evalvitals.eval_agent import RunLogger

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
    from evalvitals.eval_agent.stages.fix_pipeline import run_coded_pipeline

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

        from evalvitals.core.model import Trace

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

    from evalvitals.analyzers.attention.relative_attn import attention_heatmap

    case = FailureCase(id="x", inputs=Inputs(prompt="q", image=_bright_corner_img()))
    grid = attention_heatmap(AttnCropVLM(), case)
    assert grid is not None and grid.shape == (3, 4)
    assert np.unravel_index(grid.argmax(), grid.shape) == (0, 0)


def test_attention_capture_shared_reduction_matches_inline():
    """image_token_attention is the single reduction both consumers share —
    head-mean of the last query row over image tokens."""
    from evalvitals.analyzers.attention.relative_attn import image_token_attention
    from evalvitals.core.capability import Capability

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
    from evalvitals.eval_agent.stages import fix_internals
    from evalvitals.eval_agent.stages.fix_internals import INTERNALS_PRIMITIVES

    assert "attention_guided_crop" not in INTERNALS_PRIMITIVES
    assert all(p.tier is FixTier.L3B_INTERNALS_WRITE for p in INTERNALS_PRIMITIVES.values())
    assert not hasattr(fix_internals, "run_attention_guided_crop")
    assert not hasattr(fix_internals, "peak_box")
    assert not hasattr(fix_internals, "attention_heatmap")  # moved to relative_attn


def test_l3a_read_is_authored_not_a_canned_primitive():
    """At L3a the read lever is the coded pipeline's bridged model_attend(), not
    a primitive: (a) no primitive candidate is proposed, and (b) the coded
    candidate is tagged L3a with enable_attend=True."""
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

    # (b) the read lever still exists, but as agent-written code carrying the
    # model_attend bridge (enable_attend), tagged at the L3a tier.
    coded = FixAgent(judge=CodeWritingJudge(), max_tier="L3a", allow_codegen=True)
    cands = coded._propose([hyp], cases, AttnCropVLM())
    code_cands = [c for c in cands if c.kind == "code"]
    assert code_cands and code_cands[0].tier is FixTier.L3A_INTERNALS_READ
    assert code_cands[0].payload.get("enable_attend") is True
    assert not any(c.kind == "primitive" for c in cands)


# ── L3b: visual embedding boost ──────────────────────────────────────────────


def test_visual_embedding_boost_hook_scales_image_tokens():
    torch = pytest.importorskip("torch")
    import types

    from evalvitals.eval_agent.stages.fix_internals import visual_embedding_boost

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
    from evalvitals.analyzers.perturbation.prompt_contrast import _default_score
    from evalvitals.eval_agent.stages.fix_internals import (
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
    from evalvitals.core.spec import ModelSpec
    from evalvitals.models.backends.base import RuntimeConfig
    from evalvitals.models.backends.hf_local import HFLocalModel

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
    from evalvitals.core.spec import ModelSpec
    from evalvitals.eval_agent.stages import fix_internals
    from evalvitals.models.backends.base import RuntimeConfig
    from evalvitals.models.backends.hf_local import HFLocalModel

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
    from evalvitals.eval_agent.stages.fix_agent import FixCandidate

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
    from evalvitals.eval_agent.stages.fix_agent import FixCandidate

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


def test_assertive_grounding_offered_only_on_false_no_dominant_slice():
    """assertive_grounding is the dual of the direction gate that withholds
    VCD/ICD/etc. on a false-No-dominant slice: those methods are suppressive
    (wrong direction for under-claiming), so offer a prompt that accepts
    partial evidence instead. Must not appear when the slice is false-Yes
    dominant (or balanced) -- that's exactly the population the suppressive
    methods already handle."""
    agent = FixAgent(judge=None, max_tier="L1")
    out_false_no = agent._l1_candidates(
        "- h", "", has_images=True, tasks={"yes_no"}, binary_hallucination_supported=False
    )
    assert "assertive_grounding" in {c.name for c in out_false_no}

    out_false_yes = agent._l1_candidates(
        "- h", "", has_images=True, tasks={"yes_no"}, binary_hallucination_supported=True
    )
    assert "assertive_grounding" not in {c.name for c in out_false_yes}


def test_spec_noop_cases_are_not_applicable():
    PIL = pytest.importorskip("PIL")
    from evalvitals.eval_agent.stages.fix_tools import PipelineSpec, spec_changes_input

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


def test_baseline_repeats_flag_and_drop_unstable_cases():
    agent = FixAgent(judge=None, max_tier="L1", baseline_repeats=2)
    data = _gold_yes_batch(n=2)
    model = _OneFlakyModel()
    baseline, unstable = agent._baseline(model, data)
    assert "c0" in unstable and "c1" not in unstable

    from evalvitals.eval_agent.stages.fix_agent import FixCandidate

    cand = FixCandidate(
        tier=FixTier.L1_PROMPT,
        name="careful",
        kind="template",
        payload={"prompt_template": "Look carefully. {prompt}"},
    )
    v = agent._validate(cand, model, data, baseline, unstable)
    # c0 is noise -> dropped, not counted as fixed or broken; only c1 is judged.
    assert v.n_unstable == 1 and v.n_pairs == 1
    assert v.fixed_cases == ["c1"]


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
    from evalvitals.analyzers.perturbation.prompt_contrast import _default_score
    from evalvitals.eval_agent.stages.fix_pipeline import (
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
    from evalvitals.eval_agent.run_context import RunContext

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
    from evalvitals.eval_agent.run_context import RunContext

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
    ans = model_generate(c["id"], prompt="SECRET_MARKER " + c["prompt"])
    out.append({"sample_id": c["id"], "output": ans})
print("FIX_PIPELINE_RESULT_JSON=" + json.dumps({"per_case": out}))
"""


def test_two_coded_fix_attempts_get_separate_trial_workspaces(tmp_path):
    pytest.importorskip("PIL")
    from evalvitals.eval_agent.run_context import RunContext

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
