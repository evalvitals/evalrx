"""Model spec registry — declarative ModelSpec entries, one per family.

A plain dict (no import-side-effect decorator).  Module paths are HINTS only;
the hf_local backend discovers the real decoder-layer ModuleList at load time
(see :mod:`evalrx.models._discover`).  Token ids / merge sizes are read from
the live config via the attribute NAMES in ``VisionSpec`` — never baked as values
(GLM-4.5V 151363 vs GLM-4.1V 151343 is exactly why).

Paths reflect the verified post-VLM-refactor transformers layout (single
``.model``; Kimi/Llama4 have no outer ``.model``).  ``tool_calling=True`` marks
instruct/thinking checkpoints whose chat template renders tools (grants
TOOL_CALLS on the local backend; verified against the template at load time).
This module is torch-free.
"""

from __future__ import annotations

from evalrx.core.spec import AttnSemantics, AudioSpec, ModelSpec, ModulePaths, VisionSpec

REGISTRY: dict[str, ModelSpec] = {}


def _add(spec: ModelSpec) -> None:
    REGISTRY[spec.key] = spec


def get_spec(key: str) -> ModelSpec:
    if key not in REGISTRY:
        raise KeyError(f"Unknown model spec {key!r}. Known: {sorted(REGISTRY)}")
    return REGISTRY[key]


def list_specs() -> list[str]:
    return sorted(REGISTRY)


# ----------------------------------------------------------------------
# LLMs (no vision tower / no TokenTypeMap)
# ----------------------------------------------------------------------
_add(ModelSpec(
    key="qwen2.5-7b-instruct", family="qwen2", model_type="qwen2",
    hf_repo="Qwen/Qwen2.5-7B-Instruct", auto_class="AutoModelForCausalLM",
    processor_class="AutoTokenizer", min_transformers="4.43.0", tool_calling=True,
    module_paths=ModulePaths(decoder_layers="model.layers"),
    caveats=("matches the legacy QwenLLM default checkpoint",),
))
_add(ModelSpec(
    key="qwen3-4b", family="qwen3", model_type="qwen3",
    hf_repo="Qwen/Qwen3-4B", auto_class="AutoModelForCausalLM",
    processor_class="AutoTokenizer", min_transformers="4.51.0",
    is_reasoning=True, tool_calling=True,
    chat_template_kwargs={"enable_thinking": False},  # fast/clean tool-calling for the smoke test
    module_paths=ModulePaths(decoder_layers="model.layers"),
    caveats=("small smoke-test checkpoint; q_norm/k_norm before RoPE; emits <think> by default",),
))
_add(ModelSpec(
    key="qwen3-8b", family="qwen3", model_type="qwen3",
    hf_repo="Qwen/Qwen3-8B", auto_class="AutoModelForCausalLM",
    processor_class="AutoTokenizer", min_transformers="4.51.0",
    is_reasoning=True, tool_calling=True,
    module_paths=ModulePaths(decoder_layers="model.layers"),
    caveats=("q_norm/k_norm applied to Q/K before RoPE (account for it in lens)",),
))
_add(ModelSpec(
    key="qwen3-30b-a3b", family="qwen3_moe", model_type="qwen3_moe",
    hf_repo="Qwen/Qwen3-30B-A3B", auto_class="AutoModelForCausalLM",
    processor_class="AutoTokenizer", min_transformers="4.51.0", is_moe=True, tool_calling=True,
    module_paths=ModulePaths(decoder_layers="model.layers", router="mlp.gate", experts="mlp.experts"),
    caveats=("v5 stores experts as fused stacked Parameters (not ModuleList) — detect at runtime",),
))
_add(ModelSpec(
    key="deepseek-v3", family="deepseek_v3", model_type="deepseek_v3",
    hf_repo="deepseek-ai/DeepSeek-V3", auto_class="AutoModelForCausalLM",
    processor_class="AutoTokenizer", min_transformers="4.51.0", is_moe=True, tool_calling=True,
    attn_semantics=AttnSemantics.MLA_LATENT,
    module_paths=ModulePaths(decoder_layers="model.layers", router="mlp.gate", experts="mlp.experts"),
    caveats=(
        "MLA: eager materialises weights in decompressed latent head space, not raw token space",
        "HF impl naive; 671B multi-node — practical white-box on DeepSeek-V2-Lite",
    ),
))
_add(ModelSpec(
    key="llama-3.1-8b-instruct", family="llama", model_type="llama",
    hf_repo="meta-llama/Llama-3.1-8B-Instruct", auto_class="AutoModelForCausalLM",
    processor_class="AutoTokenizer", min_transformers="4.43.0", tool_calling=True,
    module_paths=ModulePaths(decoder_layers="model.layers"),
    caveats=(
        "cleanest arch; also loads in TransformerLens",
        "tool-call format differs from Qwen/Hermes — add a Llama codec before agent use",
    ),
))
_add(ModelSpec(
    key="gemma-3-1b-it", family="gemma3", model_type="gemma3",
    hf_repo="google/gemma-3-1b-it", auto_class="AutoModelForCausalLM",
    processor_class="AutoTokenizer", min_transformers="4.50.0",
    module_paths=ModulePaths(decoder_layers="model.layers"),
    caveats=("text-only size; tied embeddings by default",),
))

# Qwen3.5 — the checkpoints examples/dataset_selection/llm_benchmark diagnoses. Registered as
# TEXT specs on purpose, see the caveats: the released checkpoint is
# ``Qwen3_5ForConditionalGeneration`` and carries a vision tower, but this
# pipeline only ever sends text, and ``AutoModelForCausalLM`` maps ``qwen3_5``
# to ``Qwen3_5ForCausalLM`` — the language tower alone, which is both lighter
# and the thing whose attention we want to read.
for _key, _repo in (
    ("qwen3.5-2b", "Qwen/Qwen3.5-2B"),
    ("qwen3.5-4b", "Qwen/Qwen3.5-4B"),
    ("qwen3.5-9b", "Qwen/Qwen3.5-9B"),
):
    _add(ModelSpec(
        key=_key, family="qwen3_5", model_type="qwen3_5", hf_repo=_repo,
        auto_class="AutoModelForCausalLM", processor_class="AutoTokenizer",
        min_transformers="5.15.0", is_reasoning=True, tool_calling=True,
        # Thinking OFF on every template render (chat, generate, logprobs) --
        # see the caveat; pass chat_template_kwargs={} to get the template default.
        chat_template_kwargs={"enable_thinking": False},
        attn_semantics=AttnSemantics.HYBRID_SPARSE,
        module_paths=ModulePaths(decoder_layers="model.layers"),
        caveats=(
            "HYBRID stack: layer_types is [linear, linear, linear, full] x 8 "
            "(full_attention_interval=4), so a forward returns 8 attention "
            "tensors for 32 layers and position i is model layer 4i+3 — "
            "attention_rollout composes a partial path here and must not be "
            "read as a full-depth rollout",
            "checkpoint is Qwen3_5ForConditionalGeneration (vision tower + "
            "image_token_id); this spec deliberately loads the text tower only, "
            "so no image analyzer is offered and no TokenTypeMap is built",
            "min_transformers is the version VERIFIED to work (qwen3_5 is absent "
            "from 4.57.6), not a discovered floor — and nothing in the framework "
            "enforces the field, so the loader checks AutoConfig itself",
            "thinking is OFF here: this spec sends enable_thinking=False on every "
            "template render, so a completion is the answer with an empty think "
            "block in the prompt. The checkpoints disagree on the template default "
            "when the kwarg is absent (Qwen3.5-2B: off, Qwen3.5-9B: on -- there the "
            "chain lands inline with a closing '</think>' but no opening tag), "
            "which is why it is always sent explicitly",
        ),
    ))

# Qwen3.5 WITH its vision tower -- the same checkpoints as the text specs above,
# loaded as ``Qwen3_5ForConditionalGeneration`` (``AutoModelForImageTextToText``)
# for the image benchmarks (examples/m1_m4/*_qwen3_5_2b). Layout verified on
# transformers 5.15.0: ``model.language_model.layers`` (24 layers on the 2B,
# ``layer_types`` = [linear x3, full] x 6), ``model.visual.blocks``,
# ``config.image_token_id`` top-level, ``vision_config.spatial_merge_size`` = 2,
# processor emits ``image_grid_thw`` -- i.e. the Qwen3-VL shape.
for _key, _repo, _n_layers in (
    ("qwen3.5-2b-vl", "Qwen/Qwen3.5-2B", 24),
    ("qwen3.5-4b-vl", "Qwen/Qwen3.5-4B", 32),
    ("qwen3.5-9b-vl", "Qwen/Qwen3.5-9B", 32),
):
    _add(ModelSpec(
        key=_key, family="qwen3_5", model_type="qwen3_5", hf_repo=_repo,
        auto_class="AutoModelForImageTextToText", processor_class="AutoProcessor",
        min_transformers="5.15.0", is_reasoning=True, tool_calling=True,
        # Thinking OFF on every template render (chat, generate, logprobs);
        # explicit because the 2B/9B templates disagree on the default.
        chat_template_kwargs={"enable_thinking": False},
        attn_semantics=AttnSemantics.HYBRID_SPARSE,
        module_paths=ModulePaths(decoder_layers="model.language_model.layers",
                                 vision_tower="model.visual",
                                 vision_blocks="model.visual.blocks"),
        vision=VisionSpec(image_token_id_attr="image_token_id",
                          merge_size_attr="vision_config.spatial_merge_size",
                          grid_source="grid_thw"),
        caveats=(
            f"HYBRID stack: layer_types is [linear, linear, linear, full] x "
            f"{_n_layers // 4} (full_attention_interval=4), so a forward returns "
            f"{_n_layers // 4} attention tensors for {_n_layers} layers and position "
            "i is model layer 4i+3 -- attention_rollout composes a partial path "
            "here and must not be read as a full-depth rollout",
            "same checkpoint as the text spec without the -vl suffix; this one "
            "loads the vision tower (Qwen3_5ForConditionalGeneration) and builds "
            "the TokenTypeMap, so the image analyzers are offered",
            "needs transformers >= 5.15 (qwen3_5 is absent from 4.57); the "
            "package's [local] extra pins transformers < 5, install it explicitly "
            "(see examples/m1_m4/chartqa_qwen3_5_2b/Dockerfile)",
            "thinking is OFF here: the spec sends enable_thinking=False on every "
            "template render; the 2B template defaults off and the 9B template "
            "defaults on when the kwarg is absent",
        ),
    ))

# ----------------------------------------------------------------------
# VLMs (vision tower + TokenTypeMap)
# ----------------------------------------------------------------------
_add(ModelSpec(
    key="qwen3-vl-4b-instruct", family="qwen3_vl", model_type="qwen3_vl",
    hf_repo="Qwen/Qwen3-VL-4B-Instruct", auto_class="AutoModelForImageTextToText",
    processor_class="AutoProcessor", min_transformers="4.57.0", tool_calling=True,
    chat_template_kwargs={},
    module_paths=ModulePaths(
        decoder_layers="model.language_model.layers", vision_tower="model.visual",
        vision_blocks="model.visual.blocks"),
    vision=VisionSpec(image_token_id_attr="image_token_id",
                      merge_size_attr="vision_config.spatial_merge_size", grid_source="grid_thw"),
    caveats=("small smoke-test VLM checkpoint; single .model layout; DeepStack",),
))
_add(ModelSpec(
    key="llava-1.5-7b-hf", family="llava", model_type="llava",
    hf_repo="llava-hf/llava-1.5-7b-hf", auto_class="AutoModelForImageTextToText",
    processor_class="AutoProcessor", min_transformers="4.36.0",
    module_paths=ModulePaths(decoder_layers="language_model.model.layers", vision_tower="vision_tower"),
    vision=VisionSpec(
        image_token_id_attr="image_token_index", grid_source="fixed", fixed_tokens_per_tile=576
    ),
    caveats=(
        "reference architecture for ViCrop's LLaVA experiments — https://arxiv.org/abs/2502.17422",
        "CLIP ViT-L/14 at 336px yields a 24x24 image-patch grid",
    ),
))
_add(ModelSpec(
    key="instructblip-vicuna-7b", family="instructblip", model_type="instructblip",
    hf_repo="Salesforce/instructblip-vicuna-7b",
    auto_class="InstructBlipForConditionalGeneration", processor_class="InstructBlipProcessor",
    min_transformers="4.46.0",
    module_paths=ModulePaths(decoder_layers="language_model.model.layers", vision_tower="vision_model"),
    # InstructBLIP fuses visual queries into embeddings; its decoder input_ids
    # have no image-placeholder token block.  The backend treats its token map
    # as text-only while still declaring image support.
    vision=VisionSpec(grid_source="fixed"),
    caveats=(
        "reference Q-Former architecture for ICD's instruction-disturbance route",
        "ICD alters qformer_input_ids only; the decoder prompt remains unchanged",
    ),
))
_add(ModelSpec(
    key="qwen2.5-vl-7b-instruct", family="qwen2_5_vl", model_type="qwen2_5_vl",
    hf_repo="Qwen/Qwen2.5-VL-7B-Instruct", auto_class="AutoModelForImageTextToText",
    processor_class="AutoProcessor", min_transformers="4.49.0", tool_calling=True,
    module_paths=ModulePaths(
        decoder_layers="model.language_model.layers", vision_tower="model.visual",
        vision_blocks="model.visual.blocks"),
    vision=VisionSpec(image_token_id_attr="image_token_id",
                      merge_size_attr="vision_config.spatial_merge_size", grid_source="grid_thw"),
    caveats=(
        "reference model for 'MLLMs Know Where to Look' — https://arxiv.org/abs/2502.17422",
        "paper code: https://github.com/saccharomycetes/mllms_know",
        "relative_attention layer=22 recommended per the paper",
    ),
))
_add(ModelSpec(
    key="qwen2-vl-7b-instruct", family="qwen2_vl", model_type="qwen2_vl",
    hf_repo="Qwen/Qwen2-VL-7B-Instruct", auto_class="AutoModelForImageTextToText",
    processor_class="AutoProcessor", min_transformers="4.46.0", tool_calling=True,
    module_paths=ModulePaths(
        decoder_layers="model.language_model.layers", vision_tower="model.visual",
        vision_blocks="model.visual.blocks"),
    vision=VisionSpec(image_token_id_attr="image_token_id",
                      merge_size_attr="vision_config.spatial_merge_size", grid_source="grid_thw"),
    caveats=("predecessor to Qwen2.5-VL; same architecture, fewer params tuned",),
))
_add(ModelSpec(
    key="qwen3-vl-8b-instruct", family="qwen3_vl", model_type="qwen3_vl",
    hf_repo="Qwen/Qwen3-VL-8B-Instruct", auto_class="AutoModelForImageTextToText",
    processor_class="AutoProcessor", min_transformers="4.57.0", tool_calling=True,
    module_paths=ModulePaths(
        decoder_layers="model.language_model.layers", vision_tower="model.visual",
        vision_blocks="model.visual.blocks"),
    vision=VisionSpec(image_token_id_attr="image_token_id",
                      merge_size_attr="vision_config.spatial_merge_size", grid_source="grid_thw"),
    caveats=(
        "single .model (model.language_model.layers / model.visual) — verified vs transformers main",
        "DeepStack injects vision residuals into early text layers; 'embedding at pos p' not a single tensor",
        "pop token_type_ids before generate; interleaved-MRoPE 3D position_ids",
    ),
))

# ---- additional Qwen sizes/variants (same per-family fields; paths discovered at load) ----
for _key, _repo in [
    ("qwen2.5-14b-instruct", "Qwen/Qwen2.5-14B-Instruct"),
    ("qwen2.5-32b-instruct", "Qwen/Qwen2.5-32B-Instruct"),
    ("qwen2.5-72b-instruct", "Qwen/Qwen2.5-72B-Instruct"),
]:
    _add(ModelSpec(
        key=_key, family="qwen2", model_type="qwen2", hf_repo=_repo,
        auto_class="AutoModelForCausalLM", processor_class="AutoTokenizer",
        min_transformers="4.43.0", tool_calling=True,
        module_paths=ModulePaths(decoder_layers="model.layers"),
    ))
for _key, _repo in [
    ("qwen3-14b", "Qwen/Qwen3-14B"),
    ("qwen3-32b", "Qwen/Qwen3-32B"),
]:
    _add(ModelSpec(
        key=_key, family="qwen3", model_type="qwen3", hf_repo=_repo,
        auto_class="AutoModelForCausalLM", processor_class="AutoTokenizer",
        min_transformers="4.51.0", is_reasoning=True, tool_calling=True,
        module_paths=ModulePaths(decoder_layers="model.layers"),
        caveats=("q_norm/k_norm before RoPE",),
    ))
_add(ModelSpec(
    key="qwen3-235b-a22b", family="qwen3_moe", model_type="qwen3_moe",
    hf_repo="Qwen/Qwen3-235B-A22B", auto_class="AutoModelForCausalLM",
    processor_class="AutoTokenizer", min_transformers="4.51.0", is_moe=True,
    is_reasoning=True, tool_calling=True,
    module_paths=ModulePaths(decoder_layers="model.layers", router="mlp.gate", experts="mlp.experts"),
    caveats=("128 experts top-8; multi-GPU/FP8; v5 fused-experts — detect at runtime",),
))
# Qwen2.5-VL dense sizes
for _key, _repo in [
    ("qwen2.5-vl-3b-instruct", "Qwen/Qwen2.5-VL-3B-Instruct"),
    ("qwen2.5-vl-32b-instruct", "Qwen/Qwen2.5-VL-32B-Instruct"),
    ("qwen2.5-vl-72b-instruct", "Qwen/Qwen2.5-VL-72B-Instruct"),
]:
    _add(ModelSpec(
        key=_key, family="qwen2_5_vl", model_type="qwen2_5_vl", hf_repo=_repo,
        auto_class="AutoModelForImageTextToText", processor_class="AutoProcessor",
        min_transformers="4.49.0", tool_calling=True,
        module_paths=ModulePaths(decoder_layers="model.language_model.layers",
                                 vision_tower="model.visual", vision_blocks="model.visual.blocks"),
        vision=VisionSpec(image_token_id_attr="image_token_id",
                          merge_size_attr="vision_config.spatial_merge_size", grid_source="grid_thw"),
    ))
# Qwen3-VL dense + MoE sizes
_add(ModelSpec(
    key="qwen3-vl-2b-instruct", family="qwen3_vl", model_type="qwen3_vl",
    hf_repo="Qwen/Qwen3-VL-2B-Instruct", auto_class="AutoModelForImageTextToText",
    processor_class="AutoProcessor", min_transformers="4.57.0", tool_calling=True,
    module_paths=ModulePaths(decoder_layers="model.language_model.layers",
                             vision_tower="model.visual", vision_blocks="model.visual.blocks"),
    vision=VisionSpec(image_token_id_attr="image_token_id",
                      merge_size_attr="vision_config.spatial_merge_size", grid_source="grid_thw"),
    caveats=("smallest Qwen3-VL; single .model; DeepStack",),
))
for _key, _repo in [
    ("qwen3-vl-30b-a3b-instruct", "Qwen/Qwen3-VL-30B-A3B-Instruct"),
    ("qwen3-vl-235b-a22b-instruct", "Qwen/Qwen3-VL-235B-A22B-Instruct"),
]:
    _add(ModelSpec(
        key=_key, family="qwen3_vl_moe", model_type="qwen3_vl_moe", hf_repo=_repo,
        auto_class="AutoModelForImageTextToText", processor_class="AutoProcessor",
        min_transformers="4.57.0", is_moe=True, tool_calling=True,
        module_paths=ModulePaths(decoder_layers="model.language_model.layers",
                                 vision_tower="model.visual", router="mlp.gate", experts="mlp.experts"),
        vision=VisionSpec(image_token_id_attr="image_token_id",
                          merge_size_attr="vision_config.spatial_merge_size", grid_source="grid_thw"),
        caveats=("MoE VLM (multi-GPU/FP8); DeepStack; expert count read from config, not baked",),
    ))

# ----------------------------------------------------------------------
# Omni models (text + image + audio + video) — Qwen3-Omni reference.
# Not a new class fork: an omni spec just carries vision + audio (+ video), so
# ``modalities`` becomes {text, image, audio, video} and analyzers match on it.
# The thinker is the multimodal LM that emits text — what failure analysis hooks;
# the talker (speech synthesis) is out of scope. White-box token maps over the
# audio/vision towers are Stage-2 (nested ``thinker_config`` paths verified at load).
# https://github.com/QwenLM/Qwen3-Omni
# ----------------------------------------------------------------------
_OMNI_PATHS = ModulePaths(
    decoder_layers="thinker.model.layers",      # discovery resolves the real ModuleList
    vision_tower="thinker.visual", vision_blocks="thinker.visual.blocks",
    router="mlp.gate", experts="mlp.experts",   # thinker text layers are Qwen3-MoE
)
_OMNI_VISION = VisionSpec(
    # Qwen3OmniMoeConfig has no top-level attribute_map to thinker_config, so
    # the token id must be resolved through the nested path (verified against
    # transformers 4.57's Qwen3OmniMoeThinkerConfig source — the top-level
    # config genuinely does not expose image_token_id/audio_token_id itself).
    image_token_id_attr="thinker_config.image_token_id",
    merge_size_attr="thinker_config.vision_config.spatial_merge_size", grid_source="grid_thw",
)
_OMNI_AUDIO = AudioSpec(
    audio_token_id_attr="thinker_config.audio_token_id", audio_tower="thinker.audio_tower"
)
_OMNI_CAVEATS = (
    "Transformers >= 5.2.0 (Qwen3OmniMoeForConditionalGeneration / Qwen3OmniMoeProcessor)",
    "multimodal preprocessing via qwen_omni_utils.process_mm_info; pass use_audio_in_video "
    "consistently to processor AND generate",
    "config nests under thinker_config (vision_config/audio_config) — image/audio token "
    "ids read from the live config at load, never baked; white-box token maps are Stage-2",
    "30B-A3B MoE thinker; talker (speech out) not modelled — analysis targets the thinker text stream",
)
_add(ModelSpec(
    key="qwen3-omni-30b-a3b-instruct", family="qwen3_omni_moe", model_type="qwen3_omni_moe",
    hf_repo="Qwen/Qwen3-Omni-30B-A3B-Instruct",
    auto_class="Qwen3OmniMoeForConditionalGeneration", processor_class="Qwen3OmniMoeProcessor",
    min_transformers="5.2.0", is_moe=True, tool_calling=True,
    module_paths=_OMNI_PATHS, vision=_OMNI_VISION, audio=_OMNI_AUDIO, video=True,
    caveats=_OMNI_CAVEATS,
))
_add(ModelSpec(
    key="qwen3-omni-30b-a3b-thinking", family="qwen3_omni_moe", model_type="qwen3_omni_moe",
    hf_repo="Qwen/Qwen3-Omni-30B-A3B-Thinking",
    auto_class="Qwen3OmniMoeForConditionalGeneration", processor_class="Qwen3OmniMoeProcessor",
    min_transformers="5.2.0", is_moe=True, is_reasoning=True, tool_calling=True,
    module_paths=_OMNI_PATHS, vision=_OMNI_VISION, audio=_OMNI_AUDIO, video=True,
    caveats=_OMNI_CAVEATS + ("emits <think>...</think> before the answer",),
))
_add(ModelSpec(
    key="qwen3-omni-30b-a3b-captioner", family="qwen3_omni_moe", model_type="qwen3_omni_moe",
    hf_repo="Qwen/Qwen3-Omni-30B-A3B-Captioner",
    auto_class="Qwen3OmniMoeForConditionalGeneration", processor_class="Qwen3OmniMoeProcessor",
    min_transformers="5.2.0", is_moe=True,
    module_paths=_OMNI_PATHS, audio=_OMNI_AUDIO,  # audio-in / text-out only
    caveats=_OMNI_CAVEATS + ("audio-only input -> text caption; no image/video heads in use",),
))

# Qwen2.5-Omni — same thinker/talker/token2wav shape as Qwen3-Omni (dense, not
# MoE), so the vision/audio HINTS are byte-identical; reused rather than
# redefined. https://github.com/QwenLM/Qwen2.5-Omni
# https://arxiv.org/abs/2503.20215
_QWEN25_OMNI_CAVEATS = (
    "Transformers >= 4.52.0 (Qwen2_5OmniForConditionalGeneration / Qwen2_5OmniProcessor); "
    "verify against the installed changelog before pinning a release",
    "audio=<mono float32 ndarray @ 16kHz> or path/URL (decoded via ffmpeg) passed straight to "
    "the processor's audio= kwarg -- no qwen_omni_utils dependency needed for text+image+audio",
    "WhisperFeatureExtractor window is 300s (chunk_length) on this checkpoint -- longer clips "
    "raise rather than silently truncate (see hf_local._check_audio_duration)",
    "config nests under thinker_config (vision_config/audio_config) — image/audio token ids "
    "read from the live config at load, never baked",
    "7B dense thinker; talker (speech out) not modelled — analysis targets the thinker text stream",
)
_add(ModelSpec(
    key="qwen2.5-omni-7b", family="qwen2_5_omni", model_type="qwen2_5_omni",
    hf_repo="Qwen/Qwen2.5-Omni-7B",
    auto_class="Qwen2_5OmniForConditionalGeneration", processor_class="Qwen2_5OmniProcessor",
    min_transformers="4.52.0", tool_calling=True,
    module_paths=_OMNI_PATHS, vision=_OMNI_VISION, audio=_OMNI_AUDIO, video=True,
    caveats=_QWEN25_OMNI_CAVEATS,
))

# Qwen2-Audio — audio-in/text-out only (no vision tower), the simplest
# audio-capable spec: one config level, no thinker/talker nesting, so
# audio_token_id resolves directly off the top-level config.
# https://github.com/QwenLM/Qwen2-Audio · https://arxiv.org/abs/2407.10759
_add(ModelSpec(
    key="qwen2-audio-7b-instruct", family="qwen2_audio", model_type="qwen2_audio",
    hf_repo="Qwen/Qwen2-Audio-7B-Instruct",
    auto_class="Qwen2AudioForConditionalGeneration", processor_class="Qwen2AudioProcessor",
    min_transformers="4.45.0",
    module_paths=ModulePaths(decoder_layers="language_model.model.layers"),
    # transformers 5.x places the backbone under Qwen2AudioForConditionalGeneration.model;
    # the encoder is therefore model.audio_tower (the older 4.x class exposed
    # audio_tower directly on the generation wrapper).
    audio=AudioSpec(audio_token_id_attr="audio_token_id", audio_tower="model.audio_tower"),
    caveats=(
        "audio-only input -> text output; no vision/video heads at all",
        "WhisperFeatureExtractor window is 30s (chunk_length) on this checkpoint -- longer "
        "clips raise rather than silently truncate (see hf_local._check_audio_duration)",
        "audio_token_id resolves directly off the top-level config (no thinker_config nesting, "
        "unlike the Omni families)",
    ),
))

_add(ModelSpec(
    key="glm-4.5v", family="glm4v_moe", model_type="glm4v_moe",
    hf_repo="zai-org/GLM-4.5V", auto_class="AutoModelForImageTextToText",
    processor_class="AutoProcessor", min_transformers="4.57.1",
    is_moe=True, is_reasoning=True, tool_calling=True,
    module_paths=ModulePaths(decoder_layers="model.language_model.layers", vision_tower="model.visual"),
    vision=VisionSpec(image_token_id_attr="image_token_id"),
    caveats=("106B MoE needs multi-GPU/FP8", "<think> on by default", "read image_token_id from config (151363)"),
))
_add(ModelSpec(
    key="glm-4.1v-9b-thinking", family="glm4v", model_type="glm4v",
    hf_repo="THUDM/GLM-4.1V-9B-Thinking", auto_class="AutoModelForImageTextToText",
    processor_class="AutoProcessor", min_transformers="4.57.0", is_reasoning=True, tool_calling=True,
    module_paths=ModulePaths(decoder_layers="model.language_model.layers", vision_tower="model.visual"),
    vision=VisionSpec(image_token_id_attr="image_token_id"),
    caveats=("dense 40L; image_token_id=151343 (≠ GLM-4.5V) — always read from config",),
))
_add(ModelSpec(
    key="kimi-vl-a3b-thinking", family="kimi_vl", model_type="kimi_vl",
    hf_repo="moonshotai/Kimi-VL-A3B-Thinking-2506", auto_class="AutoModelForCausalLM",
    processor_class="AutoProcessor", trust_remote_code=True, min_transformers="4.51.0",
    is_moe=True, is_reasoning=True, attn_semantics=AttnSemantics.MLA_LATENT,
    module_paths=ModulePaths(
        decoder_layers="language_model.model.layers", vision_tower="vision_tower",
        vision_blocks="vision_tower.encoder.blocks"),
    vision=VisionSpec(image_token_id_attr="media_placeholder_token_id", grid_source="grid_hw"),
    caveats=(
        "permanent remote-code; LM is an embedded DeepseekV3ForCausalLM (MLA)",
        "MoonViT vision has NO self_attn (fused wqkv + functional attn) — can't use output_attentions there",
        "no outer .model: language_model.model.layers",
    ),
))
_add(ModelSpec(
    key="llama-4-scout", family="llama4", model_type="llama4",
    hf_repo="meta-llama/Llama-4-Scout-17B-16E-Instruct", auto_class="AutoModelForImageTextToText",
    processor_class="AutoProcessor", min_transformers="4.51.0", is_moe=True,
    module_paths=ModulePaths(
        decoder_layers="language_model.model.layers", vision_tower="vision_model"),
    vision=VisionSpec(image_token_id_attr="image_token_index"),
    caveats=(
        "no outer .model (language_model.model.layers)", "fused experts; iRoPE NoPE layers; chunked attention",
        "Llama tool-call format ≠ Qwen/Hermes — add a Llama codec before agent use",
    ),
))
_add(ModelSpec(
    key="step-1o-vision", family="step", model_type="step1o",
    hf_repo="", auto_class="", api_only=True, attn_semantics=AttnSemantics.NONE,
    caveats=("closed weights; api backend only; no InternalsHandle. Open white-box path: step3-vl-10b.",),
))

# ----------------------------------------------------------------------
# examples/benchmark families: Gemma 4 and Nemotron 3 Nano (2026-08-21)
# ----------------------------------------------------------------------
# Gemma 4 (google/gemma-4-*-it) — natively multimodal: text + image on every
# size, audio on E2B / E4B / 12B (the 12B is the encoder-free "Unified" variant:
# raw image patches and audio waveforms are projected straight into the
# decoder, model_type ``gemma4_unified``). Every size has a configurable
# thinking mode; the specs turn it OFF on every template render. Attention is
# a sliding/full interleave (all layers are attention, STANDARD semantics).
# The processor emits ``mm_token_type_ids``, so image positions come from
# there; the per-image patch grid is variable (aspect-ratio aware) and is NOT
# rebuilt (``grid_source="fixed"`` with no tile size leaves ``grids`` empty).
_GEMMA4_CAVEATS = (
    "thinking is OFF here: chat_template_kwargs sends enable_thinking=False on "
    "every template render (the checkpoints ship a configurable thinking mode)",
    "image positions come from the processor's mm_token_type_ids; the patch "
    "grid is variable-resolution and not rebuilt, so spatial attention maps "
    "over the image are unavailable (grids=[])",
    "needs transformers >= 5.15 (gemma4 / gemma4_unified are absent from 4.x); "
    "the package's [local] extra pins transformers < 5, install it explicitly "
    "(see examples/benchmark/docker/Dockerfile)",
    "one spec per checkpoint serves every modality: a text-only prompt takes "
    "the chat-template path with no image/audio block; the M1 modality gate "
    "matches on the MODEL, so pin a text-safe analyzer set for text tasks",
)
for _key, _repo, _model_type, _unified in (
    ("gemma-4-e2b-it", "google/gemma-4-E2B-it", "gemma4", False),
    ("gemma-4-e4b-it", "google/gemma-4-E4B-it", "gemma4", False),
    ("gemma-4-12b-it", "google/gemma-4-12B-it", "gemma4_unified", True),
):
    _add(ModelSpec(
        key=_key, family="gemma4", model_type=_model_type, hf_repo=_repo,
        auto_class="AutoModelForImageTextToText", processor_class="AutoProcessor",
        min_transformers="5.15.0", is_reasoning=True,
        chat_template_kwargs={"enable_thinking": False},
        module_paths=ModulePaths(
            decoder_layers="model.language_model.layers",
            vision_tower=None if _unified else "model.vision_tower",
        ),
        vision=VisionSpec(
            image_token_id_attr="image_token_id", merge_size_attr=None,
            grid_source="fixed", fixed_tokens_per_tile=None,
        ),
        audio=AudioSpec(
            audio_token_id_attr="audio_token_id",
            audio_tower=None if _unified else "model.audio_tower",
            use_audio_in_video=False,
        ),
        caveats=_GEMMA4_CAVEATS + ((
            "Gemma 4 12B Unified: encoder-free (no vision/audio tower modules; "
            "embed_vision / embed_audio linear projections instead), ~24 GB BF16",
        ) if _unified else (
            "per-layer embeddings (PLE): the raw parameter count is ~2.5x the "
            "effective size; ~10 GB (E2B) / ~16 GB (E4B) BF16",
        )),
    ))

# Nemotron 3 Nano (nvidia) — Mamba2 / MLP / attention hybrids. hybrid_override_pattern
# spells the stack (M = Mamba2, - = MLP, E = MoE MLP, * = attention): attention
# exists only at the '*' positions, so HYBRID_SPARSE semantics apply. NVIDIA
# publishes each checkpoint in BF16, FP8 and NVFP4. The FP8/NVFP4 files are
# ModelOpt exports (hf_quant_config.json + fp8 weight/input scales) meant for
# vLLM/TRT-LLM: transformers has no ModelOpt quantizer and its fp8 path refuses
# GPUs below compute capability 8.9, so hf_local takes the BF16 sibling and the
# FP8 keys exist for an OpenAI-compatible endpoint (``backend="api"``).
_NEMOTRON_4B_PATTERN_NOTE = (
    "HYBRID stack: hybrid_override_pattern M-M-M-MM-M-M*-M-M*-M-M-M*-M-M-MM*-MMM-M-M- "
    "(42 layers: 21 Mamba2, 17 MLP, 4 attention) — a forward returns 4 attention "
    "tensors whose positions are not layer numbers; read config.hybrid_override_pattern "
    "to map them back, and never read a rollout as full-depth"
)
_NEMOTRON_H_CAVEATS = (
    "thinking is OFF here: chat_template_kwargs sends enable_thinking=False on "
    "every render (the template defaults to thinking ON)",
    "REMOTE CODE on purpose: transformers' native `nemotron_h` module loaded this "
    "checkpoint but generated nothing except newline tokens (2026-08-21, 5.15.0); the "
    "repo's modeling_nemotron_h.py hard-imports mamba_ssm (gated RMSNorm) and uses "
    "causal_conv1d for its fast path -- prebuilt wheels exist up to torch 2.10, so the "
    "benchmark's nemotron image runs torch 2.10 / transformers 4.57 (see "
    "examples/benchmark/docker/Dockerfile)",
    "hf_local applies two post-load shims (HFLocalModel._apply_family_shims): generate() "
    "must not pre-build a DynamicCache (the repo code builds its hybrid Mamba cache only "
    "when past_key_values is None -- without the shim it recomputes every step, ~2 tok/s) "
    "and generation stops on the tokenizer's <|im_end|> as well as generation_config's </s> "
    "(the template's turn end; without it every answer pads to the cap)",
    "eager attention only: the remote NemotronHForCausalLM has no SDPA dispatch "
    "(transformers raises on attn_implementation=sdpa)",
)
_add(ModelSpec(
    key="nemotron-3-nano-4b", family="nemotron_h", model_type="nemotron_h",
    hf_repo="nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16",
    auto_class="AutoModelForCausalLM", processor_class="AutoTokenizer",
    trust_remote_code=True, min_transformers="4.48.3", is_reasoning=True,
    chat_template_kwargs={"enable_thinking": False},
    attn_semantics=AttnSemantics.HYBRID_SPARSE,
    module_paths=ModulePaths(decoder_layers="backbone.layers", self_attn="mixer", mlp="mixer"),
    caveats=(_NEMOTRON_4B_PATTERN_NOTE,) + _NEMOTRON_H_CAVEATS + (
        "BF16 sibling of nvidia/NVIDIA-Nemotron-3-Nano-4B-FP8 (same weights before "
        "ModelOpt quantization); ~9 GB",
    ),
))
_add(ModelSpec(
    key="nemotron-3-nano-4b-fp8", family="nemotron_h", model_type="nemotron_h",
    hf_repo="nvidia/NVIDIA-Nemotron-3-Nano-4B-FP8",
    auto_class="AutoModelForCausalLM", processor_class="AutoTokenizer",
    trust_remote_code=True, min_transformers="4.48.3", is_reasoning=True,
    chat_template_kwargs={"enable_thinking": False},
    attn_semantics=AttnSemantics.HYBRID_SPARSE,
    module_paths=ModulePaths(decoder_layers="backbone.layers", self_attn="mixer", mlp="mixer"),
    caveats=(
        "ENDPOINT ONLY: ModelOpt FP8 export (hf_quant_config.json, per-tensor fp8 "
        "weight/input scales, fp8 KV cache) — transformers has no ModelOpt quantizer "
        "and refuses fp8 on GPUs below compute capability 8.9 (A6000 = 8.6, A100 = 8.0); "
        "serve it with vLLM and use backend='api'; hf_local loads 'nemotron-3-nano-4b'",
        _NEMOTRON_4B_PATTERN_NOTE,
    ) + _NEMOTRON_H_CAVEATS[:1],
))

# Nemotron 3 Nano Omni (video/audio/image/text in, text out): a 30B-A3B
# NemotronH MoE backbone (52 layers: 23 Mamba2, 23 MoE MLP, 6 attention) behind a
# C-RADIO v4-H vision encoder and a Parakeet speech encoder, all in the repo's
# own code (model_type NemotronH_Nano_Omni_Reasoning_V3, trust_remote_code).
_NEMOTRON_OMNI_CAVEATS = (
    "62 GB BF16: needs device='auto' over at least two 48 GB cards (or one 80 GB); "
    "FP8 (33 GB) and NVFP4 (21 GB) siblings are vLLM-only on Ampere (see the -fp8 key)",
    "same NemotronH generate() shims as nemotron-3-nano-4b (no pre-built DynamicCache, "
    "stop on the tokenizer EOS), applied to the embedded language_model; eager attention only",
    "permanent remote code (modeling.py / processing.py / modeling_nemotron_h.py in "
    "the repo): the language model is an embedded NemotronHForCausalLM at "
    "language_model.backbone.layers, vision at vision_model (RADIO, remote code "
    "from nvidia/C-RADIOv4-H), audio at sound_encoder (Parakeet) + sound_projection",
    "HYBRID MoE stack: hybrid_override_pattern has attention at 6 of 52 positions "
    "('*'), Mamba2 at 'M', routed experts at 'E' — HYBRID_SPARSE semantics",
    "image tokens: InternVL-style dynamic tiling, 256 tokens per 512px tile "
    "((512/16)^2 * 0.5^2) plus a thumbnail tile; token id from "
    "config.img_context_token_id; audio from config.sound_context_token_id at 16 kHz",
    "thinking is OFF here: chat_template_kwargs sends enable_thinking=False (the "
    "template defaults to reasoning ON and supports reasoning_budget)",
    "the generic hf_local encode path (apply_chat_template with typed content blocks, "
    "processor(text=, images=, audio=, sampling_rate=)) has NOT been exercised against "
    "this processor yet — run the examples/benchmark --baseline-only smoke first",
)
_add(ModelSpec(
    key="nemotron-3-nano-omni-30b-a3b-reasoning", family="nemotron_h_omni",
    model_type="NemotronH_Nano_Omni_Reasoning_V3",
    hf_repo="nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16",
    auto_class="AutoModelForCausalLM", processor_class="AutoProcessor",
    trust_remote_code=True, min_transformers="4.57.0", is_moe=True, is_reasoning=True,
    chat_template_kwargs={"enable_thinking": False},
    attn_semantics=AttnSemantics.HYBRID_SPARSE,
    module_paths=ModulePaths(
        decoder_layers="language_model.backbone.layers", self_attn="mixer", mlp="mixer",
        vision_tower="vision_model",
    ),
    vision=VisionSpec(
        image_token_id_attr="img_context_token_id", merge_size_attr=None,
        grid_source="fixed", fixed_tokens_per_tile=256,
    ),
    audio=AudioSpec(
        audio_token_id_attr="sound_context_token_id", audio_tower="sound_encoder",
        use_audio_in_video=False,
    ),
    video=True,
    caveats=_NEMOTRON_OMNI_CAVEATS,
))
_add(ModelSpec(
    key="nemotron-3-nano-omni-30b-a3b-reasoning-fp8", family="nemotron_h_omni",
    model_type="NemotronH_Nano_Omni_Reasoning_V3",
    hf_repo="nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-FP8",
    auto_class="AutoModelForCausalLM", processor_class="AutoProcessor",
    trust_remote_code=True, min_transformers="4.57.0", is_moe=True, is_reasoning=True,
    chat_template_kwargs={"enable_thinking": False},
    attn_semantics=AttnSemantics.HYBRID_SPARSE,
    module_paths=ModulePaths(
        decoder_layers="language_model.backbone.layers", self_attn="mixer", mlp="mixer",
        vision_tower="vision_model",
    ),
    vision=VisionSpec(
        image_token_id_attr="img_context_token_id", merge_size_attr=None,
        grid_source="fixed", fixed_tokens_per_tile=256,
    ),
    audio=AudioSpec(
        audio_token_id_attr="sound_context_token_id", audio_tower="sound_encoder",
        use_audio_in_video=False,
    ),
    video=True,
    caveats=(
        "ENDPOINT ONLY: ModelOpt FP8 export (config quantization_config.quant_method="
        "'modelopt') — transformers cannot load it and refuses fp8 below compute "
        "capability 8.9; serve with vLLM >= 0.20 (vllm[audio]) and use backend='api'; "
        "hf_local loads 'nemotron-3-nano-omni-30b-a3b-reasoning' (BF16)",
    ) + _NEMOTRON_OMNI_CAVEATS[1:3],
))

__all__ = ["REGISTRY", "get_spec", "list_specs"]

# ----------------------------------------------------------------------
# examples/benchmark family: Gemini (Google Gen AI API, closed weights) (2026-08-25)
# ----------------------------------------------------------------------
# ``api_only``: the ``api`` backend with ``gemini_compat.gemini_runtime`` (the
# official google-genai SDK); the spec key IS the API model id (``hf_repo`` is
# empty, ``APIModel`` sends ``spec.hf_repo or spec.key``). Every model below
# lists text, image, video, audio and PDF as inputs on its model card, so one
# spec serves the llm / vlm / alm cells. Thinking: the 3.x models expose
# ``thinking_level`` (3.7-flash bottoms out at ``low``, the rest at
# ``minimal``); the 2.5 models expose ``thinking_budget`` (0 = off on flash /
# flash-lite; 2.5-pro cannot switch it off, floor 128) — the runtime sends the
# floor unless told otherwise. Logprobs: none for 3.x ("working as intended",
# Google forum 2026-08-05) and withdrawn on 2.5, so the backend is GENERATE-only
# and the fix ladder stops at L2 (no internals for L3a/L3b).
for _key, _thinking in (
    ("gemini-3.7-flash", "thinking_level floor 'low' (minimal not offered)"),
    ("gemini-3.6-flash", "thinking_level floor 'minimal'"),
    ("gemini-3.5-flash", "thinking_level floor 'minimal'"),
    ("gemini-3.5-flash-lite", "thinking_level floor 'minimal' (its default)"),
    ("gemini-3.1-flash-lite", "thinking_level floor 'minimal' (levels per the card; floor unverified)"),
    ("gemini-2.5-flash", "thinking_budget 0 turns thinking off (default on)"),
    ("gemini-2.5-flash-lite", "thinking_budget 0 (its default)"),
    ("gemini-2.5-pro", "thinking cannot be disabled (budget floor 128)"),
):
    _add(ModelSpec(
        key=_key, family="gemini", model_type="gemini", hf_repo="", auto_class="",
        api_only=True, attn_semantics=AttnSemantics.NONE, is_reasoning=True,
        vision=VisionSpec(), audio=AudioSpec(), video=True,
        caveats=(
            "closed weights: api backend only (google-genai, GEMINI_API_KEY); no internals "
            "and no logprobs -> GENERATE-only, fix ladder capped at L2",
            f"thinking: {_thinking}; the benchmark runner sends the floor unless --thinking-level",
            "inline media <= 20 MB per request; audio costs 32 tokens/s",
        ),
    ))
