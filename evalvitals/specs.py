"""Model spec registry — declarative ModelSpec entries, one per family.

A plain dict (no import-side-effect decorator).  Module paths are HINTS only;
the hf_local backend discovers the real decoder-layer ModuleList at load time
(see :mod:`evalvitals.models._discover`).  Token ids / merge sizes are read from
the live config via the attribute NAMES in ``VisionSpec`` — never baked as values
(GLM-4.5V 151363 vs GLM-4.1V 151343 is exactly why).

Paths reflect the verified post-VLM-refactor transformers layout (single
``.model``; Kimi/Llama4 have no outer ``.model``).  ``tool_calling=True`` marks
instruct/thinking checkpoints whose chat template renders tools (grants
TOOL_CALLS on the local backend; verified against the template at load time).
This module is torch-free.
"""

from __future__ import annotations

from evalvitals.core.spec import AttnSemantics, AudioSpec, ModelSpec, ModulePaths, VisionSpec

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
    audio=AudioSpec(audio_token_id_attr="audio_token_id", audio_tower="audio_tower"),
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

__all__ = ["REGISTRY", "get_spec", "list_specs"]
