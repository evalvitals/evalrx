"""ModelSpec — backend-orthogonal, per-model identity knowledge.

The key separation this module enforces:

  * **capabilities come from the BACKEND** (api / vllm_offline / hf_local),
  * **model identity comes from the SPEC** (this file).

One ``ModelSpec`` for, say, Qwen3-VL drives an API call, a vLLM batch-scoring
run, *and* an HF-eager attention capture — only the capability set differs.
``compose(spec, backend)`` (see :mod:`evalvitals.models.compose`) combines the
two into a :class:`~evalvitals.core.model.Model`.

This module is intentionally **dependency-free** (no torch / transformers), so it
imports on the light, pure-API install.  Module *paths* here are HINTS only: the
white-box backend RESOLVES them at load time against ``named_modules()`` rather
than trusting them, because they drift across transformers point-releases and
per-checkpoint wrappers (the doubled-``.model.`` / Llama4-no-``.model`` / v5
fused-experts traps).  Likewise token ids and merge sizes are read from the live
``config`` / processor, never hard-coded as values.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class AttnSemantics(str, Enum):
    """What an attention capture actually MEANS for this architecture.

    Prevents an attention analyzer from blindly assuming a dense ``[B,H,Q,K]``
    token-attention matrix when the model uses a compressed/fused attention.
    """

    STANDARD = "standard"      # eager yields a real [B, H, Q, K] token matrix
    MLA_LATENT = "mla_latent"  # DeepSeek / Kimi: materialised in decompressed latent head space
    NONE = "none"              # no usable weights (e.g. fused-only vision tower)
    UNVERIFIED = "unverified"  # e.g. Step3 MFA — probe tensor shape before trusting
    #: Hybrid stacks (Qwen3.5, Qwen3-Next): most layers are linear/SSM attention
    #: with NO query-key matrix, and only every k-th layer is full attention.
    #: Each captured tensor is a real [H, Q, K], but the returned list is SHORTER
    #: than the model's depth and its positions are not layer numbers — read
    #: ``config.text_config.layer_types`` to map them back. Anything that
    #: composes across the stack (attention rollout) is measuring a partial path.
    HYBRID_SPARSE = "hybrid_sparse"


@dataclass(frozen=True)
class ModulePaths:
    """Dotted-attribute HINTS for hooking, relative to the loaded top object.

    ``"{i}"`` marks the decoder-layer index.  These are *starting points* for
    runtime resolution, not ground truth — the hf_local backend discovers the
    real decoder-layer ``ModuleList`` and derives the rest relative to it.
    """

    decoder_layers: Optional[str] = None   # hint, e.g. "model.language_model.layers"
    self_attn: str = "self_attn"           # submodule within a decoder layer
    mlp: str = "mlp"                        # "mlp" | "feed_forward"; dense vs MoE differ
    vision_tower: Optional[str] = None      # VLM only; hint, e.g. "model.visual"
    vision_blocks: Optional[str] = None     # e.g. ".encoder.blocks" — varies a lot per tower
    router: Optional[str] = None            # MoE routing submodule (relative to layer)
    experts: Optional[str] = None           # MoE experts; ModuleList OR fused param (v5)


@dataclass(frozen=True)
class VisionSpec:
    """VLM-only knowledge for locating image tokens and rebuilding the patch grid.

    Every field is a HINT or a *config attribute name* — never a baked value —
    because image-token ids differ across checkpoints (GLM-4.5V 151363 vs
    GLM-4.1V 151343) and configs name them differently (``image_token_id`` /
    ``image_token_index`` / ``media_placeholder_token_id``).
    """

    # name of the config attribute holding the per-patch placeholder token id:
    image_token_id_attr: str = "image_token_id"
    # name of the config path holding the spatial merge size (None -> not applicable):
    merge_size_attr: Optional[str] = "vision_config.spatial_merge_size"
    grid_source: str = "grid_thw"           # "grid_thw" | "grid_hw" | "fixed"
    fixed_tokens_per_tile: Optional[int] = None
    prefer_mm_token_type_ids: bool = True    # cleanest source of truth when emitted


@dataclass(frozen=True)
class AudioSpec:
    """Omni-only knowledge for locating audio tokens (Stage-2 white-box capture).

    Symmetric to :class:`VisionSpec`: every field is a HINT or a *config attribute
    name*, never a baked value, because audio placeholder ids and tower paths drift
    across checkpoints (Qwen3-Omni's ``thinker.audio_tower`` vs others).  Its mere
    presence on a spec declares the ``"audio"`` modality; the white-box audio-token
    map that mirrors :class:`~evalvitals.core.tokentype.TokenTypeMap` is Stage-2.
    """

    # name of the config attribute holding the per-frame audio placeholder token id:
    audio_token_id_attr: str = "audio_token_id"
    audio_tower: Optional[str] = None        # hooking HINT, e.g. "thinker.audio_tower"
    # Qwen3-Omni: whether a video's own audio track is fed alongside its frames
    # (kept consistent between preprocessing and generation by the processor).
    use_audio_in_video: bool = True


@dataclass(frozen=True)
class ModelSpec:
    """Per-model identity, reused across every backend.

    Holds NO capabilities — those are the backend's.  Modality is declared by the
    *components present*, not a class fork: ``vision`` adds ``"image"``, ``audio``
    adds ``"audio"``, ``video`` adds ``"video"`` (an omni model simply carries more
    than one).  Pure-LLM specs leave them unset (no TokenTypeMap).  Closed models
    set ``api_only``.
    """

    key: str                                  # registry key, e.g. "qwen3-vl-8b-instruct"
    family: str                               # "qwen3_vl" | "deepseek_v3" | ...
    model_type: str                           # config.model_type (used to branch)
    hf_repo: str = ""                         # "" for closed/api-only models
    auto_class: str = "AutoModelForCausalLM"  # or "AutoModelForImageTextToText"
    trust_remote_code: bool = False
    min_transformers: str = "4.51.0"
    processor_class: str = "AutoProcessor"    # "AutoTokenizer" for text-only
    chat_template_kwargs: dict = field(default_factory=dict)
    is_moe: bool = False
    is_reasoning: bool = False                # emits <think>...</think>
    tool_calling: bool = False                # chat template renders tools -> grants TOOL_CALLS on local backends
    eager_required_for_attn: bool = True
    attn_semantics: AttnSemantics = AttnSemantics.STANDARD
    module_paths: Optional[ModulePaths] = None  # None -> rely fully on discovery
    vision: Optional[VisionSpec] = None        # not None -> "image" modality
    audio: Optional[AudioSpec] = None          # not None -> "audio" modality (omni)
    video: bool = False                         # True -> "video" modality (omni)
    api_only: bool = False                      # closed weights -> only the api backend
    caveats: tuple[str, ...] = ()               # honest, human-readable gotchas

    @property
    def is_vlm(self) -> bool:
        """Carries a vision tower — drives the image-token forward path."""
        return self.vision is not None

    @property
    def is_omni(self) -> bool:
        """Handles audio and/or video on top of text (and usually image)."""
        return self.audio is not None or self.video

    @property
    def needs_multimodal_encode(self) -> bool:
        """True when the backend must route through the (image/video/audio) encoder.

        Distinct from ``is_vlm`` alone so an audio-only spec (``vision is
        None``, e.g. the Qwen3-Omni captioner or a Qwen2-Audio spec) still
        takes the multimodal encode path in the backend instead of falling
        through to the plain-text tokenizer and silently dropping its audio.
        """
        return self.vision is not None or self.audio is not None

    @property
    def modalities(self) -> frozenset[str]:
        """Modalities derived from the components present (analyzers match on this)."""
        mods = {"text"}
        if self.vision is not None:
            mods.add("image")
        if self.audio is not None:
            mods.add("audio")
        if self.video:
            mods.add("video")
        return frozenset(mods)

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        kind = "Omni" if self.is_omni else ("VLM" if self.is_vlm else "LLM")
        tag = " api-only" if self.api_only else ""
        return f"ModelSpec({self.key!r}, {kind}, family={self.family}{tag})"
