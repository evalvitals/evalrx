"""A random-init, CPU-sized Qwen-VL behind the evalrx ``Model`` contract.

The write-side analyzers (attention knockout, causal tracing, gradient
attribution) act on a live torch module, which ``FakeModel`` does not have.
``TinyVLM`` is the smallest thing that is real where it matters: a genuine
transformers Qwen2.5-VL / Qwen3-VL (vision tower, M-RoPE, merged image tokens,
DeepStack for Qwen3-VL) with a few thousand parameters, fed through a
``forward`` that fills a ``Trace`` the way ``hf_local`` does — image-token
positions in a TokenTypeMap, logits, optional attentions/hidden states, and
gradients when ``CaptureSpec.with_grad`` asks for them.

Images are 16×16 PIL images: 8×8 patches of 2 px, merged 2×2 → 16 image tokens.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
import transformers

from evalrx.core.capability import Capability
from evalrx.core.case import Inputs
from evalrx.core.model import Model, Trace
from evalrx.core.spec import AttnSemantics
from evalrx.core.tokentype import TokenTypeMap

IMAGE_TOKEN, VIDEO_TOKEN, VSTART, VEND = 100, 101, 102, 103
VOCAB = 128
IMG_PX, PATCH, MERGE = 16, 2, 2
GRID = IMG_PX // PATCH          # 8 patches per side
TOKENS_SIDE = GRID // MERGE     # 4 merged tokens per side → 16 image tokens

# These models are tiny; on a many-core box torch's default one-thread-per-core
# turns every matmul into scheduling contention (minutes instead of seconds).
torch.set_num_threads(min(torch.get_num_threads(), 4))


class CharTokenizer:
    """Maps characters onto ids 1..90 — enough for the gold-target path."""

    def __call__(self, text: str, add_special_tokens: bool = False) -> dict:
        return {"input_ids": [ord(c) % 90 + 1 for c in text]}


def _build(family: str) -> Any:
    vision = dict(depth=2, hidden_size=16, intermediate_size=32, num_heads=2, in_channels=3,
                  patch_size=PATCH, spatial_merge_size=MERGE, temporal_patch_size=2,
                  out_hidden_size=32)
    text = dict(vocab_size=VOCAB, hidden_size=32, intermediate_size=64, num_hidden_layers=6,
                num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=512)
    ids = dict(image_token_id=IMAGE_TOKEN, video_token_id=VIDEO_TOKEN,
               vision_start_token_id=VSTART, vision_end_token_id=VEND)
    torch.manual_seed(0)
    if family == "qwen2_5_vl":
        vision.update(window_size=8, fullatt_block_indexes=[1])
        text.update(rope_scaling={"type": "mrope", "mrope_section": [1, 1, 2]})
        cfg = transformers.Qwen2_5_VLConfig(vision_config=vision, text_config=text, **ids)
        cls = transformers.Qwen2_5_VLForConditionalGeneration
    elif family == "qwen3_vl":
        vision.update(num_position_embeddings=64, deepstack_visual_indexes=[0, 1])
        text.update(head_dim=8, rope_scaling={"rope_type": "default", "mrope_section": [1, 1, 2],
                                              "mrope_interleaved": True})
        cfg = transformers.Qwen3VLConfig(vision_config=vision, text_config=text, **ids)
        cls = transformers.Qwen3VLForConditionalGeneration
    else:  # pragma: no cover - test helper misuse
        raise ValueError(family)
    return cls._from_config(cfg, attn_implementation="eager").eval()


def _pixel_values(image: Any) -> torch.Tensor:
    """16×16 RGB → (64 patches, 3·2·2·2) in the processor's merge-window order."""
    arr = np.asarray(image.convert("RGB").resize((IMG_PX, IMG_PX)), dtype=np.float32) / 127.5 - 1.0
    x = torch.from_numpy(arr).permute(2, 0, 1)                          # (3, 16, 16)
    x = x.reshape(3, TOKENS_SIDE, MERGE, PATCH, TOKENS_SIDE, MERGE, PATCH)
    x = x.permute(1, 4, 2, 5, 0, 3, 6)                                  # (gh, gw, mh, mw, c, ph, pw)
    x = x.reshape(GRID * GRID, 3, PATCH, PATCH)
    x = torch.stack([x, x], dim=2)                                      # temporal patch of 2
    return x.reshape(GRID * GRID, 3 * 2 * PATCH * PATCH)


class TinyVLM(Model):
    """Tiny Qwen-VL under the ``Model`` contract (see module docstring)."""

    def __init__(self, family: str = "qwen2_5_vl",
                 attn_semantics: AttnSemantics = AttnSemantics.STANDARD,
                 gradients: bool = True) -> None:
        caps = {Capability.GENERATE, Capability.LOGITS, Capability.ATTENTION,
                Capability.HIDDEN_STATES}
        if gradients:
            caps.add(Capability.GRADIENTS)
        self.capabilities = frozenset(caps)
        self.modalities = frozenset({"text", "image"})
        self.spec = SimpleNamespace(key=f"tiny-{family}", attn_semantics=attn_semantics)
        self._hf = (_build(family), CharTokenizer())
        self.n_forwards = 0

    @property
    def _loaded(self):
        return self._hf

    def generate(self, inputs: Any, **kwargs: Any) -> str:
        return "x"

    def unembed_weight(self):
        return self._hf[0].get_output_embeddings().weight

    def final_norm(self):
        from evalrx.models.backends.hf.discover import get_final_norm

        return get_final_norm(self._hf[0])

    def forward(self, inputs: Any, capture: "set[Capability]", spec: Any = None) -> Trace:
        module = self._hf[0]
        self.n_forwards += 1
        prompt = inputs.prompt if isinstance(inputs, Inputs) else str(inputs)
        image = getattr(inputs, "image", None) if isinstance(inputs, Inputs) else None
        text_ids = [ord(c) % 90 + 1 for c in prompt[:24]] or [1]
        enc: dict = {}
        if image is not None:
            n_img = TOKENS_SIDE * TOKENS_SIDE
            ids = [1, 2, VSTART] + [IMAGE_TOKEN] * n_img + [VEND] + text_ids
            enc["pixel_values"] = _pixel_values(image)
            enc["image_grid_thw"] = torch.tensor([[1, GRID, GRID]])
            image_pos = list(range(3, 3 + n_img))
            grids = [(1, TOKENS_SIDE, TOKENS_SIDE)]
            patches = [(0, 0, r, c) for r in range(TOKENS_SIDE) for c in range(TOKENS_SIDE)]
        else:
            ids = [1, 2] + text_ids
            image_pos, grids, patches = [], [], []
        enc["input_ids"] = torch.tensor([ids])
        with_grad = bool(spec is not None and getattr(spec, "with_grad", False))
        with torch.set_grad_enabled(with_grad):
            out = module(**enc, output_attentions=Capability.ATTENTION in capture,
                         output_hidden_states=Capability.HIDDEN_STATES in capture)
        img_set = set(image_pos)
        ttm = TokenTypeMap(seq_len=len(ids), image_pos=image_pos,
                           text_pos=[i for i in range(len(ids)) if i not in img_set],
                           grids=grids, patches=patches, image_token_id=IMAGE_TOKEN)
        extras: dict = {"attn_semantics": self.spec.attn_semantics.value}
        if image_pos:
            extras["image_token_mask"] = torch.tensor(ids) == IMAGE_TOKEN
            extras["image_spatial_shape"] = (TOKENS_SIDE, TOKENS_SIDE)
        provided = {Capability.LOGITS}
        trace = Trace(tokens=[f"t{i}" for i in ids], token_ids=ids, provided=provided,
                      logits=out.logits[0], token_type_map=ttm, extras=extras)
        if Capability.ATTENTION in capture:
            trace.attentions = [a[0] for a in out.attentions]
            provided.add(Capability.ATTENTION)
        if Capability.HIDDEN_STATES in capture:
            trace.hidden_states = [h[0] for h in out.hidden_states]
            provided.add(Capability.HIDDEN_STATES)
        return trace

    def __repr__(self) -> str:
        return f"TinyVLM({self.spec.key})"


def image(seed: int = 0) -> Any:
    """A deterministic 16×16 RGB test image."""
    from PIL import Image

    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 255, (IMG_PX, IMG_PX, 3), dtype=np.uint8), mode="RGB")
