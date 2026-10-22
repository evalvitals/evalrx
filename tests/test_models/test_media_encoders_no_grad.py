"""A gradient pass keeps no graph through the media encoders.

``CaptureSpec(with_grad=True)`` is for differentiating the logit margin towards decoder
activations. The vision tower's attention over a large image, kept for backward, is tens
of GB (a 918x1811 chart on Qwen3.5-2B: 6400 patches, 75 GB, out of memory), and no decoder
analyzer needs it: the encoders run without a graph and their outputs enter the decoder as
constants. ``CaptureSpec(media_grad=True)`` keeps the old behaviour.
"""

from __future__ import annotations

import inspect

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("PIL")

from evalrx.models.backends.hf._util import (  # noqa: E402
    media_encoders,
    media_encoders_without_grad,
)
from tests.tiny_whitebox import (  # noqa: E402
    GRID,
    IMAGE_TOKEN,
    TOKENS_SIDE,
    VEND,
    VSTART,
    _build,
    _pixel_values,
    image,
)


def _inputs():
    ids = [1, 2, VSTART] + [IMAGE_TOKEN] * TOKENS_SIDE * TOKENS_SIDE + [VEND] + [5, 6, 7]   # tokens after the merge
    enc = {"input_ids": torch.tensor([ids]), "pixel_values": _pixel_values(image(0)),
           "image_grid_thw": torch.tensor([[1, GRID, GRID]])}
    if "mm_token_type_ids" in inspect.signature(_build("qwen2_5_vl").forward).parameters:
        enc["mm_token_type_ids"] = (enc["input_ids"] == IMAGE_TOKEN).long()
    return enc


def _visual_output_requires_grad(module, enc) -> "tuple[bool, bool]":
    """``(the vision tower's output requires grad, the logits do)`` for one forward."""
    seen = []
    (name, visual), = media_encoders(module)
    handle = visual.register_forward_hook(lambda m, i, out: seen.append(bool(getattr(out, "requires_grad", False))))
    try:
        with torch.set_grad_enabled(True):
            out = module(**enc)
    finally:
        handle.remove()
    return seen[0], bool(out.logits.requires_grad)


def test_the_vision_tower_is_found_once_and_a_text_model_has_none():
    module = _build("qwen2_5_vl")
    assert [name for name, _ in media_encoders(module)] == ["model.visual"]
    assert [name for name, _ in media_encoders(module, hint="model.visual")] == ["model.visual"]      # not twice
    assert [name for name, _ in media_encoders(module, hint="model.no_such_tower")] == ["model.visual"]
    assert media_encoders(_build("qwen3_next")) == []


def test_with_grad_keeps_the_decoder_graph_but_not_the_encoders():
    module, enc = _build("qwen2_5_vl"), _inputs()
    assert _visual_output_requires_grad(module, enc) == (True, True)       # plain: the whole graph
    with media_encoders_without_grad(module) as guard:
        assert guard.names == ["model.visual"]
        assert _visual_output_requires_grad(module, enc) == (False, True)  # encoder constant, decoder live
    assert "forward" not in module.model.visual.__dict__                   # the class method is back
    assert _visual_output_requires_grad(module, enc) == (True, True)


def test_the_encoder_forward_is_restored_after_an_error_and_its_values_do_not_change():
    module, enc = _build("qwen2_5_vl"), _inputs()
    with torch.no_grad():
        plain = module(**enc).logits
    with pytest.raises(RuntimeError, match="boom"):
        with media_encoders_without_grad(module):
            raise RuntimeError("boom")
    assert "forward" not in module.model.visual.__dict__
    with media_encoders_without_grad(module), torch.set_grad_enabled(True):
        wrapped = module(**enc).logits
    assert torch.allclose(plain, wrapped.detach())


def test_the_local_backend_runs_its_encoders_without_grad_unless_asked():
    from evalrx.core.model import CaptureSpec
    from evalrx.models.backends.hf.model import HFLocalModel

    source = inspect.getsource(HFLocalModel.forward)
    assert "set_grad_enabled(with_grad)" in source and "media_encoders_without_grad(model" in source
    assert "media_grad" in source and CaptureSpec().media_grad is False
