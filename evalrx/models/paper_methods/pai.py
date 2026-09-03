"""PAI decoding executor from *Paying More Attention to Image*.

PAI's released LLaVA route increases the pre-softmax attention score from the
currently generated token to image tokens in a contiguous layer interval. The
paired classifier-free-guidance branch uses a separate image-free KV cache.
"""

from __future__ import annotations

import types
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any


@contextmanager
def image_attention_boost(
    model: Any,
    *,
    image_start: int,
    image_end: int,
    alpha: float = 0.2,
    start_layer: int = 2,
    end_layer: int = 32,
) -> Iterator[None]:
    """Temporarily apply PAI's released pre-softmax image-attention formula.

    ``image_end`` is exclusive.  The patch targets only LLaMA eager attention;
    callers must restore it after generation so a candidate cannot leak into a
    later baseline or a different repair attempt.
    """
    import torch
    import torch.nn.functional as F
    from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv

    decoder = getattr(model, "model", model)
    all_layers = decoder.layers
    layers = all_layers[start_layer : min(end_layer, len(all_layers))]
    original_forwards = [layer.self_attn.forward for layer in layers]

    def pai_forward(
        attention: Any,
        hidden_states: Any,
        position_embeddings: tuple[Any, Any],
        attention_mask: Any = None,
        past_key_values: Any = None,
        cache_position: Any = None,
        **kwargs: Any,
    ) -> tuple[Any, Any]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, attention.head_dim)
        query_states = attention.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = attention.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = attention.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(
                key_states, value_states, attention.layer_idx, cache_kwargs
            )
        keys = repeat_kv(key_states, attention.num_key_value_groups)
        values = repeat_kv(value_states, attention.num_key_value_groups)
        weights = torch.matmul(query_states, keys.transpose(2, 3)) * attention.scaling
        if attention_mask is not None:
            weights = weights + attention_mask[:, :, :, : keys.shape[-2]]
        # PAI (attention.py): a <- a + alpha * abs(a), at the final query
        # position and the image token interval, before softmax.  Its CFG
        # processor makes an image-free language-model call using this same
        # temporary patch; source PAI disables the boost for that call.
        stop = min(int(image_end), weights.shape[-1])
        if not getattr(attention, "_evalrx_pai_suspend", False) and int(image_start) < stop:
            target = weights[:, :, -1, int(image_start) : stop]
            weights[:, :, -1, int(image_start) : stop] = target + target.abs() * float(alpha)
        weights = F.softmax(weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        weights = F.dropout(
            weights,
            p=0.0 if not attention.training else attention.attention_dropout,
            training=attention.training,
        )
        output = torch.matmul(weights, values).transpose(1, 2).contiguous()
        output = output.reshape(*input_shape, -1).contiguous()
        output = attention.o_proj(output)
        return output, weights if kwargs.get("output_attentions", False) else None

    try:
        for layer in layers:
            layer.self_attn.forward = types.MethodType(pai_forward, layer.self_attn)
        yield
    finally:
        for layer, original in zip(layers, original_forwards):
            layer.self_attn.forward = original


class PAICFGLogitsProcessor:
    """Released PAI classifier-free-guidance decoding for a LLaMA LM.

    PAI derives an unconditional cache from the same prompt with the image
    token span removed.  It combines conditional and unconditional log
    probabilities at every generation step, while temporarily suppressing the
    image-attention boost in the unconditional pass.  This mirrors
    ``CFG.py`` in the authors' release, without persisting a second model.
    """

    def __init__(
        self,
        model: Any,
        unconditional_input_ids: Any,
        *,
        guidance_scale: float = 2.0,
        start_layer: int = 2,
        end_layer: int = 32,
    ) -> None:
        # HF LLaVA stores a bare ``LlamaModel`` in ``language_model`` and the
        # final LM head on the outer wrapper.  Use the wrapper for CFG logits,
        # but locate the inner decoder layers to suspend PAI's attention boost.
        self.model = model
        self.unconditional_input_ids = unconditional_input_ids
        self.guidance_scale = float(guidance_scale)
        language_model = getattr(model, "language_model", model)
        decoder = getattr(language_model, "model", language_model)
        self.layers = decoder.layers[start_layer : min(end_layer, len(decoder.layers))]
        self._output: Any = None

    @contextmanager
    def _without_attention_boost(self) -> Iterator[None]:
        previous = [
            getattr(layer.self_attn, "_evalrx_pai_suspend", False) for layer in self.layers
        ]
        try:
            for layer in self.layers:
                layer.self_attn._evalrx_pai_suspend = True
            yield
        finally:
            for layer, value in zip(self.layers, previous):
                layer.self_attn._evalrx_pai_suspend = value

    def __call__(self, input_ids: Any, scores: Any) -> Any:
        import math

        import torch
        import torch.nn.functional as F

        conditional = F.log_softmax(scores, dim=-1)
        if self.guidance_scale == 1.0:
            return conditional
        with self._without_attention_boost(), torch.no_grad():
            if self._output is None:
                self._output = self.model(
                    input_ids=self.unconditional_input_ids,
                    use_cache=True,
                    return_dict=True,
                )
            else:
                self._output = self.model(
                    input_ids=input_ids[:, -1:],
                    use_cache=True,
                    past_key_values=self._output.past_key_values,
                    return_dict=True,
                )
        unconditional = F.log_softmax(self._output.logits[:, -1, :], dim=-1)
        cutoff = conditional.max(dim=-1, keepdim=True).values + math.log(0.1)
        guided = self.guidance_scale * (conditional - unconditional) + unconditional
        return guided.masked_fill(conditional < cutoff, -float("inf"))
