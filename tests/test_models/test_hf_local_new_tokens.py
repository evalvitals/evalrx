"""``_new_tokens`` slices the continuation out of a ``generate()`` output.

Normally ``out`` is the prompt concatenated with the continuation, so the new
tokens start after the prompt. A model whose own ``generate()`` converts the
input to ``inputs_embeds`` internally (Qwen3-Omni and other omni/audio
architectures whose thinker merges audio/vision embeddings before the LM)
never hands ``input_ids`` back to ``GenerationMixin``, which then cannot
prepend prompt token ids it was never given -- it returns ONLY the new
tokens. Every ``HFLocalModel`` call site used to slice at the prompt's
LENGTH unconditionally, so on this class of model every single generation
decoded to "" (seen live on Qwen3-Omni-30B-A3B-Instruct/MMAU, 128/128 cases,
2026-08-27).

A length check alone is ambiguous at the edges (a concatenated output with
zero new tokens vs. an inputs_embeds-only output that happens to be exactly
as long as the prompt), so the fix compares the actual leading token ids.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from evalrx.models.backends.hf_local import _new_tokens  # noqa: E402


def test_concatenated_output_slices_off_the_prompt():
    input_ids = torch.tensor([1, 2, 3, 4, 5])
    out = torch.cat([input_ids, torch.tensor([6, 7, 8])])
    assert _new_tokens(out, input_ids).tolist() == [6, 7, 8]


def test_inputs_embeds_only_output_is_returned_unsliced():
    # out is ONLY the continuation (3 tokens) -- shorter than the 5-token
    # prompt it was generated from, because generate() never received
    # input_ids to prepend. Slicing at len(input_ids) here is exactly the
    # bug: it cuts past the end of a 3-length tensor and silently returns
    # empty. The prefix doesn't match input_ids at all, which is the signal
    # that distinguishes this case from the one above.
    input_ids = torch.tensor([1, 2, 3, 4, 5])
    out = torch.tensor([6, 7, 8])
    assert _new_tokens(out, input_ids).tolist() == [6, 7, 8]


def test_inputs_embeds_only_output_exactly_as_long_as_the_prompt():
    # The length-only heuristic this replaced would misread this as "zero
    # new tokens" (out.shape[0] == len(input_ids), not >). The prefix still
    # doesn't match input_ids, so it's still read correctly as unsliced.
    input_ids = torch.tensor([1, 2, 3])
    out = torch.tensor([9, 9, 9])
    assert _new_tokens(out, input_ids).tolist() == [9, 9, 9]


def test_concatenated_output_with_zero_new_tokens_is_empty():
    # The length-only heuristic this replaced would misread this as
    # inputs_embeds-only (out.shape[0] == len(input_ids), not >) and return
    # the whole prompt instead of nothing. The prefix DOES match input_ids
    # here, so the fix correctly slices to empty.
    input_ids = torch.tensor([1, 2, 3])
    out = torch.tensor([1, 2, 3])
    assert _new_tokens(out, input_ids).tolist() == []
