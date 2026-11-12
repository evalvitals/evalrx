"""One media-flow edit across every kind of decoder layer.

"Cut (``weight=0``) or scale what the text reads from the media tokens" means a
different operation per token mixer:

* softmax attention — the attention tap (:mod:`.attn_tap`): ``-inf`` on the
  query→media scores, or ``+log w`` (renormalised scaling);
* gated-delta-rule linear attention — :mod:`.linear_mix`: blend the layer's
  output toward its re-run on the sequence without the media tokens.

:func:`media_flow_edit` applies both in one block, so an analyzer edits every
layer of a hybrid stack (Qwen3.5) the same way it edits a dense one, and
:func:`require_edit_fidelity` checks both mechanisms before anything is reported.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Iterable, Iterator

from evalrx.analyzers._internals.access import decoder_stack, torch_module
from evalrx.analyzers._internals.attn_tap import attention_tap, check_fidelity, knockout, reweight
from evalrx.analyzers._internals.linear_mix import check_linear_fidelity, linear_media_edit

if TYPE_CHECKING:
    from evalrx.core.model import Model


@contextmanager
def media_flow_edit(
    model: "Model",
    media: Iterable[int],
    *,
    queries: "Iterable[int] | None" = None,
    layers: "Iterable[int] | None" = None,
    weight: float = 0.0,
    analyzer: str = "media edit",
) -> Iterator[None]:
    """Cut (``weight=0``) or scale what *queries* read from *media* in *layers*.

    ``layers=None`` = every layer. Softmax layers scale renormalised (rows still
    sum to one); linear layers scale the media's contribution to their output.
    """
    stack = decoder_stack(torch_module(model, analyzer))
    media = [int(m) for m in media]
    queries = None if queries is None else [int(q) for q in queries]
    chosen = range(stack.n_layers) if layers is None else [int(i) for i in layers]
    softmax = [i for i in chosen if stack.attn[i] is not None]
    linear = [i for i in chosen if stack.linear[i] is not None]
    edits = []
    if softmax:
        edit = (knockout(media, queries=queries, layers=softmax) if weight == 0
                else reweight(media, weight, queries=queries, layers=softmax))
        edits.append(edit)
    with attention_tap(model, edits, analyzer=analyzer), \
            linear_media_edit(model, media, queries=queries, layers=linear, weight=weight,
                              analyzer=analyzer):
        yield


def require_edit_fidelity(model: "Model", inputs: Any, media: Iterable[int], *,
                          rtol: float = 1e-2, analyzer: str = "media edit") -> "dict[str, Any]":
    """Verify every edit mechanism the stack needs; raise when one cannot be trusted."""
    stack = decoder_stack(torch_module(model, analyzer))
    verdict: dict[str, Any] = {"softmax_layers": stack.n_softmax, "linear_layers": stack.n_linear}
    if stack.n_softmax:
        verdict.update(check_fidelity(model, inputs, rtol=rtol, analyzer=analyzer))
    if stack.n_linear:
        verdict.update(check_linear_fidelity(model, inputs, media, rtol=rtol, analyzer=analyzer))
    verdict["faithful"] = bool(verdict.get("tap_faithful", True) and verdict.get("linear_faithful", True))
    if not verdict["faithful"]:
        raise RuntimeError(
            f"{analyzer}: the attention edits cannot be trusted on this model ({verdict}). "
            "tap: engaged < expected layers means the architecture bypasses the transformers "
            "attention interface, a logit diff means its attention is not eager-equivalent; "
            "linear: the media-absent re-run must engage every linear layer and reproduce "
            "the outputs before the first media token."
        )
    return verdict
