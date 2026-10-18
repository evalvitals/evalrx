"""One media-flow edit across every kind of decoder layer.

"Cut (``weight=0``) or scale what the text reads from the media tokens" means a
different operation per token mixer:

* softmax attention — the attention tap (:mod:`.attn_tap`): ``-inf`` on the
  query→media scores, or ``+log w`` (renormalised scaling);
* gated-delta-rule linear attention — :mod:`.linear_mix`: blend the layer's
  output toward its re-run on the sequence without the media tokens.

:func:`media_flow_edit` applies both in one block, so an analyzer edits every
layer of a hybrid stack (Qwen3.5) the same way it edits a dense one, and
:class:`FidelityGate` checks both mechanisms on every case before anything is
reported (:func:`require_edit_fidelity` is the one-input form).
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Iterable, Iterator

from evalrx.analyzers._internals.access import decoder_stack, torch_module
from evalrx.analyzers._internals.attn_tap import (
    FidelityError,
    attention_tap,
    check_fidelity,
    knockout,
    reweight,
)
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


class FidelityGate:
    """Every case an analyzer reports on passes the edit-free self-checks, or it refuses.

    Before each case's measurement, :meth:`check` runs the tap check on the
    softmax layers and, with ``linear=True``, the media-absent re-run check on
    the linear layers, on THAT case's input. The first failing case raises
    :class:`FidelityError` carrying the record so far, and the analyzer reports
    nothing: keeping only the cases that happen to pass would select on
    numerical noise (bf16 ``sdpa`` vs the eager-equivalent tap sits around the
    tolerance and varies by input). :meth:`summary` is the analyzer's
    ``tap_fidelity`` finding: the worst case per mechanism plus every check.

    ``linear=False`` for edits that touch softmax layers only (temperature,
    relevance recording).
    """

    def __init__(self, model: "Model", *, rtol: float = 1e-2, analyzer: str = "media edit",
                 linear: bool = True) -> None:
        stack = decoder_stack(torch_module(model, analyzer))
        self.model, self.rtol, self.analyzer = model, rtol, analyzer
        self.n_softmax, self.n_linear = stack.n_softmax, stack.n_linear
        self.check_linear = bool(linear and stack.n_linear)
        self.checks: list[dict[str, Any]] = []
        self._tap_meta: dict[str, Any] = {}

    def check(self, case_id: Any, inputs: Any, media: Iterable[int] = (), *,
              base: Any = None) -> "dict[str, Any]":
        """Check one case; *base* = its native logits if the caller already has them."""
        record: dict[str, Any] = {"case": str(case_id)}
        if self.n_softmax:
            tap = check_fidelity(self.model, inputs, rtol=self.rtol, analyzer=self.analyzer, base=base)
            self._tap_meta = {k: tap[k] for k in ("tap_layers_expected", "tap_rtol",
                                                  "native_attn_implementation")}
            record.update({k: tap[k] for k in ("tap_faithful", "tap_layers_engaged", "tap_max_logit_diff",
                                                "tap_rel_diff", "tap_argmax_same")})
        if self.check_linear:
            lin = check_linear_fidelity(self.model, inputs, media, rtol=self.rtol, analyzer=self.analyzer)
            record.update({k: lin[k] for k in ("linear_faithful", "linear_layers_engaged",
                                                "linear_prefix_max_rel_diff")})
        record["faithful"] = bool(record.get("tap_faithful", True) and record.get("linear_faithful", True))
        self.checks.append(record)
        if not record["faithful"]:
            raise FidelityError(
                f"{self.analyzer}: the edit-free self-check failed on case {record['case']} "
                f"(check {len(self.checks)}; {record}); refusing to report edited-attention effects "
                "for this batch. tap: engaged < expected layers means the architecture bypasses the "
                "transformers attention interface, a logit diff means its attention is not "
                "eager-equivalent (bf16 sdpa often is not: use attn_impl='eager'); linear: the "
                "media-absent re-run must engage every linear layer and reproduce the outputs "
                "before the first media token.",
                self.summary(),
            )
        return record

    def summary(self) -> "dict[str, Any]":
        out: dict[str, Any] = {
            "checked": "every analysed case",
            "n_checked": len(self.checks),
            "softmax_layers": self.n_softmax,
            "linear_layers": self.n_linear,
            "faithful": all(c["faithful"] for c in self.checks),
        }
        taps = [c for c in self.checks if "tap_faithful" in c]
        if self.n_softmax:
            worst = max(taps, key=lambda c: c["tap_rel_diff"], default=None)
            out.update({
                "tap_faithful": all(c["tap_faithful"] for c in taps),
                "tap_layers_engaged": min((c["tap_layers_engaged"] for c in taps), default=0),
                "tap_max_logit_diff": max((c["tap_max_logit_diff"] for c in taps), default=None),
                "tap_max_rel_diff": worst["tap_rel_diff"] if worst else None,
                "tap_worst_case": worst["case"] if worst else None,
                "tap_n_argmax_changed": sum(not c["tap_argmax_same"] for c in taps),
                **self._tap_meta,
            })
        lins = [c for c in self.checks if "linear_faithful" in c]
        if self.check_linear:
            worst = max(lins, key=lambda c: c["linear_prefix_max_rel_diff"], default=None)
            out.update({
                "linear_faithful": all(c["linear_faithful"] for c in lins),
                "linear_layers_engaged": min((c["linear_layers_engaged"] for c in lins), default=0),
                "linear_layers_expected": self.n_linear,
                "linear_prefix_max_rel_diff": worst["linear_prefix_max_rel_diff"] if worst else None,
                "linear_worst_case": worst["case"] if worst else None,
                "linear_rtol": self.rtol,
            })
        failed = [c["case"] for c in self.checks if not c["faithful"]]
        if failed:
            out["failed_case"] = failed[0]
        out["cases"] = list(self.checks)
        return out


def require_edit_fidelity(model: "Model", inputs: Any, media: Iterable[int], *,
                          rtol: float = 1e-2, analyzer: str = "media edit") -> "dict[str, Any]":
    """Verify every edit mechanism the stack needs on one input; raise when one cannot be trusted.

    Analyzers that report on a batch use :class:`FidelityGate`, which runs
    this check on every case rather than on the first one.
    """
    stack = decoder_stack(torch_module(model, analyzer))
    verdict: dict[str, Any] = {"softmax_layers": stack.n_softmax, "linear_layers": stack.n_linear}
    if stack.n_softmax:
        verdict.update(check_fidelity(model, inputs, rtol=rtol, analyzer=analyzer))
    if stack.n_linear:
        verdict.update(check_linear_fidelity(model, inputs, media, rtol=rtol, analyzer=analyzer))
    verdict["faithful"] = bool(verdict.get("tap_faithful", True) and verdict.get("linear_faithful", True))
    if not verdict["faithful"]:
        raise FidelityError(
            f"{analyzer}: the attention edits cannot be trusted on this model ({verdict}). "
            "tap: engaged < expected layers means the architecture bypasses the transformers "
            "attention interface, a logit diff means its attention is not eager-equivalent; "
            "linear: the media-absent re-run must engage every linear layer and reproduce "
            "the outputs before the first media token.",
            verdict,
        )
    return verdict
