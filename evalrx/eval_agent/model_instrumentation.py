"""InstrumentedModel — records every call an M1 analyzer makes to the target model.

Before this existed, ``RunLogger`` gave verbatim prompt+response coverage to
every *judge/coder* LLM call (M1 selection, M2 analysis, M3 diagnosis, M4/M5
protocol judge, explore's coder agent) via ``_save_judge_io`` — but the model
actually UNDER EVALUATION is called from inside analyzers too, and those calls
went straight from ``Analyzer._run`` to the model and back with nothing in
between. Most analyzers only keep a derived scalar (e.g. ``selfcheck.py``
generates ``n_samples`` resamples to score self-consistency, then discards the
raw text — only a 200-char fragment of the WORST sentence survives). For
anything that calls ``model.generate/forward/logprobs/chat`` more than once
per case — self-consistency resampling, counterfactual/ablation regeneration,
rollout search, contrastive prompts — the intermediate calls were simply gone.

``InstrumentedModel`` closes that hole *without touching any analyzer*: wrap
the model handed to ``analyzer.run(model, data)`` in this proxy and every call
is durably logged via :meth:`RunLogger.log_model_call`, tagged with
``(cycle, analyzer, method, call_index)`` — the primary key that was missing
before. See ``run_logger.py``'s ``log_model_call`` docstring for where the
record ends up (``model_calls.jsonl``) and how it is mirrored into Langfuse.

Scope — this covers the catalog analyzers that run in-process through
``ProbeAgent._run_direct``, which is most of the fan-out-heavy ones
(self_consistency, selfcheck, coverage_gap, cot_faithfulness,
format_sensitivity, ...). It is a plain forwarding proxy, not a ``Model``
subclass, so ``isinstance(model, SomeBackend)`` checks made OUTSIDE that
narrow window (e.g. ``case_discovery.py``'s API-model concurrency check) are
unaffected — they never see the proxy. NOT covered by this wrapper, as of
this writing:

- Docker-mode black-box analyzers (``ProbeAgent(use_docker=True)``) — the
  model runs inside a subprocess ``_is_blackbox_compatible`` routes to, out of
  this process entirely. Since that routing picks exactly the GENERATE/LOGPROBS
  analyzers this instrumentation targets, ``use_docker=True`` makes it inert
  for the calls it matters most for.
- Tier-(b)/(c) codegen'd probes (``probe_generator.py`` /
  ``whitebox_probe_generator.py``) — a separate execution path from
  ``_run_direct``, called from ``ProbeAgent._maybe_generate``.
- M4 repair-candidate generation (``fix_agent.py``, ``fix_internals.py``,
  ``fix_tools.py``, ``fix_pipeline.py``) — but these already have a durable,
  pre-existing home: per-candidate per-case outputs land in
  ``fixes/<trial>/outputs.jsonl`` via ``RunLogger.log_fix``.
- Pre-loop ``case_discovery.py`` harvesting — its ``model.generate()`` calls
  produce a case's ``observed`` field, which is captured once the case enters
  the loop via ``RunLogger.log_cases``; the call itself (latency, exact
  wording of intermediate attempts) is not.

Each of those is a plausible next analyzer to wrap the same way; this pass
targets the M1 catalog because that is where the derived-score-only,
raw-calls-discarded pattern was found and verified (self_consistency,
selfcheck, cot_faithfulness).
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from evalrx.core.capability import Capability
    from evalrx.core.model import CaptureSpec, Model, Trace
    from evalrx.eval_agent.run_logger import RunLogger

#: Longest single string field kept verbatim before it is truncated. Generous
#: enough for almost every prompt/response; a truncated field says so rather
#: than silently clipping, so a reader never mistakes it for the whole thing.
_MAX_FIELD_CHARS = 8000

#: How many of the batch's case ids get repeated verbatim on EVERY model_call
#: record (see ``batch_case_ids`` in ``_record``). Unlike prompts/outputs this
#: field is identical across every call an analyzer makes in one cycle, so on
#: a real batch (136 cases in the run this instrumentation was built against)
#: it would otherwise duplicate ~5KB of uuids per record, times however many
#: calls the analyzer makes — the one field here that scales with batch size
#: rather than with prompt/response length, so it gets its own cap instead of
#: reusing ``_MAX_FIELD_CHARS``.
_MAX_BATCH_IDS_INLINE = 20


def _truncate(text: str) -> str:
    if len(text) <= _MAX_FIELD_CHARS:
        return text
    return text[:_MAX_FIELD_CHARS] + f"...[truncated, {len(text)} chars total]"


def _capped_ids(ids: "list[str]") -> "tuple[list[str], int]":
    """First ``_MAX_BATCH_IDS_INLINE`` ids, plus the true total count."""
    return ids[:_MAX_BATCH_IDS_INLINE], len(ids)


def _snapshot_inputs(inputs: Any) -> Any:
    """A JSON-safe stand-in for whatever an analyzer passed as *inputs*.

    Handles the three shapes analyzers actually use: a bare string (some
    analyzers build their own prompt, e.g. ``verbalized_conf.py``), an
    ``Inputs``-like object (``.prompt`` + optional ``.image``/``.audio``/
    ``.video``), or anything else (falls back to ``str()``). Media slots are
    never inlined — a decoded ``PIL.Image``/waveform is far too large and a
    path/URL is recorded as-is, the same convention ``RunLogger.log_cases``
    uses for baseline case media.
    """
    if isinstance(inputs, str):
        return _truncate(inputs)
    prompt = getattr(inputs, "prompt", None)
    if prompt is None:
        return _truncate(str(inputs))
    snapshot: dict[str, Any] = {"prompt": _truncate(str(prompt))}
    for kind in ("image", "audio", "video"):
        value = getattr(inputs, kind, None)
        if value is None:
            continue
        snapshot[kind] = str(value) if isinstance(value, (str, Path)) else f"<{type(value).__name__}>"
    return snapshot


def _snapshot_trace(trace: "Trace") -> "dict[str, Any]":
    """Token-level summary of a ``forward()`` result — never the raw tensors.

    The heavy arrays (attentions/hidden_states/logits) already have a durable
    home: whichever ones an analyzer chooses to return via ``Result.artifacts``
    are saved by ``RunLogger._save_probe_artifacts``. Duplicating them here
    would multiply disk use for no new evidence, so this keeps only what a
    reader needs to tell one ``forward()`` call apart from another.
    """
    return {
        "tokens": list(trace.tokens),
        "provided": sorted(c.value for c in trace.provided),
    }


class InstrumentedModel:
    """Forwarding proxy around a ``Model`` that reports every call to a ``RunLogger``.

    One instance per analyzer invocation — construct fresh in
    ``ProbeAgent._run_direct`` rather than sharing across analyzers, so
    ``call_index`` is scoped to exactly the (cycle, analyzer) pair it is
    tagged with. Safe to use from a ``ThreadPoolExecutor`` worker: each
    instance is only ever touched by the one thread running that analyzer,
    and the underlying JSONL sink (``RunLogger``'s dedicated file handler)
    is itself thread-safe.
    """

    def __init__(
        self,
        model: "Model",
        run_logger: "RunLogger",
        *,
        cycle: int,
        analyzer: str,
        case_prompts: "dict[str, str] | None" = None,
        batch_case_ids: "list[str] | None" = None,
    ) -> None:
        self._model = model
        self._run_logger = run_logger
        self._cycle = cycle
        self._analyzer = analyzer
        self._call_index = 0
        self._lock = threading.Lock()
        # Best-effort (prompt -> case_id) for this analyzer's batch, so a call
        # can usually be tied back to the case that produced it — the query a
        # UI actually wants ("show me every call for case X"). Exact-match
        # only: an analyzer that rewrites the prompt (format_sensitivity,
        # cot_faithfulness, ...) won't hit this map, which is why every record
        # also carries batch_case_ids — an unmatched call is at least scoped
        # to a known, bounded set of cases instead of floating free.
        self._case_prompts = case_prompts or {}
        # Computed once (not per call): identical on every record this
        # instance produces, so there is no benefit to redoing it each time.
        self._batch_case_ids_inline, self._n_batch_cases = _capped_ids(list(batch_case_ids or []))

    # Delegate everything not explicitly instrumented below — capabilities,
    # modalities, supports(), unembed_weight(), and any backend-specific
    # extras an analyzer reads off the model directly.
    def __getattr__(self, name: str) -> Any:
        return getattr(self._model, name)

    def __repr__(self) -> str:
        # Experiment.fingerprint() and every analyzer's Result(model=repr(model),
        # ...) depend on this matching the wrapped model's repr exactly — a
        # proxy-identity repr would fragment the experiment cache (a fresh
        # proxy is minted per call) and mislabel every Result.
        return repr(self._model)

    def _next_call_index(self) -> int:
        with self._lock:
            self._call_index += 1
            return self._call_index

    def _record(
        self, method: str, inputs: Any, kwargs: "dict[str, Any] | None",
        output: Any, duration_sec: float, error: "str | None",
    ) -> None:
        prompt = inputs if isinstance(inputs, str) else getattr(inputs, "prompt", None)
        case_id = self._case_prompts.get(str(prompt)) if prompt is not None else None
        self._run_logger.log_model_call(
            cycle=self._cycle,
            analyzer=self._analyzer,
            call_index=self._next_call_index(),
            method=method,
            case_id=case_id,
            batch_case_ids=self._batch_case_ids_inline,
            n_batch_cases=self._n_batch_cases,
            inputs=_snapshot_inputs(inputs),
            kwargs={k: str(v) for k, v in (kwargs or {}).items()},
            output=output,
            duration_sec=duration_sec,
            error=error,
        )

    def generate(self, inputs: Any, **kwargs: Any) -> str:
        t0 = time.monotonic()
        try:
            out = self._model.generate(inputs, **kwargs)
        except Exception as exc:  # noqa: BLE001 - record the failure, then re-raise it
            self._record("generate", inputs, kwargs, None, time.monotonic() - t0, repr(exc))
            raise
        self._record("generate", inputs, kwargs, _truncate(str(out)), time.monotonic() - t0, None)
        return out

    def forward(
        self, inputs: Any, capture: "set[Capability]", spec: "CaptureSpec | None" = None,
    ) -> "Trace":
        t0 = time.monotonic()
        meta = {"capture": sorted(c.value for c in capture)}
        try:
            trace = self._model.forward(inputs, capture, spec)
        except Exception as exc:  # noqa: BLE001
            self._record("forward", inputs, meta, None, time.monotonic() - t0, repr(exc))
            raise
        self._record("forward", inputs, meta, _snapshot_trace(trace), time.monotonic() - t0, None)
        return trace

    def logprobs(self, inputs: Any, **kwargs: Any) -> "list[Any]":
        t0 = time.monotonic()
        try:
            toks = self._model.logprobs(inputs, **kwargs)
        except Exception as exc:  # noqa: BLE001
            self._record("logprobs", inputs, kwargs, None, time.monotonic() - t0, repr(exc))
            raise
        summary = [{"token": t.token, "logprob": round(t.logprob, 4)} for t in toks]
        self._record("logprobs", inputs, kwargs, summary, time.monotonic() - t0, None)
        return toks

    def chat(self, messages: list, tools: "list | None" = None) -> Any:
        t0 = time.monotonic()
        try:
            turn = self._model.chat(messages, tools)
        except Exception as exc:  # noqa: BLE001
            self._record("chat", messages, {"tools": tools}, None, time.monotonic() - t0, repr(exc))
            raise
        output = {"text": _truncate(str(getattr(turn, "text", turn)))}
        raw_tool_calls = getattr(turn, "raw_tool_calls", None)
        if raw_tool_calls:
            output["tool_calls"] = str(raw_tool_calls)
        self._record("chat", messages, {"tools": tools}, output, time.monotonic() - t0, None)
        return turn
