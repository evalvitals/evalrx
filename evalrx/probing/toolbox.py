"""Call a probe by name with JSON arguments.

:func:`build_probe` turns ``(name, params)`` into a constructed analyzer after
checking the arguments against the probe's schema, and :class:`ProbeToolbox`
binds a model and a batch so each probe becomes a
:class:`~evalrx.core.tool.Tool` whose function runs it::

    box = ProbeToolbox(model, cases)
    box.describe()["attention_knockout"]["available"]
    result = box.call("logit_lens", {"top_k": 5})
    tools = box.tools()            # OpenAI / HF tool schemas, ready for an agent

A bad call fails before any model work with a message that names the accepted
arguments, so an LLM caller can correct itself on the next turn.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Iterable, Mapping

from evalrx.core.registry import registry
from evalrx.core.tool import Tool, ToolResult
from evalrx.probing.catalog import (
    ProbeParam,
    ProbeToolSpec,
    availability,
    describe_probes,
    probe_specs,
    spec_for,
)
from evalrx.probing.router import ProbeRoute, ProbeRouter

if TYPE_CHECKING:
    from evalrx.core.analyzer import Analyzer
    from evalrx.core.model import Model
    from evalrx.core.result import Result


class ProbeCallError(ValueError):
    """A probe call that cannot be executed as written.

    ``code`` is one of ``unknown_probe``, ``bad_params``, ``unavailable``.
    """

    def __init__(self, message: str, code: str = "bad_params") -> None:
        super().__init__(message)
        self.code = code


def _coerce(param: ProbeParam, value: Any, probe: str) -> Any:
    """Check *value* against the declared JSON type; widen int to float only."""
    kind = param.json_type
    if value is None:
        if param.nullable or kind is None:
            return None
        raise ProbeCallError(f"{probe}: '{param.name}' may not be null (expected {kind})")
    if kind is None:
        return value
    ok = {
        "boolean": isinstance(value, bool),
        "integer": isinstance(value, int) and not isinstance(value, bool),
        "number": isinstance(value, (int, float)) and not isinstance(value, bool),
        "string": isinstance(value, str),
        "array": isinstance(value, (list, tuple)),
        "object": isinstance(value, dict),
    }[kind]
    if not ok and kind == "integer" and isinstance(value, float) and value.is_integer():
        return int(value)  # JSON has one number type; 3.0 for an integer field is 3
    if not ok:
        raise ProbeCallError(
            f"{probe}: '{param.name}' expects {kind}, got {type(value).__name__} ({value!r})"
        )
    if kind == "number":
        return float(value) if isinstance(param.default, float) else value
    if kind == "array" and isinstance(param.default, tuple):
        return tuple(value)
    return value


def check_params(spec: ProbeToolSpec, params: "Mapping[str, Any] | None") -> dict[str, Any]:
    """Validate JSON arguments for *spec*; return them ready for the constructor."""
    params = dict(params or {})
    accepted = {p.name: p for p in spec.callable_params}
    unknown = sorted(set(params) - set(accepted))
    if unknown:
        injected = [n for n in unknown if n in {p.name for p in spec.params}]
        hint = (f" ({', '.join(injected)} must be supplied by the host, not in the call)"
                if injected else "")
        raise ProbeCallError(
            f"{spec.name}: unknown argument(s) {unknown}{hint}; "
            f"accepted: {sorted(accepted) or 'none'}"
        )
    lacking = [p.name for p in accepted.values() if p.required and p.name not in params]
    if lacking:
        raise ProbeCallError(f"{spec.name}: missing required argument(s) {lacking}")
    return {name: _coerce(accepted[name], value, spec.name) for name, value in params.items()}


def build_probe(name: str, params: "Mapping[str, Any] | None" = None, *,
                overrides: "Mapping[str, Analyzer] | None" = None) -> "Analyzer":
    """Construct the analyzer for one probe call.

    An entry in *overrides* supplies arguments a call cannot carry (a judge
    model, a weight edit). With no *params* that instance is used as is; with
    *params* a new instance of the same class is built from the override's
    parameters updated by the call, so the override itself is never mutated.
    """
    if not registry.analyzers.has(name):
        raise ProbeCallError(
            f"unknown probe {name!r}; registered: {registry.analyzers.list()}", "unknown_probe"
        )
    spec = spec_for(name)
    cleaned = check_params(spec, params)
    override = (overrides or {}).get(name)
    try:
        if override is not None:
            if not cleaned:
                return override
            return type(override)(**{**override.get_params(), **cleaned})
        if spec.injected_params:
            raise ProbeCallError(
                f"{name}: the host must supply {list(spec.injected_params)} "
                "(pass a constructed analyzer via analyzer_overrides)", "unavailable",
            )
        return spec.analyzer_cls(**cleaned)
    except ProbeCallError:
        raise
    except (TypeError, ValueError) as exc:
        raise ProbeCallError(f"{name}: rejected its arguments: {exc}") from exc


class ProbeToolbox:
    """Probes bound to one model and (optionally) one batch.

    Args:
        model:     The model under test.
        data:      Default batch for :meth:`call`; also used for availability.
        overrides: ``{name: analyzer}`` for probes that need host-supplied
                   constructor arguments.
        max_cases: When positive, every call sees at most this many cases
                   (a stratified head of the batch).
    """

    def __init__(self, model: "Model | None", data: Any = None, *,
                 overrides: "Mapping[str, Analyzer] | None" = None,
                 max_cases: int = 0) -> None:
        self.model = model
        self.data = data
        self.overrides: dict[str, "Analyzer"] = dict(overrides or {})
        self.max_cases = int(max_cases or 0)

    # -- discovery ------------------------------------------------------
    def specs(self) -> dict[str, ProbeToolSpec]:
        return probe_specs()

    def describe(self, families: "Iterable[str] | None" = None) -> dict[str, dict[str, Any]]:
        """Every probe with its availability here (see :func:`describe_probes`)."""
        return describe_probes(self.model, self.data, overrides=self.overrides,
                               families=families)

    def available(self) -> list[str]:
        return sorted(n for n, row in self.describe().items() if row["available"])

    def route(self, **request: Any) -> ProbeRoute:
        """Rank runnable probes for a request (see :meth:`ProbeRouter.route`)."""
        return ProbeRouter(overrides=self.overrides).route(self.model, self.data, **request)

    # -- execution ------------------------------------------------------
    def call(self, name: str, params: "Mapping[str, Any] | None" = None,
             data: Any = None) -> "Result":
        """Run probe *name* with JSON *params* on *data* (default: the bound batch)."""
        batch = self.data if data is None else data
        if batch is None:
            raise ProbeCallError(f"{name}: no data to probe", "unavailable")
        analyzer = build_probe(name, params, overrides=self.overrides)
        verdict = availability(spec_for(name), self.model, batch, overrides=self.overrides)
        if not verdict.available:
            raise ProbeCallError(f"{name} cannot run here: {'; '.join(verdict.missing)}",
                                 "unavailable")
        return analyzer.run(self.model, self._capped(batch))

    def tools(self, names: "Iterable[str] | None" = None, *,
              only_available: bool = True) -> list[Tool]:
        """Probes as :class:`~evalrx.core.tool.Tool` objects bound to this toolbox."""
        report = self.describe()
        chosen = list(names) if names is not None else sorted(report)
        out: list[Tool] = []
        for name in chosen:
            if name not in report:
                raise ProbeCallError(f"unknown probe {name!r}", "unknown_probe")
            if only_available and not report[name]["available"]:
                continue
            out.append(spec_for(name).to_tool(fn=self._tool_fn(name)))
        return out

    # ------------------------------------------------------------------
    def _tool_fn(self, name: str):
        def run(**params: Any) -> ToolResult:
            result = self.call(name, params)
            return ToolResult(text=summarize_result(result), meta={"result": result})

        run.__name__ = name
        return run

    def _capped(self, batch: Any) -> Any:
        if self.max_cases <= 0:
            return batch
        head = getattr(batch, "stratified_head", None)
        try:
            if callable(head) and len(batch) > self.max_cases:
                return head(self.max_cases)
        except TypeError:
            pass
        return batch


def summarize_result(result: "Result", max_chars: int = 1500) -> str:
    """Scalar findings of a result as compact text (no per-case rows)."""
    findings = getattr(result, "findings", None) or {}
    scalars = {k: v for k, v in findings.items()
               if isinstance(v, (int, float, bool, str)) and not str(k).startswith("_")}
    n_rows = len(findings.get("per_case", []) or [])
    text = f"[{getattr(result, 'analyzer', '?')}] {scalars}"
    if n_rows:
        text += f" ({n_rows} per-case rows)"
    return text if len(text) <= max_chars else text[: max_chars - 1] + "…"
