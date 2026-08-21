"""Agent-composed report contracts for the dynamic EvalVitals UI.

The evidence model is deterministic and task agnostic.  A Report Agent may
choose how to arrange a small, typed component catalog, but every value shown
by the UI is resolved from :class:`ReportData`; the model cannot invent data or
arbitrary executable UI.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

REPORT_DATA_VERSION = 1
REPORT_SCHEMA_VERSION = 1
JSON_RENDER_VERSION = "0.19.0"
CATALOG_VERSION = "evalvitals-report@1"

ALLOWED_COMPONENTS = frozenset(
    {
        "ReportPage",
        "SettingHero",
        "MetricStrip",
        "Journey",
        "FindingGrid",
        "ChartGrid",
        "OutcomeCard",
        "CasePreview",
        "EvidenceIndex",
    }
)
REQUIRED_COMPONENTS = frozenset({"SettingHero", "Journey", "OutcomeCard"})
MAX_ELEMENTS = 40
MAX_DEPTH = 6


@dataclass(frozen=True)
class PublishedReport:
    """Paths and identity of a cached dynamic report publication."""

    data_path: Path
    spec_path: Path
    trace_id: str
    generated_by: str
    sha256: str


class ReportSpecError(ValueError):
    """Raised when a Report Agent emits an unsafe or invalid layout."""


def build_report_data(
    run_dir: str | Path,
    *,
    example_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Build the renderer-neutral data model for any completed EvalVitals run."""
    root = Path(run_dir).resolve()
    from evalvitals.reporting.html_report import extract_run_data

    resolved_example = Path(example_dir).resolve() if example_dir else _infer_example_dir(root)
    raw = extract_run_data(root, resolved_example)
    events = _read_events(root)
    case_events = [event.get("case") for event in events if event.get("event") == "case_record"]
    cases = [case for case in case_events if isinstance(case, dict)] or list(raw.get("cases") or [])
    run = dict(raw.get("run") or {})
    trace_id = str((events[-1] if events else {}).get("trace_id") or root.name)
    reader = dict(raw.get("reader_report") or {})

    stages = _stages(raw)
    findings = _findings(reader, raw)
    charts = _charts(run, raw)
    repairs = _repairs(raw)
    media = _media_index(cases)
    normalized_cases = [_normalise_case(case, media) for case in cases if isinstance(case, dict)]

    return _json_safe(
        {
            "version": REPORT_DATA_VERSION,
            "trace_id": trace_id,
            "source_event_seq": max((int(e.get("event_seq") or 0) for e in events), default=0),
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "setting": {
                "model": run.get("model") or "Target model",
                "dataset": run.get("benchmark_name") or "Evaluation dataset",
                "question": reader.get("question") or "Why did the model fail, and can we fix it?",
                "protocol": run.get("protocol") or "",
                "n_cases": int(run.get("n_cases") or len(normalized_cases)),
            },
            "summary": {
                "headline": reader.get("headline") or "Failure analysis completed",
                "answer": reader.get("answer") or "Inspect the evidence journey below.",
                "confidence": reader.get("confidence") or "unknown",
                "stopped_by": run.get("stopped_by") or "completed",
            },
            "metrics": _metrics(run, raw, normalized_cases),
            "stages": stages,
            "findings": findings,
            "charts": charts,
            "repairs": repairs,
            "cases": normalized_cases,
            "media": media,
            "debug": {
                "event_count": len(events),
                "events": [_debug_event(event) for event in events],
            },
        }
    )


class ReportAgent:
    """Ask an injected text model to compose a json-render layout.

    The model sees a compact evidence summary and a list of legal identifiers,
    not raw media or full debug logs.  One repair attempt is made for malformed
    output; callers should use :func:`fallback_spec` if both attempts fail.
    """

    def __init__(self, model: Any, *, catalog_version: str = CATALOG_VERSION) -> None:
        if not hasattr(model, "generate"):
            raise TypeError("ReportAgent model must expose generate(prompt) -> str")
        self.model = model
        self.catalog_version = catalog_version

    def compose(self, data: Mapping[str, Any]) -> dict[str, Any]:
        prompt = self._prompt(data)
        raw = str(self.model.generate(prompt))
        try:
            return validate_spec(_parse_json(raw), data=data)
        except (ReportSpecError, json.JSONDecodeError, TypeError, ValueError) as first:
            repair = (
                "Your previous report layout was invalid. Return ONLY corrected JSON.\n"
                f"Validation error: {first}\n\nOriginal output:\n{raw[:12000]}\n\n{prompt}"
            )
            return validate_spec(_parse_json(str(self.model.generate(repair))), data=data)

    def _prompt(self, data: Mapping[str, Any]) -> str:
        compact = {
            "setting": data.get("setting"),
            "summary": data.get("summary"),
            "metrics": data.get("metrics"),
            "stages": data.get("stages"),
            "finding_ids": [item.get("id") for item in data.get("findings", [])],
            "chart_ids": [item.get("id") for item in data.get("charts", [])],
            "repair_ids": [item.get("id") for item in data.get("repairs", [])],
            "case_ids": [item.get("id") for item in data.get("cases", [])[:8]],
        }
        return f"""You are the Report Agent for EvalVitals. Compose a concise visual report
showing the journey: model fails on a dataset -> M1 probes behavior -> M2 screens
associations -> M3 proposes mechanisms -> M5 tests on held-out evidence -> M4 repairs
the model. A passer-by must understand the setting and outcome without knowing EvalVitals.

Return ONLY a json-render tree with shape {{"root":"id","elements":{{...}}}}.
Allowed component types: {', '.join(sorted(ALLOWED_COMPONENTS))}.
Required exactly once or more: SettingHero, Journey, OutcomeCard.
ReportPage may have children. Other elements use props only.
Legal props are data references such as {{"findingIds":["finding-1"]}}; never copy,
rewrite, or invent evidence text/numbers. Prefer charts and journey graphics over prose.
At most 3 findings, 2 charts, {MAX_ELEMENTS} total elements, depth {MAX_DEPTH}.
Catalog: {self.catalog_version}.

Available report data identifiers:
{json.dumps(compact, ensure_ascii=False, separators=(',', ':'))}
"""


def fallback_spec(data: Mapping[str, Any]) -> dict[str, Any]:
    """Deterministic, evidence-first layout used without or after a failed model."""
    finding_ids = [str(item.get("id")) for item in data.get("findings", [])[:3]]
    chart_ids = [str(item.get("id")) for item in data.get("charts", [])[:2]]
    case_ids = [str(item.get("id")) for item in data.get("cases", [])[:4]]
    children = ["setting", "metrics", "journey"]
    elements: dict[str, Any] = {
        "page": {"type": "ReportPage", "props": {}, "children": children},
        "setting": {"type": "SettingHero", "props": {}},
        "metrics": {"type": "MetricStrip", "props": {}},
        "journey": {"type": "Journey", "props": {}},
        "outcome": {"type": "OutcomeCard", "props": {}},
        "evidence": {"type": "EvidenceIndex", "props": {}},
    }
    if finding_ids:
        elements["findings"] = {"type": "FindingGrid", "props": {"findingIds": finding_ids}}
        children.append("findings")
    if chart_ids:
        elements["charts"] = {"type": "ChartGrid", "props": {"chartIds": chart_ids}}
        children.append("charts")
    children.append("outcome")
    if case_ids:
        elements["cases"] = {"type": "CasePreview", "props": {"caseIds": case_ids}}
        children.append("cases")
    children.append("evidence")
    return elements and {"root": "page", "elements": elements}


def validate_spec(spec: Any, *, data: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Validate the closed json-render catalog and bounded tree shape."""
    if not isinstance(spec, dict) or not isinstance(spec.get("root"), str):
        raise ReportSpecError("spec needs string root and elements")
    elements = spec.get("elements")
    if not isinstance(elements, dict) or spec["root"] not in elements:
        raise ReportSpecError("root must reference an element")
    if len(elements) > MAX_ELEMENTS:
        raise ReportSpecError(f"at most {MAX_ELEMENTS} elements are allowed")
    seen_types: list[str] = []
    for element_id, element in elements.items():
        if not isinstance(element_id, str) or not isinstance(element, dict):
            raise ReportSpecError("elements must be named objects")
        component = element.get("type")
        if component not in ALLOWED_COMPONENTS:
            raise ReportSpecError(f"component {component!r} is not in the catalog")
        seen_types.append(component)
        props = element.get("props", {})
        if not isinstance(props, dict):
            raise ReportSpecError(f"{element_id}.props must be an object")
        _validate_props(props)
        allowed_props = {
            "FindingGrid": {"findingIds"}, "ChartGrid": {"chartIds"},
            "CasePreview": {"caseIds"},
        }.get(str(component), set())
        unknown_props = set(props).difference(allowed_props)
        if unknown_props:
            raise ReportSpecError(f"{element_id} has unknown props: {', '.join(sorted(unknown_props))}")
        if component == "FindingGrid" and len(props.get("findingIds") or []) > 3:
            raise ReportSpecError("FindingGrid accepts at most 3 findings")
        if component == "ChartGrid" and len(props.get("chartIds") or []) > 2:
            raise ReportSpecError("ChartGrid accepts at most 2 charts")
        if data is not None:
            _validate_data_refs(component, props, data)
        children = element.get("children", [])
        if children is not None and (
            not isinstance(children, list) or not all(isinstance(child, str) for child in children)
        ):
            raise ReportSpecError(f"{element_id}.children must be element ids")
        for child in children or []:
            if child not in elements:
                raise ReportSpecError(f"unknown child {child!r}")
    missing = REQUIRED_COMPONENTS.difference(seen_types)
    if missing:
        raise ReportSpecError(f"missing required components: {', '.join(sorted(missing))}")
    _check_tree_depth(spec["root"], elements, set(), 1)
    return {"root": spec["root"], "elements": elements}


def publish_report(
    run_dir: str | Path,
    *,
    example_dir: str | Path | None = None,
    model: Any | None = None,
    run_logger: Any | None = None,
) -> PublishedReport:
    """Compile, compose, validate, cache, and optionally log a completed report."""
    root = Path(run_dir).resolve()
    data = build_report_data(root, example_dir=example_dir)
    mode = "deterministic"
    model_name: str | None = None
    spec = fallback_spec(data)
    if model is not None:
        model_name = getattr(model, "model_name", None) or getattr(model, "name", None) or type(model).__name__
        try:
            spec = ReportAgent(model).compose(data)
            mode = "agent"
        except Exception:
            mode = "fallback"
            spec = fallback_spec(data)
    spec = validate_spec(spec, data=data)
    canonical_spec = json.dumps(spec, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(canonical_spec.encode("utf-8")).hexdigest()
    envelope = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "format": "json-render",
        "json_render_version": JSON_RENDER_VERSION,
        "catalog_version": CATALOG_VERSION,
        "trace_id": data["trace_id"],
        "source_event_seq": data["source_event_seq"],
        "generated_by": {"mode": mode, "model": model_name},
        "spec": spec,
        "sha256": digest,
    }
    report_dir = root / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    data_path = report_dir / "report_data.json"
    spec_path = report_dir / "report_spec.json"
    data_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    spec_path.write_text(json.dumps(envelope, ensure_ascii=False, indent=2), encoding="utf-8")
    if run_logger is not None and hasattr(run_logger, "log_report_published"):
        run_logger.log_report_published(envelope)
    return PublishedReport(data_path, spec_path, str(data["trace_id"]), mode, digest)


def load_published_report(run_dir: str | Path) -> tuple[dict[str, Any], dict[str, Any]]:
    root = Path(run_dir).resolve() / "report"
    return (
        json.loads((root / "report_data.json").read_text(encoding="utf-8")),
        json.loads((root / "report_spec.json").read_text(encoding="utf-8")),
    )


def report_is_current(run_dir: str | Path) -> bool:
    """Whether the cached layout covers every non-publication run event."""
    root = Path(run_dir).resolve()
    try:
        _, envelope = load_published_report(root)
        cached_seq = int(envelope.get("source_event_seq") or 0)
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return False
    evidence_seq = max(
        (
            int(event.get("event_seq") or 0)
            for event in _read_events(root)
            if event.get("event") != "report_published"
        ),
        default=0,
    )
    return cached_seq >= evidence_seq


def _stages(raw: Mapping[str, Any]) -> list[dict[str, Any]]:
    definitions = [
        ("M1", "Probe", "Measure where successful and failed cases behave differently", raw.get("m1"), "results"),
        ("M2", "Analyze", "Screen the measurements for reliable failure-linked patterns", raw.get("m2"), "stats"),
        ("M3", "Hypothesize", "Turn the strongest evidence into falsifiable mechanisms", raw.get("m3"), "hypotheses"),
        ("M5", "Verify", "Challenge the mechanisms on independent evidence", raw.get("m5"), "results"),
        ("M4", "Repair", "Apply a targeted intervention and compare before vs. after", raw.get("m4_fix"), "selection"),
    ]
    result = []
    for code, title, purpose, payload, evidence_key in definitions:
        stage = dict(payload or {})
        evidence = stage.get(evidence_key) or []
        ran = bool(stage.get("ran")) if "ran" in stage else bool(evidence or stage.get("duration"))
        status = "completed" if ran else "not-run"
        if code == "M4" and ran:
            status = "improved" if stage.get("fixed") else "no-improvement"
        result.append({
            "id": code.lower(), "code": code, "title": title, "purpose": purpose,
            "status": status, "evidence_count": len(evidence) if isinstance(evidence, list) else 1,
        })
    return result


def _findings(reader: Mapping[str, Any], raw: Mapping[str, Any]) -> list[dict[str, Any]]:
    result = []
    for index, finding in enumerate(reader.get("key_findings") or []):
        if isinstance(finding, dict):
            result.append({"id": f"finding-{index + 1}", **finding})
    if result:
        return result
    hypotheses = (raw.get("m3") or {}).get("hypotheses") or []
    for index, hypothesis in enumerate(hypotheses[:3]):
        if isinstance(hypothesis, dict):
            result.append({
                "id": f"finding-{index + 1}",
                "title": hypothesis.get("failure_mode") or "Candidate mechanism",
                "summary": hypothesis.get("plain_statement") or hypothesis.get("statement") or "",
                "evidence_level": hypothesis.get("status") or "Proposed",
                "limitation": "See M5 for independent verification.",
            })
    return result


def _charts(run: Mapping[str, Any], raw: Mapping[str, Any]) -> list[dict[str, Any]]:
    charts: list[dict[str, Any]] = []
    labels = run.get("label_distribution") or {}
    if isinstance(labels, dict) and labels:
        charts.append({
            "id": "outcomes", "kind": "donut", "title": "Evaluation outcomes",
            "series": [{"label": str(k).title(), "value": v} for k, v in labels.items() if isinstance(v, (int, float))],
        })
    stats = (raw.get("m2") or {}).get("stats") or []
    effect_rows = []
    for item in stats[:8]:
        if isinstance(item, dict) and isinstance(item.get("effect"), (int, float)):
            effect_rows.append({"label": str(item.get("tool") or "signal"), "value": item["effect"], "highlight": bool(item.get("reject"))})
    if effect_rows:
        charts.append({"id": "effects", "kind": "bar", "title": "Strongest measured associations", "series": effect_rows})
    return charts


def _repairs(raw: Mapping[str, Any]) -> list[dict[str, Any]]:
    fix = raw.get("m4_fix") or {}
    if not fix.get("ran"):
        return []
    confirm = fix.get("confirm") or {}
    return [{
        "id": "repair-1", "fixed": bool(fix.get("fixed")),
        "title": (fix.get("best") or {}).get("name") or confirm.get("name") or "Targeted repair",
        "effect": confirm.get("effect"),
        "fixed_cases": len(confirm.get("fixed_cases") or []),
        "broken_cases": len(confirm.get("broken_cases") or []),
    }]


def _metrics(run: Mapping[str, Any], raw: Mapping[str, Any], cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    labels = run.get("label_distribution") or {}
    failed = labels.get("fail", labels.get("FAIL", 0)) if isinstance(labels, dict) else 0
    verified = len((raw.get("m5") or {}).get("results") or [])
    fixed = sum(1 for case in cases if case.get("status") == "fixed")
    return [
        {"id": "evaluated", "label": "Cases evaluated", "value": int(run.get("n_cases") or len(cases))},
        {"id": "failed", "label": "Initial failures", "value": int(failed or 0)},
        {"id": "verified", "label": "Mechanisms checked", "value": verified},
        {"id": "fixed", "label": "Cases repaired", "value": fixed},
    ]


def _media_index(cases: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    media: list[dict[str, Any]] = []
    seen: set[str] = set()
    for case in cases:
        inputs = case.get("inputs") if isinstance(case.get("inputs"), dict) else case
        for kind in ("audio", "image", "video"):
            path = inputs.get(kind) or inputs.get(f"{kind}_path")
            if not isinstance(path, str) or not path or path in seen or path.startswith("<"):
                continue
            seen.add(path)
            media.append({"id": f"media-{len(media) + 1}", "kind": kind, "path": path})
    return media


def _normalise_case(case: Mapping[str, Any], media: list[dict[str, Any]]) -> dict[str, Any]:
    inputs = case.get("inputs") if isinstance(case.get("inputs"), dict) else {}
    prompt = inputs.get("prompt") or case.get("instruction") or case.get("prompt") or ""
    media_by_path = {item["path"]: item["id"] for item in media}
    media_ids = []
    for kind in ("audio", "image", "video"):
        value = inputs.get(kind) or case.get(f"{kind}_path") or case.get(kind)
        if isinstance(value, str) and value in media_by_path:
            media_ids.append(media_by_path[value])
    return {
        "id": str(case.get("id") or case.get("case_id") or f"case-{id(case)}"),
        "status": str(case.get("status") or case.get("label") or "unknown").lower(),
        "prompt": prompt,
        "expected": case.get("expected"),
        "observed": case.get("observed", case.get("output")),
        "choices": case.get("choices") or [],
        "tags": case.get("tags") or case.get("probe_flags") or [],
        "task": case.get("task") or (case.get("metadata") or {}).get("category") or "",
        "media_ids": media_ids,
        "trajectory": case.get("trajectory"),
    }


def _read_events(root: Path) -> list[dict[str, Any]]:
    candidates = [root / "run_log.jsonl", *sorted(root.glob("logs*/run_log.jsonl"))]
    path = next((candidate for candidate in candidates if candidate.exists()), None)
    if path is None:
        return []
    events = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            events.append(value)
    if events:
        trace = next((event.get("trace_id") for event in reversed(events) if event.get("event") == "run_start"), None)
        if trace and sum(event.get("trace_id") == trace for event in events) > 1:
            events = [event for event in events if event.get("trace_id") == trace]
    return events


def _infer_example_dir(root: Path) -> Path | None:
    """Find a nearby legacy manifest; v4 case events do not need this join."""
    for candidate in (root, root.parent, root.parent.parent):
        if any((candidate / "data").glob("*.jsonl")) or any((candidate / "data").glob("*.json")):
            return candidate
    return None


def _debug_event(event: Mapping[str, Any]) -> dict[str, Any]:
    event_type = str(event.get("event") or "")
    inferred_stage = {
        "probe": "M1", "explore": "M2", "analysis": "M2", "diagnosis": "M3",
        "surgery": "M5" if event.get("module") == "m5" else "M4",
        "experiment": "M4", "fix": "M4", "case_record": "DATA",
        "report_published": "REPORT",
    }.get(event_type, "RUN")
    return {
        "event": event.get("event"), "stage": event.get("stage") or inferred_stage, "cycle": event.get("cycle"),
        "event_seq": event.get("event_seq"), "ts": event.get("ts"), "span_id": event.get("span_id"),
        "summary": _event_summary(event),
    }


def _event_summary(event: Mapping[str, Any]) -> str:
    for key in ("conclusion", "narrative", "selection_rationale", "status", "stopped_by"):
        value = event.get(key)
        if value not in (None, ""):
            return str(value)[:280]
    return "Recorded pipeline event"


def _validate_props(value: Any, *, depth: int = 0) -> None:
    if depth > 4:
        raise ReportSpecError("props are too deeply nested")
    if isinstance(value, str) and len(value) > 200:
        raise ReportSpecError("layout strings may not exceed 200 characters")
    if isinstance(value, dict):
        for child in value.values():
            _validate_props(child, depth=depth + 1)
    elif isinstance(value, list):
        if len(value) > 20:
            raise ReportSpecError("layout lists may not exceed 20 items")
        for child in value:
            _validate_props(child, depth=depth + 1)
    elif value is not None and not isinstance(value, (str, bool, int, float)):
        raise ReportSpecError("props must be JSON scalar/list/object values")


def _validate_data_refs(component: str, props: Mapping[str, Any], data: Mapping[str, Any]) -> None:
    mapping = {
        "FindingGrid": ("findingIds", "findings"),
        "ChartGrid": ("chartIds", "charts"),
        "CasePreview": ("caseIds", "cases"),
    }
    if component not in mapping:
        return
    prop, collection = mapping[component]
    refs = props.get(prop) or []
    if not isinstance(refs, list) or not all(isinstance(ref, str) for ref in refs):
        raise ReportSpecError(f"{prop} must be a list of strings")
    legal = {str(item.get("id")) for item in data.get(collection, []) if isinstance(item, dict)}
    unknown = set(refs).difference(legal)
    if unknown:
        raise ReportSpecError(f"unknown {collection} ids: {', '.join(sorted(unknown))}")


def _check_tree_depth(node: str, elements: Mapping[str, Any], stack: set[str], depth: int) -> None:
    if depth > MAX_DEPTH:
        raise ReportSpecError(f"tree depth exceeds {MAX_DEPTH}")
    if node in stack:
        raise ReportSpecError("element tree contains a cycle")
    next_stack = {*stack, node}
    for child in elements[node].get("children") or []:
        _check_tree_depth(child, elements, next_stack, depth + 1)


def _parse_json(text: str) -> Any:
    stripped = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", stripped, re.DOTALL)
    return json.loads(fenced.group(1) if fenced else stripped)


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "value"):
        return _json_safe(value.value)
    return str(value)
