"""Agent-composed report contracts for the dynamic EvalRX UI.

The evidence model is deterministic and task agnostic.  A Report Agent may
choose how to arrange a small, typed component catalog, but every value shown
by the UI is resolved from :class:`ReportData`; the model cannot invent data or
arbitrary executable UI.

:func:`build_report_data` assembles that data from
:mod:`~evalrx.reporting.report_inputs` (what the run recorded),
:mod:`~evalrx.reporting.stage_views` (the per-stage views, with
:mod:`~evalrx.reporting.stage_figures` and :mod:`~evalrx.reporting.stage_examples`)
and :mod:`~evalrx.reporting.report_summary` (the top-line reading).
"""

from __future__ import annotations

import base64
import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from evalrx.reporting.report_inputs import (
    _assign_partitions,
    _contract_model_name,
    _contract_payloads,
    _media_index,
    _normalise_case,
    _repair_outcomes,
)
from evalrx.reporting.report_summary import _charts, _explore_metrics, _findings, _metrics, _repairs
from evalrx.reporting.stage_views import _stage_detail, _stages

# 21: `setting.finished` / `setting.last_event_at`, for a run still being
#     written (the log's `run_end`, schema 7).
# 20: `setting.kind` (an explore output renders as its own report) and the
#     explore agent's analysis on `stage_detail.m2.agent`.
# 19: the M5 confirmation joins the winning candidate a v2 run recorded, and
#     the "Cases repaired" tile reads it.
# 18: `setting.partitions` + `case.split` — which band of the frozen batch
#     each case sat in (explore / held-out / confirm), so the Case Studio can
#     show the split the run was actually measured on.
# 17: the case-study sheet - the whole run as one page.
# 16: `setting.hero_image` — the run's own cover figure, embedded.
# 15: `setting.diagnosed_by` — which agent drove the run.
# 14: M3 hypotheses recover the plain sentence from proposed_hypotheses.
# 13: paired M2 rows carry the strategy pair they compared, so several tests
#     that share one tool no longer share one label. Bumping this is what
#     makes an already-published run pick the change up — `report_is_current`
#     only tracks new EVENTS, so without the bump every existing run would
#     keep serving the labels its old compiler produced.
REPORT_DATA_VERSION = 21
REPORT_SCHEMA_VERSION = 1
JSON_RENDER_VERSION = "0.19.0"
CATALOG_VERSION = "evalrx-report@2"

ALLOWED_COMPONENTS = frozenset(
    {
        "ReportPage",
        "SettingHero",
        "MetricStrip",
        "Journey",
        "FindingGrid",
        "ChartGrid",
        "CasePreview",
        "CaseStudySheet",
        "LoopFigure",
        "EvidenceIndex",
    }
)
REQUIRED_COMPONENTS = frozenset({"SettingHero"})
# The page needs one picture of the M1-M5 loop: the plain stage strip, or the
# loop figure that folds the case-study sheet into it.
PIPELINE_COMPONENTS = frozenset({"Journey", "LoopFigure"})
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


def build_report_data(run_dir: str | Path) -> dict[str, Any]:
    """Build the renderer-neutral data model for any completed EvalRX run."""
    root = Path(run_dir).resolve()
    from evalrx.reporting.case_study import build_case_study
    from evalrx.reporting.run_data import extract_run_data

    raw = extract_run_data(root)
    events = _read_events(root)
    # The logger copies each case's media under artifacts/case_media/ and records
    # the RUN-RELATIVE path beside the case. Carry it onto the case: inputs.<slot>
    # holds the absolute path on the producing host, so a run read anywhere else
    # -- unzipped on a laptop, which is the normal way these are looked at --
    # resolves every preview to a file that is not there.
    case_events = []
    for event in events:
        if event.get("event") != "case_record":
            continue
        case = event.get("case")
        if isinstance(case, dict):
            local = [p for p in (event.get("media_paths") or []) if isinstance(p, str)]
            # The partition the loop logged the case under (see
            # RunLoggerV2.log_cases). Absent on runs from before it was recorded.
            split = event.get("split")
            extra: dict[str, Any] = {}
            if local:
                extra["_media_paths"] = local
            if isinstance(split, str) and split:
                extra["_split"] = split
            case_events.append({**case, **extra} if extra else case)
    cases = case_events
    run = dict(raw.get("run") or {})
    trace_id = str((events[-1] if events else {}).get("trace_id") or root.name)
    reader = dict(raw.get("reader_report") or {})

    # Read before anything that can use it: the typed payloads outrank the run
    # log wherever both describe the same thing.
    contract = _contract_payloads(root)
    findings = _findings(reader, raw)
    charts = _charts(run, raw, contract)
    media = _media_index(cases)
    normalized_cases = [_normalise_case(case, media) for case in cases if isinstance(case, dict)]
    # Attach what M5's repair answered on each case, so "12 repaired, 1 broken"
    # is thirteen cases a reader can open rather than two numbers.
    repair_outcomes = _repair_outcomes(events)
    for case in normalized_cases:
        hit = repair_outcomes.get(case["id"])
        if hit:
            case["repair"] = hit
    # Which partition each case sat in -- explore for M1-M3, the withheld
    # confirm / test pools for M4 and M5 -- as the run tagged them.
    partitions = _assign_partitions(normalized_cases)
    stage_detail = _stage_detail(raw, root, normalized_cases, events)
    m5_detail = stage_detail.get("m5") if isinstance(stage_detail, dict) else None
    stages = _stages(raw, m5_detail)
    repairs = _repairs(raw, m5_detail)
    setting = {
        "model": _contract_model_name(contract) or run.get("model") or "Target model",
        "dataset": run.get("benchmark_name") or "Evaluation dataset",
        "n_cases": int(run.get("n_cases") or len(normalized_cases)),
        "partitions": partitions,
    }
    # The same run, assembled as one failure-to-repair sheet. None when the run
    # has no probe or stats artifacts to build it from -- the section is dropped
    # rather than rendered empty.
    case_study = build_case_study(root, events, setting=setting)

    return _json_safe(
        {
            "version": REPORT_DATA_VERSION,
            "trace_id": trace_id,
            "source_event_seq": max((int(e.get("event_seq") or 0) for e in events), default=0),
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "setting": {
                # The contract's ModelRef is a NAME; run.model is whatever the
                # producer stringified, which for a custom Model subclass was
                # repr() -- an unreadable line ending in a memory address.
                "model": _contract_model_name(contract) or run.get("model") or "Target model",
                "dataset": run.get("benchmark_name") or "Evaluation dataset",
                # What produced the run: the diagnose loop, or an explore step
                # on its own (no model was evaluated; only M2/M3 ran).
                "kind": run.get("kind") or "loop",
                "question": reader.get("question") or "Why did the model fail, and can we fix it?",
                "protocol": run.get("protocol") or "",
                # Who drove the pipeline, as opposed to `model`, which is what
                # it was pointed at. Empty when the run recorded neither.
                "diagnosed_by": _diagnosed_by(run),
                # Whether the run is over, and when it last logged anything:
                # what the UI needs to say "in progress" without guessing.
                "finished": _run_finished(events, root),
                "last_event_at": max((str(e["ts"]) for e in events if e.get("ts")), default=None),
                # The cover figure the run shipped, if any — see _hero_image.
                "hero_image": _hero_image(root),
                "n_cases": int(run.get("n_cases") or len(normalized_cases)),
                # The frozen batch's partitions, in the order they are drawn:
                # explore, then what was withheld. Empty when the run recorded
                # no split.
                "partitions": partitions,
            },
            "summary": {
                "headline": reader.get("headline") or "Failure analysis completed",
                "answer": reader.get("answer") or "Inspect the evidence journey below.",
                "confidence": reader.get("confidence") or "unknown",
                "stopped_by": run.get("stopped_by") or "completed",
            },
            "metrics": (
                _explore_metrics(stage_detail) if run.get("kind") == "explore"
                else _metrics(run, raw, normalized_cases, m5_detail)
            ),
            "stages": stages,
            "findings": findings,
            "charts": charts,
            "repairs": repairs,
            "stage_detail": stage_detail,
            "case_study": case_study,
            "contract": contract,
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
            # Not the sheet itself -- it is large, and the agent may not copy
            # numbers. Only whether CaseStudySheet has anything to render.
            "has_case_study": bool(data.get("case_study")),
        }
        return f"""You are the Report Agent for EvalRX. Compose a concise visual report
showing the journey: model fails on a dataset -> M1 probes behavior -> M2 screens
associations -> M3 proposes mechanisms -> M4 tests on held-out evidence -> M5 repairs
the model. A passer-by must understand the setting and outcome without knowing EvalRX.

Return ONLY a json-render tree with shape {{"root":"id","elements":{{...}}}}.
Allowed component types: {', '.join(sorted(ALLOWED_COMPONENTS))}.
Required exactly once or more: SettingHero, and one of Journey or LoopFigure.
When has_case_study is true prefer LoopFigure (inputs → explore M1·M2·M3 → held-out
M4 → repair M5 → health card, one clickable figure) in place of Journey; use
CaseStudySheet only alongside Journey, directly after it.
ReportPage may have children. Other elements use props only.
Every element MUST include a JSON object `"props": {{}}`, even when it has no
properties. This is required by the json-render runtime.
Legal props are data references such as {{"findingIds":["finding-1"]}}; never copy,
rewrite, or invent evidence text/numbers. Prefer charts and journey graphics over prose.
At most 3 findings, 2 charts, {MAX_ELEMENTS} total elements, depth {MAX_DEPTH}.
Catalog: {self.catalog_version}.

Available report data identifiers:
{json.dumps(compact, ensure_ascii=False, separators=(',', ':'))}
"""


def fallback_spec(data: Mapping[str, Any]) -> dict[str, Any]:
    """Deterministic, evidence-first layout used without or after a failed model.

    Deliberately short: hero, metrics, the M1–M5 pipeline, the takeaway sheet,
    and the audit index (the one-line outcome card was dropped 2026-09-10: it
    restated the loop figure's health card). The finding grid, the chart
    grid and the representative-case preview were dropped from the landing
    page — on a real run they repeated the metric strip (pass/fail donut) or
    said "no finding" in a full-width card, and every one of them is still a
    click away in the Evidence / Cases views. FindingGrid, ChartGrid and
    CasePreview stay in the catalog for agent-composed layouts.
    """
    children = ["setting", "metrics"]
    has_sheet = bool(data.get("case_study"))
    elements: dict[str, Any] = {
        "page": {"type": "ReportPage", "props": {}, "children": children},
        "setting": {"type": "SettingHero", "props": {}},
        "metrics": {"type": "MetricStrip", "props": {}},
        "evidence": {"type": "EvidenceIndex", "props": {}},
    }
    # With a case study the loop figure IS the pipeline strip and the sheet in
    # one picture; without one there is nothing to fill its cards, so the plain
    # stage strip stands in.
    if has_sheet:
        elements["loop"] = {"type": "LoopFigure", "props": {}}
        children.append("loop")
    else:
        elements["journey"] = {"type": "Journey", "props": {}}
        children.append("journey")
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
        if "props" not in element:
            raise ReportSpecError(f"{element_id}.props must be present (use {{}} when empty)")
        props = element["props"]
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
    if not PIPELINE_COMPONENTS.intersection(seen_types):
        raise ReportSpecError("missing required components: Journey or LoopFigure")
    _check_tree_depth(spec["root"], elements, set(), 1)
    return {"root": spec["root"], "elements": elements}


def publish_report(
    run_dir: str | Path,
    *,
    model: Any | None = None,
    run_logger: Any | None = None,
) -> PublishedReport:
    """Compile, compose, validate, cache, and optionally log a completed report."""
    root = Path(run_dir).resolve()
    data = build_report_data(root)
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
    run_root = Path(run_dir).resolve()
    root = run_root / "report"
    data = json.loads((root / "report_data.json").read_text(encoding="utf-8"))
    envelope = json.loads((root / "report_spec.json").read_text(encoding="utf-8"))
    # Do not make an old agent-composed layout disappear merely because the
    # renderer learned a richer data projection.  Rebuild its data in memory;
    # the saved layout and its provenance remain untouched.
    if int(data.get("version") or 0) < REPORT_DATA_VERSION:
        data = build_report_data(run_root)
    return data, envelope


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
    if cached_seq < evidence_seq:
        return False
    # An explore output has no event log: a rerun shows only as a report file
    # newer than the cached layout.
    from evalrx.reporting.run_data import EXPLORE_REPORTS

    cached_at = (root / "report" / "report_spec.json").stat().st_mtime
    return not any(
        (root / name).is_file() and (root / name).stat().st_mtime > cached_at
        for name in EXPLORE_REPORTS
    )


#: CLI providers under the names their vendors use. A provider absent here is
#: shown as the run recorded it — a slug is a poor label but an honest one, and
#: better than a guess at what product it belongs to.
_AGENT_LABEL = {
    "claude": "Claude Code", "claude_code": "Claude Code",
    "agy": "Antigravity", "antigravity": "Antigravity",
    "codex": "Codex", "gemini_cli": "Gemini CLI", "opencode": "OpenCode",
    "kimi_cli": "Kimi CLI", "llm": "LLM",
}


#: Extensions a run's cover figure may use, with the media type each carries.
_HERO_IMAGE_TYPES = {
    ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
    ".svg": "image/svg+xml",
}

#: Above this, embedding the cover would dominate the payload the browser has
#: to parse before anything renders — the figure is skipped, not downscaled,
#: because resampling someone's diagram is worse than not showing it.
_HERO_IMAGE_MAX_BYTES = 10_000_000


def _hero_image(root: Path) -> str:
    """The run's own cover figure as a data URI, or "" when it ships none.

    A run directory may carry ``evalrx_main.(png|jpg|jpeg|svg)`` at its
    top level — beside baseline.json, where a person browsing the folder would
    put the one picture that explains the run (a pipeline diagram, a headline
    chart). The report's hero has always reserved its right half for a
    decorative void; a run that brought its own figure fills it instead.

    Embedded as a data URI rather than served by path so the portable export,
    a dropped zip and the live server all render it identically — and checked
    in the logs directory as well as its parent, because ``root`` is the logs
    dir for a benchmark run and the run root for a flat one.
    """
    candidates = [root] + ([root.parent] if root.name == "logs" else [])
    for directory in candidates:
        for ext, mime in _HERO_IMAGE_TYPES.items():
            path = directory / f"evalrx_main{ext}"
            try:
                if not path.is_file() or path.stat().st_size > _HERO_IMAGE_MAX_BYTES:
                    continue
                payload = base64.b64encode(path.read_bytes()).decode("ascii")
            except OSError:
                continue
            return f"data:{mime};base64,{payload}"
    return ""


def _run_finished(events: Sequence[Mapping[str, Any]], root: Path) -> bool:
    """Whether the run logged its completion signal.

    ``run_end`` is written when the logger closes. An output with no run log
    at all (an explore step's report) is finished as written; a run log with
    no readable events yet has just begun.
    """
    if not events:
        return not any((where / "run.json").is_file() for where in (root, root / "logs"))
    return any(e.get("event") == "run_end" for e in events)


def _diagnosed_by(run: Mapping[str, Any]) -> str:
    """Which agent drove this run, as one display string, or "" if unrecorded.

    Two runs of the same benchmark against the same model differ ONLY in the
    agent that drove them, and without this the report never says which. The
    run_start event records it as `coder` ("claude_code:claude-opus-5").
    """
    provider, _, model = str(run.get("coder") or "").partition(":")
    if not provider:
        return ""
    label = _AGENT_LABEL.get(provider.strip().lower(), provider.strip())
    return f"{label} · {model.strip()}" if model.strip() else label


def _read_events(root: Path) -> list[dict[str, Any]]:
    from evalrx.reporting.run_events import read_v2_events

    return read_v2_events(root)


def _debug_event(event: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "event": event.get("event"), "stage": event.get("stage"), "cycle": event.get("cycle"),
        "event_seq": event.get("event_seq"), "ts": event.get("ts"), "span_id": event.get("span_id"),
        "summary": _event_summary(event),
    }


def _event_summary(event: Mapping[str, Any]) -> str:
    if str(event.get("event") or "") == "model_call":
        # The model calls are most of the stream; "Recorded pipeline event"
        # 500 times over says nothing. Name the probe, the operation, and the
        # size of the call instead.
        who = event.get("analyzer") or event.get("role") or "model"
        op = event.get("method") or event.get("operation") or "call"
        n_batch = event.get("n_batch_cases")
        scope = (f"{n_batch} cases" if n_batch else (f"case {str(event.get('case_id'))[:12]}" if event.get("case_id") else ""))
        took = event.get("duration_sec")
        parts = [f"{who} · {op}"] + ([scope] if scope else []) + ([f"{float(took):.1f}s"] if isinstance(took, (int, float)) else [])
        if event.get("error"):
            parts.append(f"error: {str(event['error'])[:80]}")
        return " · ".join(parts)
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
