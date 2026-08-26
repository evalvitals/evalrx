"""Agent-composed report contracts for the dynamic EvalVitals UI.

The evidence model is deterministic and task agnostic.  A Report Agent may
choose how to arrange a small, typed component catalog, but every value shown
by the UI is resolved from :class:`ReportData`; the model cannot invent data or
arbitrary executable UI.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

# 15: `setting.diagnosed_by` — which agent drove the run.
# 14: M3 hypotheses recover the plain sentence from proposed_hypotheses.
# 13: paired M2 rows carry the strategy pair they compared, so several tests
#     that share one tool no longer share one label. Bumping this is what
#     makes an already-published run pick the change up — `report_is_current`
#     only tracks new EVENTS, so without the bump every existing run would
#     keep serving the labels its old compiler produced.
REPORT_DATA_VERSION = 15
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

# These labels are deliberately about the behaviour being measured, rather
# than the analyzer implementation that happened to measure it.  Unknown keys
# still get a deterministic readable fallback and preserve their raw key in
# the audit-only fields below.
PLAIN_LABELS = {
    "answer_extraction_audit": "Did the model give a usable answer?",
    "calibration": "Is the model confident when it is wrong?",
    "termination_audit": "Did the model stop before finishing?",
    "format_sensitivity": "Does changing answer order change its choice?",
    "selfcheck_consistency": "Does the model contradict itself?",
    "coverage_verification_gap": "Can repeated attempts ever find the right answer?",
    "logprob_entropy": "Does the model appear uncertain while answering?",
    "self_consistency": "Does the answer stay the same across retries?",
    "signal_label_assoc": "How strongly is this behavior linked to errors?",
    "rank_corr": "Do more of this behavior and more errors move together?",
    "single_rate_evalue": "Is this rate unlikely to be chance?",
    "n_correct": "Number of correct attempts",
    "any_correct": "At least one correct attempt",
    "majority_correct": "Most attempts were correct",
    "format_flip_rate": "Answer changed after reordering choices",
    "positional_bias": "Tendency to favor one answer position",
    "output_chars": "Length of the model answer",
    "output_truncated": "Answer appears cut short",
    "recovered_by_continuation": "Could continue to a usable answer",
    "pass_at_k": "Share of retries that were correct",
    "n_sentences": "Number of answer sentences checked",
    "conf_logprob": "How certain the model appeared",
    "label_disagrees": "Recorded answer disagrees with the label",
    "has_answer_tag": "Used the requested answer format",
    "extracted_answer": "Answer read from the output",
    "labelled_fail": "This case was an error",
}


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
            case_events.append({**case, "_media_paths": local} if local else case)
    cases = [case for case in case_events if isinstance(case, dict)] or list(raw.get("cases") or [])
    run = dict(raw.get("run") or {})
    trace_id = str((events[-1] if events else {}).get("trace_id") or root.name)
    reader = dict(raw.get("reader_report") or {})

    # Read before anything that can use it: the typed payloads outrank the run
    # log wherever both describe the same thing.
    contract = _contract_payloads(root)
    findings = _findings(reader, raw)
    charts = _charts(run, raw, contract)
    repairs = _repairs(raw)
    media = _media_index(cases)
    normalized_cases = [_normalise_case(case, media) for case in cases if isinstance(case, dict)]
    normalized_cases = _merge_recorded_case_evidence(normalized_cases, root)
    # Attach what M4's repair answered on each case, so "12 repaired, 1 broken"
    # is thirteen cases a reader can open rather than two numbers.
    repairs = _repair_outcomes(root, contract)
    for case in normalized_cases:
        hit = repairs.get(case["id"])
        if hit:
            case["repair"] = hit
    stage_detail = _stage_detail(raw, root, normalized_cases, events)
    stages = _stages(raw, stage_detail.get("m4") if isinstance(stage_detail, dict) else None)

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
                "question": reader.get("question") or "Why did the model fail, and can we fix it?",
                "protocol": run.get("protocol") or "",
                # Who drove the pipeline, as opposed to `model`, which is what
                # it was pointed at. Empty when the run recorded neither.
                "diagnosed_by": _diagnosed_by(root, run),
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
            "stage_detail": stage_detail,
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
        }
        return f"""You are the Report Agent for EvalVitals. Compose a concise visual report
showing the journey: model fails on a dataset -> M1 probes behavior -> M2 screens
associations -> M3 proposes mechanisms -> M5 tests on held-out evidence -> M4 repairs
the model. A passer-by must understand the setting and outcome without knowing EvalVitals.

Return ONLY a json-render tree with shape {{"root":"id","elements":{{...}}}}.
Allowed component types: {', '.join(sorted(ALLOWED_COMPONENTS))}.
Required exactly once or more: SettingHero, Journey, OutcomeCard.
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
    run_root = Path(run_dir).resolve()
    root = run_root / "report"
    data = json.loads((root / "report_data.json").read_text(encoding="utf-8"))
    envelope = json.loads((root / "report_spec.json").read_text(encoding="utf-8"))
    # Reports published before the explicit-props contract omitted `props` for
    # simple catalog elements.  Make those immutable artifacts viewable while
    # newly generated layouts are rejected and repaired by ReportAgent.
    spec = envelope.get("spec")
    if isinstance(spec, dict) and isinstance(spec.get("elements"), dict):
        for element in spec["elements"].values():
            if isinstance(element, dict):
                element.setdefault("props", {})
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
    return cached_seq >= evidence_seq


#: CLI providers under the names their vendors use. A provider absent here is
#: shown as the run recorded it — a slug is a poor label but an honest one, and
#: better than a guess at what product it belongs to.
_AGENT_LABEL = {
    "claude": "Claude Code", "claude_code": "Claude Code",
    "agy": "Antigravity", "antigravity": "Antigravity",
    "codex": "Codex", "gemini_cli": "Gemini CLI", "opencode": "OpenCode",
    "kimi_cli": "Kimi CLI", "llm": "LLM",
}


def _diagnosed_by(root: Path, run: Mapping[str, Any]) -> str:
    """Which agent drove this run, as one display string, or "" if unrecorded.

    Two runs of the same benchmark against the same model differ ONLY in the
    agent that drove them, and the report never said which — open both and they
    are indistinguishable. The run manifest records the choice cleanly
    (`judge_provider` / `judge_model`); the run_start event's `coder`
    ("claude_code:opus") is the fallback for a run written before the manifest,
    and its `judge` is a repr, so it is read only for its provider prefix.
    """
    config = _load_json(root / "manifest.json")
    config = (config or {}).get("config") if isinstance(config, Mapping) else None
    provider = model = ""
    if isinstance(config, Mapping):
        provider = str(config.get("judge_provider") or "")
        model = str(config.get("judge_model") or "")
    if not provider:
        coder = str(run.get("coder") or "")
        if ":" in coder:
            provider, _, model = coder.partition(":")
        else:
            provider = coder
    if not provider:
        return ""
    label = _AGENT_LABEL.get(provider.strip().lower(), provider.strip())
    return f"{label} · {model.strip()}" if model.strip() else label


def _stages(raw: Mapping[str, Any], m4_detail: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    definitions = [
        ("M1", "Check behavior", "See what the model does differently when it succeeds or fails", raw.get("m1"), "results"),
        ("M2", "Find patterns", "Look for behaviors that reliably appear alongside errors", raw.get("m2"), "stats"),
        ("M3", "Suggest causes", "Turn the strongest patterns into ideas that can be tested", raw.get("m3"), "hypotheses"),
        ("M5", "Test on new cases", "Check those ideas on evidence not used to create them", raw.get("m5"), "results"),
        ("M4", "Try repairs", "Compare possible fixes with the unchanged model", raw.get("m4_fix"), "selection"),
    ]
    result = []
    for code, title, purpose, payload, evidence_key in definitions:
        stage = dict(payload or {})
        evidence = stage.get(evidence_key) or []
        ran = bool(stage.get("ran")) if "ran" in stage else bool(evidence or stage.get("duration"))
        status = "completed" if ran else "not-run"
        if code == "M4" and m4_detail and m4_detail.get("skipped"):
            status = "skipped"
        if code == "M4" and ran:
            status = "improved" if stage.get("fixed") else "no-improvement"
        result.append({
            "id": code.lower(), "code": code, "title": title, "purpose": purpose,
            "status": status, "evidence_count": len(evidence) if isinstance(evidence, list) else 1,
        })
    return result


def _stage_detail(
    raw: Mapping[str, Any], root: Path, cases: list[dict[str, Any]], events: list[Mapping[str, Any]]
) -> dict[str, Any]:
    """Compile the mature stage-specific views into a bounded web contract.

    The contract preserves the old workbench's load-bearing distinctions:
    M2 is descriptive, M3 is proposal-only, M5 owns held-out verdicts, and M4
    compares every intervention against the unchanged baseline.
    """
    logs_dir = Path(raw.get("logs_dir") or root)
    m1 = dict(raw.get("m1") or {})
    probes = []
    for result in m1.get("results") or []:
        if not isinstance(result, dict):
            continue
        rows = list(result.get("per_case") or [])
        probes.append({
            "id": str(result.get("name") or "probe"),
            "title": _plain_label(result.get("name") or result.get("display_name") or "Probe"),
            "raw_name": str(result.get("name") or ""),
            "question": str(result.get("question") or ""),
            "description": str(result.get("description") or ""),
            "n_cases": result.get("n") or len(rows),
            "metrics": list(result.get("headline") or []),
            "finding_summary": _plain_mapping(_compact_mapping(result.get("findings"))),
            "raw_finding_summary": _compact_mapping(result.get("findings")),
            "sample_rows": rows[:12],
            "n_sample_rows": len(rows),
        })

    m2 = dict(raw.get("m2") or {})
    explore = dict(m2.get("explore") or {})
    figure_dir = Path(raw.get("explore_dir") or logs_dir.parent / "explore") / "figures"
    figures = _explore_figures(explore, figure_dir, root)
    takeaways = []
    for item in explore.get("takeaways") or []:
        if isinstance(item, dict):
            takeaways.append({key: item.get(key) for key in (
                "plain_title", "title", "analysis", "caveat", "chart_names", "table_names",
            )})

    judges = list((raw.get("agents") or {}).get("judge_calls") or [])
    m3_call = next((call for call in judges if "M3" in str(call.get("stage") or "")), {})
    agent_response = str(m3_call.get("response") or "")
    m3 = dict(raw.get("m3") or {})
    hypotheses = list(m3.get("hypotheses") or explore.get("hypotheses") or [])
    # Runs recorded before the log writer carried `plain_statement` on this list
    # still have it on `proposed_hypotheses`, which the critic step copies from.
    # Joining on the statement recovers the plain sentence the judge did write,
    # instead of leaving the report showing only the technical line. A run with
    # neither is left exactly as it is — nothing is paraphrased into existence.
    proposed = {str(h.get("statement") or ""): h
                for h in (m3.get("proposed_hypotheses") or []) if isinstance(h, Mapping)}
    for h in hypotheses:
        if isinstance(h, dict) and not h.get("plain_statement"):
            recovered = proposed.get(str(h.get("statement") or ""), {}).get("plain_statement")
            if recovered:
                h["plain_statement"] = recovered

    m5 = dict(raw.get("m5") or {})
    saved_m5 = _load_json(logs_dir / "report" / "m5_results.json")
    m5_results = saved_m5 if isinstance(saved_m5, list) and saved_m5 else list(m5.get("results") or [])
    m5_event = dict(m5.get("event") or {})

    m4 = dict(raw.get("m4_fix") or {})
    # html_report may see an older sibling log when an example contains
    # multiple attempts.  The event stream is trace-filtered and is the source
    # of truth for the run currently being rendered.
    fix_event = next((event for event in reversed(events) if event.get("event") == "fix"), {})
    skipped_event = next((
        event for event in reversed(events)
        if event.get("event") == "stage_skipped" and str(event.get("stage")) == "M4"
    ), {})
    if fix_event:
        attempted = [item for item in fix_event.get("attempted") or [] if isinstance(item, Mapping)]
        best_name = fix_event.get("best")
        best = next((item for item in attempted if item.get("name") == best_name), {})
        m4.update({
            "ran": True, "fixed": bool(fix_event.get("fixed")), "selection": attempted,
            "best": best, "confirm": best, "recommendation": fix_event.get("recommendation"),
        })
    if skipped_event:
        m4.update({
            "ran": False,
            "skipped": True,
            "skip_reason": skipped_event.get("reason_code") or "not_attempted",
            "skip_detail": skipped_event.get("detail") or "",
        })
    candidates = [_repair_candidate(item) for item in m4.get("selection") or [] if isinstance(item, dict)]
    confirmation = _repair_candidate(m4.get("confirm") or {})
    if confirmation.get("name"):
        existing = next((i for i, item in enumerate(candidates) if item.get("name") == confirmation["name"]), None)
        if existing is None:
            candidates.append(confirmation)
        else:
            candidates[existing] = {**candidates[existing], **confirmation}
    # Examples are a first-class, bounded evidence type.  They are deliberately
    # derived from recorded case I/O and per-case measurements, never written
    # by the report agent.  A single case makes a measurement legible; it does
    # not establish the stage conclusion (the UI states that boundary too).
    logged_m1_examples = [
        item for event in events if event.get("event") == "probe"
        for item in (event.get("examples") or []) if isinstance(item, Mapping)
    ]
    logged_m5_event = next((event for event in reversed(events) if event.get("event") == "surgery" and event.get("module") == "m5"), {})
    m1_examples = [dict(item) for item in logged_m1_examples[:2]] or _m1_examples(probes, cases)
    m5_examples = _m5_examples(m5_results, {**m5_event, **logged_m5_event}, cases)
    m1_examples = _attach_example_media(m1_examples, cases)
    m5_examples = _attach_example_media(m5_examples, cases)
    enriched_stats = _enrich_m2_stats(
        [item for item in m2.get("stats") or [] if isinstance(item, Mapping)], logs_dir
    )
    display_stats = [_display_stat(item) for item in enriched_stats]
    repair_examples = _m4_examples(candidates, cases, list(m1.get("results") or []))
    # A trajectory is the only source allowed to say that an agent actually
    # used an intermediate operation (zoom, crop, search, ...).  Candidate
    # specs are kept separately below because a proposed operation is not an
    # executed one.
    operation_examples = _operation_examples(cases)
    repair_operation_previews = _repair_operation_previews(candidates, cases)
    return {
        "m1": {
            "duration": m1.get("duration"), "probes": probes,
            "n_probes": len(probes), "n_measured": max((int(p.get("n_cases") or 0) for p in probes), default=0),
            "examples": m1_examples, "operations": operation_examples,
        },
        "m2": {
            "mode": "descriptive", "conclusion": m2.get("conclusion") or "",
            "narrative": m2.get("narrative") or "", "stats": display_stats,
            "takeaways": takeaways, "figures": figures,
            "observations": list(explore.get("observations") or []),
            "caveats": list(explore.get("caveats") or []),
            "candidate_signals": list(explore.get("candidate_signals") or []),
            "recommended_tests": list(explore.get("recommended_confirmatory_tests") or []),
        },
        "m3": {
            "mode": "proposal", "hypotheses": hypotheses,
            "unparsed_proposals": _recover_unparsed_hypotheses(agent_response) if not hypotheses else [],
            "agent_response": agent_response,
            "candidate_signals": list(explore.get("candidate_signals") or []),
            "recommended_tests": list(explore.get("recommended_confirmatory_tests") or []),
            "evidence_figures": figures[:3], "evidence_stats": display_stats[:6],
        },
        "m5": {
            "ran": bool(m5.get("ran") or m5_results), "mode": "confirmatory",
            "results": m5_results, "event": m5_event, "examples": m5_examples,
        },
        "m4": {
            "ran": bool(m4.get("ran")), "fixed": bool(m4.get("fixed")),
            "skipped": bool(m4.get("skipped")),
            "skip_reason": str(m4.get("skip_reason") or ""),
            "skip_detail": str(m4.get("skip_detail") or ""),
            "candidates": candidates, "confirmation": confirmation,
            "best": _repair_candidate(m4.get("best") or {}),
            "examples": repair_examples,
            "operation_previews": repair_operation_previews,
            "surgeries": list((raw.get("m4_surgery") or {}).get("surgeries") or []),
            "prompt_template": str(m4.get("prompt_template") or ""),
        },
    }


def _m1_examples(probes: list[Mapping[str, Any]], cases: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Pick recorded, human-readable M1 case walk-throughs.

    Recent loggers may include ``examples`` directly on a probe.  Older runs
    are reconstructed by joining a per-case measurement to the saved baseline
    case record.  The latter intentionally says *check result*, not
    *after-answer*: many analyzers do not issue a second generation.
    """
    case_by_id = {str(case.get("id")): case for case in cases}
    result: list[dict[str, Any]] = []
    used: set[str] = set()
    for probe in probes:
        supplied = probe.get("examples")
        if isinstance(supplied, list):
            for item in supplied:
                if isinstance(item, Mapping):
                    result.append(dict(item))
                    if len(result) >= 2:
                        return result
        rows = [row for row in probe.get("sample_rows") or [] if isinstance(row, Mapping)]
        # Failures are most intuitive when available.  A pass is still useful
        # when the run has no labelled failures in this particular probe.
        rows.sort(key=lambda row: 0 if str(case_by_id.get(str(row.get("sample_id")), {}).get("status")) == "fail" else 1)
        for row in rows:
            case_id = str(row.get("sample_id") or row.get("case_id") or "")
            case = case_by_id.get(case_id)
            if not case or case_id in used:
                continue
            measured = {
                str(key).replace("_", " "): value
                for key, value in row.items()
                if key not in {"sample_id", "case_id"} and isinstance(value, (str, int, float, bool))
            }
            if not measured:
                continue
            used.add(case_id)
            result.append({
                "id": f"m1-{probe.get('id', 'probe')}-{case_id}",
                "kind": "case_measurement",
                "case_id": case_id,
                "probe_title": probe.get("title") or "Behavioral check",
                "probe_question": probe.get("question") or "",
                "input": case.get("prompt") or "",
                "baseline_output": case.get("observed"),
                "expected": case.get("expected"),
                "outcome": case.get("status") or "unknown",
                "media_ids": list(case.get("media_ids") or []),
                "check_result": measured,
                "plain_reading": "This one case shows what the check records. The overall pattern comes from every measured case, not this example alone.",
                "evidence_scope": "one recorded case within M1",
            })
            break
        if len(result) >= 2:
            break
    return result


def _attach_example_media(examples: list[dict[str, Any]], cases: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    case_by_id = {str(case.get("id")): case for case in cases}
    for example in examples:
        if example.get("probe_title"):
            example["probe_title"] = _plain_label(example["probe_title"])
        if example.get("media_ids"):
            continue
        case = case_by_id.get(str(example.get("case_id") or ""))
        if case:
            example["media_ids"] = list(case.get("media_ids") or [])
    return examples


def _m5_examples(
    results: list[Any], event: Mapping[str, Any], cases: list[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    """Return explicit validation examples, with a truthful aggregate fallback."""
    supplied = event.get("validation_examples") if isinstance(event, Mapping) else None
    if isinstance(supplied, list):
        first = next((item for item in results if isinstance(item, Mapping)), {})
        evidence = first.get("evidence") if isinstance(first.get("evidence"), Mapping) else {}
        kept = [{
            **dict(item),
            "hypothesis": item.get("hypothesis") or first.get("hypothesis") or first.get("statement"),
            "test": item.get("test") or first.get("test_name") or evidence.get("chosen_tool"),
            "status": item.get("status") or first.get("status"),
        } for item in supplied if isinstance(item, Mapping)]
        if kept:
            return kept[:2]
    if not results:
        return []
    first = next((item for item in results if isinstance(item, Mapping)), None)
    if first is None:
        return []
    evidence = first.get("evidence") if isinstance(first.get("evidence"), Mapping) else {}
    # Legacy M5 logs retain the frozen claim and the actual validation statistic
    # but not a row-level verdict.  That is still a concrete example of the
    # validation operation; label it as an aggregate test rather than pretend a
    # case was individually adjudicated.
    return [{
        "id": "m5-validation-test-1",
        "kind": "validation_test",
        "hypothesis": first.get("hypothesis") or first.get("statement") or "Frozen hypothesis",
        "test": first.get("test_name") or evidence.get("chosen_tool") or "held-out statistical test",
        "effect": first.get("effect_size", evidence.get("effect_size")),
        "interval": evidence.get("ci"),
        "status": first.get("status") or "inconclusive",
        "verdict": first.get("verdict") or evidence.get("m5_verdict") or "No validation explanation was retained.",
        "plain_reading": "This is one frozen claim tested on the independent validation evidence. The verdict uses the full validation set, not a single hand-picked case.",
        "evidence_scope": "aggregate independent-validation test",
    }]


def _display_stat(value: Mapping[str, Any]) -> dict[str, Any]:
    """Keep statistical evidence machine-auditable but primary labels human."""
    config = value.get("config") if isinstance(value.get("config"), Mapping) else {}
    signal = str(config.get("signal") or value.get("signal") or value.get("tool") or "")
    details = value.get("details") if isinstance(value.get("details"), Mapping) else {}
    # A PAIRED tool compares two strategies and carries no `signal`, so `signal`
    # falls through to the tool name and every such row was labelled with the
    # same words ("Mcnemar Evalue"). Three identically-labelled bars is not a
    # chart. The pair it actually compared is right there in the config, and it
    # is the only thing that tells the rows apart.
    groups = [str(name) for name in (config.get("strategies") or []) if name]
    label = _stat_label(config, signal, value.get("tool"))
    return {
        "label": label, "raw_signal": signal, "groups": groups,
        "effect": value.get("effect"), "ci": value.get("ci"),
        "reject": bool(value.get("reject")), "underpowered": bool(value.get("underpowered")),
        "summary": value.get("summary") or "", "p_value": value.get("p_value"),
        "tool": value.get("tool"), "e_value": value.get("e_value"),
        "fail_rate_signal": details.get("fail_rate_signal"),
        "fail_rate_control": details.get("fail_rate_control"),
    }


def _enrich_m2_stats(stats: list[Mapping[str, Any]], logs_dir: Path) -> list[dict[str, Any]]:
    """Restore detailed M2 rate fields from durable tool-result artifacts.

    The compact HTML reader deliberately omits bulky ``details``.  Its
    omission used to leave the dynamic M2 view with only one generic effect
    chart, despite the run already having stored the two groups needed for a
    direct error-rate comparison.  This is a local join keyed by the exact
    test and signal; no values are re-computed or inferred.
    """
    artifact_rows: dict[tuple[str, str], Mapping[str, Any]] = {}
    for path in sorted((logs_dir / "artifacts").glob("*m2_stats_results*.json")):
        value = _load_json(path)
        if not isinstance(value, list):
            continue
        for item in value:
            if not isinstance(item, Mapping):
                continue
            config = item.get("config") if isinstance(item.get("config"), Mapping) else {}
            key = (str(item.get("tool") or ""), str(config.get("signal") or item.get("signal") or ""))
            if key[1]:
                artifact_rows[key] = item
    result: list[dict[str, Any]] = []
    for item in stats:
        config = item.get("config") if isinstance(item.get("config"), Mapping) else {}
        key = (str(item.get("tool") or ""), str(config.get("signal") or item.get("signal") or ""))
        saved = artifact_rows.get(key)
        result.append({**dict(item), **({"details": saved.get("details")} if saved and saved.get("details") else {})})
    return result


def _operation_examples(cases: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Project recorded agent tool trajectories into compact, case-linked UI cards.

    This intentionally ignores prompt text that *mentions* a tool.  An action
    appears only if a persisted trajectory step carries a structured
    ``tool_call``.  That preserves the distinction between an actual zoom and
    an LLM merely proposing one.
    """
    result: list[dict[str, Any]] = []
    for case in cases:
        trajectory = case.get("trajectory")
        if not isinstance(trajectory, Mapping):
            continue
        steps = trajectory.get("steps")
        if not isinstance(steps, list):
            continue
        calls = []
        for step in steps:
            if not isinstance(step, Mapping) or not isinstance(step.get("tool_call"), Mapping):
                continue
            call = step["tool_call"]
            name = str(call.get("name") or "tool")
            calls.append({
                "order": int(step.get("idx") or len(calls) + 1),
                "action": _plain_operation(name), "raw_action": name,
                "parameters": dict(call.get("args") or {}) if isinstance(call.get("args"), Mapping) else {},
                "thought": str(step.get("content") or "")[:500],
            })
        if not calls:
            continue
        result.append({
            "id": f"operations-{case.get('id')}", "case_id": str(case.get("id") or ""),
            "input": case.get("prompt") or "", "expected": case.get("expected"),
            "observed": case.get("observed"), "outcome": case.get("status") or "unknown",
            "media_ids": list(case.get("media_ids") or []), "steps": calls[:6],
            "plain_reading": "These are the actions recorded in this agent's own trajectory for this case.",
        })
        if len(result) >= 2:
            break
    return result


def _repair_operation_previews(
    candidates: list[Mapping[str, Any]], cases: list[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    """Show what an image-operation repair *would* apply, without claiming execution.

    Older runs preserve a repair spec but not a per-call visual artifact.  The
    report can still pair its declared image operations with the original input
    image for a representative case, as long as the card is explicitly marked
    as a candidate preview.  Actual execution is represented by
    :func:`_operation_examples` instead.
    """
    image_cases = [case for case in cases if case.get("media_ids")]
    result: list[dict[str, Any]] = []
    for candidate in candidates:
        payload = candidate.get("payload") if isinstance(candidate.get("payload"), Mapping) else {}
        ops = [op for op in payload.get("image_ops") or [] if isinstance(op, Mapping) and op.get("tool")]
        if not ops:
            continue
        preferred_ids = [str(item) for item in (candidate.get("fixed_cases") or candidate.get("broken_cases") or [])]
        case = next((item for item in image_cases if str(item.get("id")) in preferred_ids), None)
        case = case or next((item for item in image_cases if str(item.get("status")) == "fail"), None)
        # A legacy run can retain its generated baseline answers in a separate
        # artifact whose ids no longer line up with the saved image manifest.
        # In that case an arbitrary corpus image is still useful to explain the
        # *declared transform*, provided the UI calls it a preview and never a
        # validated before/after result.
        case = case or (image_cases[0] if image_cases else None)
        if case is None:
            continue
        result.append({
            "id": f"repair-operations-{candidate.get('name')}", "candidate": candidate.get("name"),
            "case_id": str(case.get("id") or ""), "input": case.get("prompt") or "",
            "expected": case.get("expected"), "observed": case.get("observed"),
            "media_ids": list(case.get("media_ids") or []),
            "executed": bool(candidate.get("n_pairs")),
            "operations": [{
                "action": _plain_operation(op.get("tool")), "raw_action": str(op.get("tool")),
                "parameters": dict(op.get("params") or {}) if isinstance(op.get("params"), Mapping) else {},
            } for op in ops],
        })
        if len(result) >= 2:
            break
    return result


def _plain_operation(value: Any) -> str:
    names = {
        "image_zoom_in": "Zoom in on part of the image", "zoom_center": "Zoom into the chart center",
        "crop": "Crop the visual evidence", "crop_case_bbox": "Crop the relevant chart region",
        "sharpen": "Sharpen chart details", "contrast": "Increase visual contrast",
        "equalize": "Balance the image contrast", "detect": "Locate an object or region",
    }
    raw = str(value or "")
    return names.get(raw, _plain_label(raw))


def _m4_examples(
    candidates: list[Mapping[str, Any]], cases: list[Mapping[str, Any]], probe_results: list[Any]
) -> list[dict[str, Any]]:
    """Reconstruct a truthful before/after repair example — confirmed or not.

    A candidate that never cleared the significance bar is still what the
    search actually tried, and a reader debugging a failed repair needs to
    see a real case, not just "no candidate passed the repair gate". Prefer a
    `fixed` winner (statistically significant, net-positive); when none
    exists, fall back to the strongest attempt by effect size — the same
    fallback `M4Detail` already uses for its own "best repair" KPI — and tag
    the example `confirmed: False` so the UI can say plainly it is one
    candidate's attempt, not an accepted repair. When that candidate never
    flipped a case to correct either, fall back once more to a case it broke:
    still evidence, just of the failure mode rather than the fix.
    """
    winner = next((item for item in candidates if item.get("fixed")), None)
    confirmed = winner is not None
    if winner is None:
        winner = max(
            (item for item in candidates if isinstance(item.get("effect"), (int, float))),
            key=lambda item: item["effect"],
            default=(candidates[0] if candidates else None),
        )
    if winner is None:
        return []
    case_by_id = {str(case.get("id")): case for case in cases}
    baseline_by_id: dict[str, Any] = {}
    for probe in probe_results:
        if not isinstance(probe, Mapping):
            continue
        for row in probe.get("per_case") or []:
            if not isinstance(row, Mapping):
                continue
            case_id = str(row.get("sample_id") or row.get("case_id") or "")
            output = row.get("extracted_answer", row.get("modal_answer"))
            if case_id and output not in (None, ""):
                baseline_by_id.setdefault(case_id, output)

    def _build(case_id: str, kind: str) -> "dict[str, Any] | None":
        case = case_by_id.get(case_id)
        if not case:
            return None
        repaired = case.get("observed")
        baseline = baseline_by_id.get(case_id)
        if repaired in (None, ""):
            return None
        if confirmed:
            reading = ("This case was counted as fixed in the paired repair evaluation. "
                       "The repair decision still depends on all tested cases and its "
                       "regression checks.")
        elif kind == "fixed":
            reading = ("This candidate flipped this case from wrong to right, but the "
                       "repair overall did not clear the significance bar against luck — "
                       "read it as one attempt worth inspecting, not a confirmed fix.")
        else:
            reading = ("This candidate did not clear the significance bar, and did not "
                       "flip any case to correct either — this is a case it broke instead, "
                       "shown so the failure mode is inspectable.")
        return {
            "id": f"m4-{winner.get('name', 'repair')}-{case_id}", "case_id": case_id,
            "kind": kind, "confirmed": confirmed,
            "repair_name": winner.get("name") or "Recorded repair", "input": case.get("prompt") or "",
            "expected": case.get("expected"), "baseline_output": baseline,
            "repaired_output": repaired, "media_ids": list(case.get("media_ids") or []),
            "plain_reading": reading,
            "baseline_available": baseline not in (None, ""),
        }

    for case_id in [str(item) for item in winner.get("fixed_cases") or []]:
        example = _build(case_id, "fixed")
        if example:
            return [example]
    for case_id in [str(item) for item in winner.get("broken_cases") or []]:
        example = _build(case_id, "broken")
        if example:
            return [example]
    return []
    return []


def _compact_mapping(value: Any, *, limit: int = 18) -> dict[str, Any]:
    """Retain readable aggregate findings, excluding unbounded per-case maps."""
    if not isinstance(value, Mapping):
        return {}
    compact: dict[str, Any] = {}
    for key, item in value.items():
        if key in {"per_case", "per_case_values", "case_values", "samples", "caveat"}:
            continue
        if isinstance(item, (str, int, float, bool)) or item is None:
            compact[str(key)] = item
        elif isinstance(item, list) and len(item) <= 12:
            compact[str(key)] = item
        elif isinstance(item, Mapping) and len(item) <= 12:
            nested = {str(k): v for k, v in item.items() if isinstance(v, (str, int, float, bool)) or v is None}
            if nested:
                compact[str(key)] = nested
        if len(compact) >= limit:
            break
    return compact


def _plain_label(value: Any) -> str:
    raw = str(value or "").strip()
    if not raw:
        return "Recorded measurement"
    if raw in PLAIN_LABELS:
        return PLAIN_LABELS[raw]
    return re.sub(r"\s+", " ", raw.replace("_", " ")).strip().capitalize()


def _stat_label(config: "Mapping[str, Any] | None", signal: Any, tool: Any) -> str:
    """A name for what one statistical test measured, unique within a report.

    A signal test names its signal. A PAIRED test names none — it compares two
    strategies — so the label used to fall through to the tool, and every paired
    row in the report came out reading "Mcnemar Evalue". The pair it compared is
    the thing that tells those rows apart, and it is already in the config.
    """
    cfg = config if isinstance(config, Mapping) else {}
    if cfg.get("signal"):
        return _plain_signal(cfg["signal"])
    groups = [str(name) for name in (cfg.get("strategies") or []) if name]
    if len(groups) >= 2:
        return f"{_plain_label(groups[-1])} vs {_plain_label(groups[0])}"
    return _plain_signal(signal or tool)


def _plain_signal(value: Any) -> str:
    """Convert ``analyzer.metric`` into a question a non-expert can read."""
    raw = str(value or "")
    if not raw:
        return "Recorded behavior"
    analyzer, _, metric = raw.partition(".")
    if analyzer and metric:
        return f"{_plain_label(analyzer)} — {_plain_label(metric)}"
    return _plain_label(raw)


def _plain_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    return {_plain_label(key): item for key, item in value.items()}


def _visual_key(value: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", Path(str(value or "")).stem.lower())


def _explore_figures(explore: Mapping[str, Any], figure_dir: Path, root: Path) -> list[dict[str, Any]]:
    plans = [item for item in explore.get("visual_plan") or [] if isinstance(item, Mapping)]
    readings = [item for item in explore.get("chart_readings") or [] if isinstance(item, Mapping)]
    files = sorted(figure_dir.glob("*.png")) if figure_dir.is_dir() else []
    result: list[dict[str, Any]] = []
    used: set[Path] = set()
    for plan in plans:
        name = str(plan.get("name") or "")
        key = _visual_key(name)
        match = next((path for path in files if key and key in _visual_key(path.name)), None)
        if match is None:
            continue
        used.add(match)
        reading = next((item for item in readings if _visual_key(item.get("chart")) == key), {})
        result.append({
            "id": name, "title": str(plan.get("display_name") or name.replace("_", " ").title()),
            "question": str(plan.get("question") or ""), "path": os.path.relpath(match, root),
            "reading": str(reading.get("reading") or ""),
            "do_not_infer": str(reading.get("do_not_infer") or ""),
            "disposition": str(plan.get("disposition") or "supporting"),
            "not_promoted_reason": str(plan.get("not_promoted_reason") or ""),
        })
    for path in files:
        if path not in used:
            result.append({
                "id": path.stem, "title": path.stem.replace("_", " ").title(),
                "question": "", "path": os.path.relpath(path, root), "reading": "",
                "do_not_infer": "No agent-authored interpretation was saved for this figure.",
                "disposition": "audit", "not_promoted_reason": "Unlinked exploratory artifact",
            })
    return result[:30]


def _recover_unparsed_hypotheses(response: str) -> list[dict[str, Any]]:
    """Recover display-only proposals while preserving the parser failure state."""
    proposals = []
    for block in re.split(r"\n\s*\n", response.strip()):
        fields: dict[str, str] = {}
        for line in block.splitlines():
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            fields[key.strip().lower()] = value.strip()
        if fields.get("hypothesis"):
            proposals.append({
                "statement": fields["hypothesis"],
                "plain_statement": fields.get("plain_statement", ""),
                "failure_mode": fields.get("failure_mode", ""),
                "test_design": fields.get("test", ""),
                "expected_association": fields.get("expected_association", ""),
                "accepted_by_pipeline": False,
            })
    return proposals[:12]


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None


def _repair_headline(value: Mapping[str, Any]) -> str:
    """The candidate's own sentence, or the host's for a candidate it authored.

    Looking `visual_grounding` up in the fix agent's own table is not a
    consumer guessing at a slug -- it is the package that named the candidate
    saying what it does. Runs recorded before `headline` existed become
    readable this way; a judge-invented name the host never heard of still
    resolves to "" and renders blank.
    """
    headline = " ".join(str(value.get("headline") or "").split())
    if headline:
        return headline[:300]
    try:
        from evalvitals.eval_agent.stages.fix_agent import _BUILTIN_DESCRIPTIONS
    except Exception:
        return ""
    return _BUILTIN_DESCRIPTIONS.get(str(value.get("name") or ""), "")


def _repair_candidate(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    fixed_cases = list(value.get("fixed_cases") or [])
    broken_cases = list(value.get("broken_cases") or [])
    payload = value.get("payload") if isinstance(value.get("payload"), Mapping) else {}
    return {
        "tier": value.get("tier"), "name": value.get("name"), "kind": value.get("kind"),
        # `ref` and `headline` are what a reader sees; `name` is only the join
        # key. Both are carried here so the legacy report path shows the same
        # identifier and the same sentence as the contract-backed views.
        "ref": value.get("ref") or "", "headline": _repair_headline(value),
        "source": value.get("source"), "verdict": value.get("verdict"),
        "n_pairs": value.get("n_pairs"), "n_baseline_correct": value.get("n_baseline_correct"),
        "n_candidate_correct": value.get("n_candidate_correct"),
        "n_fixed": value.get("n_fixed", len(fixed_cases)), "n_broken": value.get("n_broken", len(broken_cases)),
        "effect": value.get("effect"), "e_value": value.get("e_value"),
        "coverage": value.get("coverage"), "reject": value.get("reject"), "fixed": value.get("fixed"),
        "n_model_independent": value.get("n_model_independent"),
        "n_unstable": value.get("n_unstable"),
        "summary": value.get("summary"), "payload": dict(payload),
        "fixed_cases": fixed_cases[:20], "broken_cases": broken_cases[:20],
    }


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


def _charts(run: Mapping[str, Any], raw: Mapping[str, Any],
            contract: "Mapping[str, Any] | None" = None) -> list[dict[str, Any]]:
    charts: list[dict[str, Any]] = []
    labels = run.get("label_distribution") or {}
    if isinstance(labels, dict) and labels:
        charts.append({
            "id": "outcomes", "kind": "donut", "title": "Evaluation outcomes",
            "series": [{"label": str(k).title(), "value": v} for k, v in labels.items() if isinstance(v, (int, float))],
        })
    # Prefer the contract's rows: `measured` is a name for the SUBJECT and is
    # unique across the report, so the axis cannot end up with two bars reading
    # "Mcnemar evalue" (the tool) or several truncating to the same sentence.
    contract_rows = _contract_stats(contract or {})
    effect_rows = []
    for item in contract_rows[:8]:
        if isinstance(item.get("effect"), (int, float)):
            signal = (item.get("config") or {}).get("signal")
            effect_rows.append({
                "label": item.get("measured") or _plain_signal(signal or item.get("tool")),
                # The analyzer's own sentence. Absent means undocumented, and the
                # UI must say that rather than paraphrase the identifier.
                "means": item.get("means") or "",
                "raw_label": str(signal or item.get("analysis_key") or item.get("tool") or ""),
                "value": item["effect"], "highlight": bool(item.get("fdr_corrected")),
            })
    if not effect_rows:
        stats = (raw.get("m2") or {}).get("stats") or []
        for item in stats[:8]:
            if isinstance(item, dict) and isinstance(item.get("effect"), (int, float)):
                signal = ((item.get("config") or {}).get("signal") if isinstance(item.get("config"), Mapping) else None)
                effect_rows.append({
                    "label": _stat_label(item.get("config"), signal, item.get("tool")),
                    "raw_label": str(signal or item.get("tool") or ""),
                    "value": item["effect"], "highlight": bool(item.get("reject")),
                })
    if effect_rows:
        charts.append({
            "id": "effects", "kind": "bar", "title": "Which behaviors appear most connected to errors?",
            "subtitle": "A larger bar means a stronger observed pattern. It does not prove the behavior caused the error.",
            "series": effect_rows,
        })
    return charts


def _repairs(raw: Mapping[str, Any]) -> list[dict[str, Any]]:
    fix = raw.get("m4_fix") or {}
    if not fix.get("ran"):
        return []
    confirm = fix.get("confirm") if isinstance(fix.get("confirm"), Mapping) else {}
    # Early runs recorded the winning candidate as its bare name; later ones
    # record the whole candidate. Both name the same repair.
    best = fix.get("best")
    best_name = str(best.get("name") or "") if isinstance(best, Mapping) else str(best or "")
    return [{
        "id": "repair-1", "fixed": bool(fix.get("fixed")),
        "title": best_name or confirm.get("name") or "Targeted repair",
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


def _contract_payloads(root: Path) -> dict[str, Any]:
    """The validated per-stage payloads a run emitted, keyed by span id.

    Passed through verbatim. ``stage_detail`` above is a hand-built view whose
    every field access is a defensive ``.get(x) or y`` — it has to be, because
    it reads whatever the run log happened to contain. These payloads were
    validated against the wire models on the way out, so a frontend can decode
    them with the generated TypeScript instead of re-deriving the shape.

    Absent for a run produced before contract emission existed, or one where the
    ``contract`` extra was not installed. ``{}`` says "this run emitted none",
    which a reader must not confuse with "this run had no stages".
    """
    out: dict[str, Any] = {}
    directory = root / "contract"
    if not directory.is_dir():
        return out
    for path in sorted(directory.glob("*.json")):
        if path.name == "index.json":
            continue
        payload = _load_json(path)
        if payload is None:
            continue
        # "c0.m1.json" -> "c0.m1"; ".invalid.json" keeps its suffix so a reader
        # can see that a stage failed validation rather than silently missing it.
        out[path.name[: -len(".json")]] = payload
    for key, payload in out.items():
        if key.endswith("m4_fix"):
            _backfill_repair_identity(payload)
    return out


def _backfill_repair_identity(payload: Any) -> None:
    """Give a pre-`ref` run the same numbering a current run would emit.

    Runs recorded before these fields existed carry neither, and a consumer
    left to number rows itself gives the frozen candidate one number in the
    sweep and another on its card. The assignment here is the emitter's, done
    the emitter's way -- selection order first, one number per distinct name --
    so an old report and a new one read alike.

    It only ever FILLS BLANKS. What a producer actually wrote is never
    overwritten, and a `headline` is only ever taken from the fix agent's table
    of its own candidates: a judge-invented name nobody described stays blank,
    because inventing a sentence for it here would be this file guessing.
    """
    if not isinstance(payload, dict):
        return
    try:
        from evalvitals.eval_agent.stages.fix_agent import _BUILTIN_DESCRIPTIONS
    except Exception:
        _BUILTIN_DESCRIPTIONS = {}
    refs: dict[str, str] = {}
    for group in ("selection", "attempted"):
        for row in payload.get(group) or []:
            if not isinstance(row, dict):
                continue
            name = str(row.get("name") or "")
            ref = refs.get(name)
            if ref is None:
                ref = f"R{len(refs) + 1}"
                refs[name] = ref
            row.setdefault("ref", "")
            if not row["ref"]:
                row["ref"] = ref
            row.setdefault("headline", "")
            if not row["headline"]:
                row["headline"] = _BUILTIN_DESCRIPTIONS.get(name, "")


def _repair_outcomes(root: Path, contract: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """What each repair candidate answered, per case.

    ``case_id -> {candidate, tier, status, output}``. The counts in M4 say twelve
    cases were repaired and one broken; this is what lets a reader open those
    thirteen and see what actually changed. Without it a repair is a number.

    Read from each attempt's ``trial_root/outputs.jsonl`` rather than the wire,
    because it is one row per case per candidate — the contract points at the
    file instead of inlining it, and this is the reader following the pointer.
    Confirmation attempts only: the selection sweep chose the candidate and its
    per-case results are not evidence about it.
    """
    out: dict[str, dict[str, Any]] = {}
    for key in sorted(contract):
        if not key.endswith("m4_fix"):
            continue
        for attempt in (contract[key] or {}).get("attempted") or []:
            trial = str(attempt.get("trial_root") or "")
            if not trial:
                continue
            path = (root / trial / "outputs.jsonl")
            if not path.is_file():
                path = Path(trial) / "outputs.jsonl"   # a run read where it was produced
            if not path.is_file():
                continue
            for line in path.read_text(encoding="utf-8").splitlines():
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                cid = str(row.get("case_id") or "")
                if not cid:
                    continue
                out[cid] = {
                    "candidate": attempt.get("name"),
                    "tier": attempt.get("tier"),
                    "status": row.get("status"),
                    "output": row.get("output"),
                }
    return out


def _contract_model_name(contract: Mapping[str, Any]) -> str:
    """The model's name from M1's payload, or "" for a run that emitted none."""
    for key in sorted(contract):
        if not key.endswith(".m1"):
            continue
        ref = (contract[key] or {}).get("model") or {}
        name = str(ref.get("name") or "").strip()
        if name:
            return name
    return ""


def _contract_stats(contract: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """M2's statistical rows from the newest cycle that has any."""
    for key in sorted(contract, reverse=True):
        if not key.endswith(".m2"):
            continue
        rows = (contract[key] or {}).get("stats_results")
        if isinstance(rows, list) and rows:
            return rows
    return []


def _media_index(cases: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """One entry per distinct media file, preferring the copy inside the run.

    ``inputs.<slot>`` is where the producer read the file from — an absolute path
    on that host, and meaningless on any other. The run logger already copied the
    file to ``artifacts/case_media/`` and recorded that relative path, so a run
    unzipped elsewhere still has its media; index by that when it exists and keep
    the original only as the fallback for runs that predate the copy.
    """
    media: list[dict[str, Any]] = []
    seen: set[str] = set()
    for case in cases:
        inputs = case.get("inputs") if isinstance(case.get("inputs"), dict) else case
        # Slots and copies are recorded in the same order, so they zip up.
        local = [p for p in (case.get("_media_paths") or []) if isinstance(p, str)]
        i = 0
        for kind in ("audio", "image", "video"):
            source = inputs.get(kind) or inputs.get(f"{kind}_path")
            if not isinstance(source, str) or not source or source.startswith("<"):
                continue
            path = local[i] if i < len(local) else source
            i += 1
            if path in seen:
                continue
            seen.add(path)
            media.append({"id": f"media-{len(media) + 1}", "kind": kind,
                          "path": path, "source": source})
    return media


def _normalise_case(case: Mapping[str, Any], media: list[dict[str, Any]]) -> dict[str, Any]:
    inputs = case.get("inputs") if isinstance(case.get("inputs"), dict) else {}
    prompt = inputs.get("prompt") or case.get("instruction") or case.get("prompt") or ""
    # Match on either key: the index prefers the in-run copy, and a run without
    # copies still resolves through the producer's original path.
    media_by_path: dict[str, str] = {}
    for item in media:
        media_by_path[item["path"]] = item["id"]
        if item.get("source"):
            media_by_path[item["source"]] = item["id"]
    media_ids = []
    for path in (case.get("_media_paths") or []):
        if isinstance(path, str) and path in media_by_path and media_by_path[path] not in media_ids:
            media_ids.append(media_by_path[path])
    for kind in ("audio", "image", "video"):
        value = inputs.get(kind) or case.get(f"{kind}_path") or case.get(kind)
        if isinstance(value, str) and value in media_by_path \
                and media_by_path[value] not in media_ids:
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


def _merge_recorded_case_evidence(cases: list[dict[str, Any]], root: Path) -> list[dict[str, Any]]:
    """Supplement legacy manifests with the run's recorded baseline I/O.

    Older benchmark exports often keep prompts/media in a manifest while
    baseline answers live separately in ``report/discovery_cases.json``.  This
    is a generic run artifact, not a task-specific fallback.  New JSONL case
    records already contain both and therefore simply win when populated.
    """
    snapshots: list[Mapping[str, Any]] = []
    for base in (root, root.parent):
        value = _load_json(base / "report" / "discovery_cases.json")
        if isinstance(value, list):
            snapshots = [item for item in value if isinstance(item, Mapping)]
            if snapshots:
                break
    if not snapshots:
        return cases
    existing = {str(case.get("id")): case for case in cases}
    for raw_case in snapshots:
        case_id = str(raw_case.get("id") or raw_case.get("case_id") or "")
        if not case_id:
            continue
        update = _normalise_case(raw_case, [])
        target = existing.get(case_id)
        if target is None:
            cases.append(update)
            existing[case_id] = update
            continue
        for key in ("prompt", "expected", "observed"):
            if not target.get(key) and update.get(key):
                target[key] = update[key]
        if target.get("status") in {"", "unknown", "unchanged"} and update.get("status"):
            target["status"] = update["status"]
    return cases


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
