"""Published, machine-consumable schema for ``run_log.jsonl``.

``run_log.jsonl`` is the contract between a diagnosis run and everything
downstream (dashboards, parsers, the agent's own memory).  Until now that
contract lived only in docstrings + a table in ``docs/extending.md`` and an
opaque ``schema_version`` int — a consumer had nothing to *validate* against.

This module is that contract, as code:

* :func:`build_schema` returns a JSON Schema (Draft 2020-12) describing every
  event type, built purely from the stdlib so importing it adds **no runtime
  dependency** (the package's light-install promise is preserved — see
  pyproject ``dependencies``).
* The rendered schema is committed next to this file as
  ``run_log.schema.json`` (shipped as package data) so non-Python consumers can
  validate too; :func:`load_schema` reads it and a drift test keeps the two in
  lock-step.
* :func:`validate_event` / :func:`iter_log_errors` validate real log lines.
  They need the ``jsonschema`` package (a *dev*/optional dep, not core) and
  raise a clear, actionable error if it is missing rather than failing obscurely.

Design note — the schema is intentionally **permissive**: it pins the common
envelope, the event discriminator, required fields and core types, but does
*not* forbid extra properties.  That matches ``RUN_LOG_SCHEMA_VERSION``'s rule
that *additive* fields don't bump the version, so a newer producer emitting an
extra field still validates against an older schema.  What it catches is the
breakage that actually matters: a missing required field, a wrong type, a
malformed timestamp, or an unknown event name.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from evalvitals.eval_agent.run_logger import RUN_LOG_SCHEMA_VERSION

#: The event types emitted to ``run_log.jsonl``.
EVENT_TYPES: tuple[str, ...] = (
    "run_start",
    "probe",
    "explore",
    "analysis",
    "diagnosis",
    "surgery",
    "experiment",
    "fix",
    "loop_end",
    "tool_codegen",
    "tool_registry",
    "agent_decision",
    "agent_tool",
    "case_record",
    "report_published",
)

#: Path to the committed, rendered schema shipped as package data.
SCHEMA_PATH = Path(__file__).with_name("run_log.schema.json")

# Reusable sub-schemas -------------------------------------------------------

_HYPOTHESIS = {
    "type": "object",
    "properties": {
        "statement": {"type": "string"},
        "plain_statement": {"type": ["string", "null"]},
        "failure_mode": {"type": ["string", "null"]},
        "status": {"type": ["string", "null"]},
        "test_design": {"type": ["string", "null"]},
    },
}

_JUDGE_IO = {
    "type": "object",
    "properties": {
        "prompt_path": {"type": "string"},
        "prompt_chars": {"type": "integer"},
        "raw_path": {"type": "string"},
        "raw_chars": {"type": "integer"},
    },
}

# An M2 stats field is either its raw value (object/array) or, once it exceeds
# the inline-size threshold, an externalized {path, n_items, bytes} pointer.
_INLINE_OR_EXTERNAL = {"type": ["object", "array"]}

# Per-event field schemas + the fields each event always carries -------------
# (event, schema_version, ts, trace_id come from the shared envelope.)

_EVENTS: dict[str, dict[str, Any]] = {
    "run_start": {
        "required": [],
        "properties": {
            "evalvitals_version": {"type": "string"},
            "python_version": {"type": "string"},
            "git_commit": {"type": ["string", "null"]},
            "data_fingerprint": {"type": "string"},
            "label_distribution": {"type": "object"},
            "n_cases": {"type": "integer"},
            "model": {"type": "string"},
            # AgenticDiagnoseLoop only: the decision judge (distinct from the M3
            # stage judge already covered by "judge") and its action-turn budget.
            "decision_judge": {"type": "string"},
            "max_actions": {"type": "integer"},
            # In-cycle explore step configured (provider:model of its coder).
            "explorer": {"type": "string"},
        },
    },
    "probe": {
        "required": ["cycle", "analyzers", "findings", "artifact_paths"],
        "properties": {
            "analyzers": {"type": "array", "items": {"type": "string"}},
            "findings": {"type": "object"},
            "artifact_paths": {"type": "object"},
            # Additive M1 fields (permissive schema): result_paths (per-analyzer
            # complete result json), failed_analyzers (selected-but-errored),
            # selected_analyzers — see RunLogger.log_probe.
            "selection_rationale": {"type": "string"},
            "judge_io": _JUDGE_IO,
            "duration_sec": {"type": "number"},
        },
    },
    "analysis": {
        "required": ["cycle", "severity", "n_findings", "findings", "narrative"],
        "properties": {
            "severity": {"type": "string"},
            "n_findings": {"type": "integer"},
            "findings": {"type": "array", "items": {"type": "string"}},
            "narrative": {"type": "string"},
            "conclusion": {"type": "string"},
            "evidence_chain": {"type": "array"},
            "stats_tool_results": _INLINE_OR_EXTERNAL,
            "stats_plan": _INLINE_OR_EXTERNAL,
            "stats_results": _INLINE_OR_EXTERNAL,
            "corrected_rejections": _INLINE_OR_EXTERNAL,
            "visualizations": {"type": "array"},
            "figures": {"type": "array"},
            "judge_io": _JUDGE_IO,
            "duration_sec": {"type": "number"},
        },
    },
    "explore": {
        # Optional in-cycle explore step (VLDiagnoseLoop(explorer=...)): a
        # free-form EDA pass over M1's per-case table between M1 and M2.
        # DESCRIPTIVE ONLY — counts, observations, rendered figure paths and
        # where the report/tables/figures were persisted; the explorer's
        # in-sample candidate verdicts appear only as counts under
        # `adjudication` and never enter the confirmatory M2 family.
        "required": ["cycle", "ok", "n_observations", "n_charts", "n_tables"],
        "properties": {
            "ok": {"type": "boolean"},
            "n_observations": {"type": "integer"},
            "n_charts": {"type": "integer"},
            "n_charts_rendered": {"type": "integer"},
            "n_tables": {"type": "integer"},
            "n_candidate_signals": {"type": "integer"},
            "n_hypotheses": {"type": "integer"},
            "adjudication": {"type": "object"},
            "observations": {"type": "array", "items": {"type": "string"}},
            "caveats": {"type": "array", "items": {"type": "string"}},
            "figures": {"type": "array", "items": {"type": "string"}},
            "attempts": {"type": "integer"},
            "error": {"type": "string"},
            "out_dir": {"type": "string"},
            "report_path": {"type": "string"},
            "duration_sec": {"type": "number"},
        },
    },
    "diagnosis": {
        "required": ["cycle", "model_name", "n_hypotheses", "hypotheses", "raw_judge_output"],
        "properties": {
            "model_name": {"type": ["string", "null"]},
            "n_hypotheses": {"type": "integer"},
            "hypotheses": {"type": "array", "items": _HYPOTHESIS},
            "raw_judge_output": {"type": ["string", "null"]},
            # Provenance of the (UNCONFIRMED) explorer mechanism notes M3 was shown.
            # Descriptive/display-only — never enters M2/M5/fix.
            "referenced_charts": {"type": "array", "items": {"type": "string"}},
            "explore_context_used": {"type": "boolean"},
            "explore_figures": {"type": "array", "items": {"type": "string"}},
            # Whether a (clustered, descriptive-only) FailureModeReport was
            # shown to M3 — additive field, no schema version bump needed.
            "failure_modes_used": {"type": "boolean"},
            "judge_io": _JUDGE_IO,
            "duration_sec": {"type": "number"},
        },
    },
    "surgery": {
        "required": ["cycle", "module", "hypothesis", "failure_mode", "status", "fixed"],
        "properties": {
            "module": {"type": "string"},
            "hypothesis": {"type": "string"},
            "failure_mode": {"type": ["string", "null"]},
            "status": {"type": "string"},
            "fixed": {"type": "boolean"},
            "confidence_score": {"type": ["number", "null"]},
            "evidence": {"type": ["object", "null"]},
            "n_refocused_cases": {"type": ["integer", "null"]},
            "duration_sec": {"type": "number"},
            # Additive: M5 protocol-consistency judge prompt/response, persisted
            # under prompts/ (same pattern as probe/analysis/diagnosis).
            "judge_io": _JUDGE_IO,
        },
    },
    "experiment": {
        "required": ["cycle", "module", "hypothesis", "failure_mode", "status", "fixed"],
        "properties": {
            "module": {"type": "string"},
            "hypothesis": {"type": "string"},
            "failure_mode": {"type": ["string", "null"]},
            "status": {"type": ["string", "null"]},
            "fixed": {"type": "boolean"},
            "provider": {"type": ["string", "null"]},
            "metrics": {"type": ["object", "null"]},
            "returncode": {"type": ["integer", "null"]},
            "timed_out": {"type": ["boolean", "null"]},
            "llm_calls": {"type": ["integer", "null"]},
            "code_paths": {"type": "object"},
            "output_paths": {"type": "object"},
            "workspace_snapshot": {"type": ["object", "null"]},
            "record": {"type": "string"},
        },
    },
    "fix": {
        "required": ["cycle", "module"],
        "properties": {
            "module": {"type": "string"},
            "record": {"type": "string"},
        },
    },
    "loop_end": {
        "required": ["cycles"],
        "properties": {
            "cycles": {"type": "integer"},
            "tokens_used": {"type": ["integer", "null"]},
            "timings_sec": {"type": "object"},
            "total_duration_sec": {"type": "number"},
            "resolved": {"type": "boolean"},
            "n_hypotheses": {"type": "integer"},
            "final_hypotheses": {"type": "array", "items": _HYPOTHESIS},
            "stopped_by": {"type": ["string", "null"]},
            "n_verified": {"type": "integer"},
            "verified_hypotheses": {"type": "array"},
        },
    },
    "tool_codegen": {
        "required": ["cycle", "module", "tool_name", "need", "source", "ok"],
        "properties": {
            "module": {"type": "string"},
            "tool_name": {"type": "string"},
            "need": {"type": "string"},
            "source": {"type": "string"},
            "ok": {"type": "boolean"},
            "error": {"type": ["string", "null"]},
            "code_paths": {"type": "object"},
        },
    },
    "tool_registry": {
        "required": ["cycle", "module", "n_tools", "tools"],
        "properties": {
            "module": {"type": "string"},
            "n_tools": {"type": "integer"},
            "tools": {"type": "array"},
        },
    },
    "agent_decision": {
        "required": ["step", "action", "valid"],
        "properties": {
            "step": {"type": "integer"},
            "action": {"type": "string"},
            "params": {"type": "object"},
            "rationale": {"type": "string"},
            "valid": {"type": "boolean"},
            "repair_attempts": {"type": "integer"},
            "fallback_used": {"type": "boolean"},
            "judge_io": _JUDGE_IO,
            "duration_sec": {"type": "number"},
        },
    },
    "agent_tool": {
        "required": ["step", "tool", "ok"],
        "properties": {
            "step": {"type": "integer"},
            "tool": {"type": "string"},
            "ok": {"type": "boolean"},
            "summary": {"type": "string"},
            "error": {"type": ["string", "null"]},
            "duration_sec": {"type": "number"},
        },
    },
    "case_record": {
        "required": ["case_id", "case", "media_paths"],
        "properties": {
            "case_id": {"type": "string"},
            "case": {"type": "object"},
            "media_paths": {"type": "array", "items": {"type": "string"}},
        },
    },
    "report_published": {
        "required": ["report_schema_version", "catalog_version", "sha256", "generated_by"],
        "properties": {
            "report_schema_version": {"type": "integer"},
            "catalog_version": {"type": "string"},
            "json_render_version": {"type": "string"},
            "source_event_seq": {"type": "integer"},
            "sha256": {"type": "string"},
            "generated_by": {"type": "object"},
            "report_paths": {"type": "array", "items": {"type": "string"}},
        },
    },
}


def build_schema() -> dict[str, Any]:
    """Return the JSON Schema (Draft 2020-12) for ``run_log.jsonl`` events.

    Built from the stdlib only — no third-party import — so this stays callable
    in the light install.  The single source of truth for the version pin is
    :data:`~evalvitals.eval_agent.run_logger.RUN_LOG_SCHEMA_VERSION`.
    """
    envelope = {
        "type": "object",
        "required": ["event", "schema_version", "ts", "trace_id"],
        "properties": {
            "event": {"type": "string", "enum": list(EVENT_TYPES)},
            "schema_version": {"const": RUN_LOG_SCHEMA_VERSION},
            # `format` is for documentation/interop; jsonschema only enforces it
            # when the optional rfc3339-validator is installed, so a `pattern`
            # pins the ISO-8601 shape unconditionally (it matches the producer's
            # datetime.now(UTC).isoformat(timespec="microseconds")).
            "ts": {
                "type": "string",
                "format": "date-time",
                "pattern": r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?(Z|[+-]\d{2}:\d{2})?$",
            },
            "trace_id": {"type": "string"},
            "event_seq": {"type": "integer", "minimum": 1},
            "span_id": {"type": "string"},
            "cycle": {"type": "integer"},
        },
    }

    defs: dict[str, Any] = {"envelope": envelope}
    branches: list[dict[str, Any]] = []
    for name, spec in _EVENTS.items():
        branch = {
            "allOf": [
                {"$ref": "#/$defs/envelope"},
                {
                    "type": "object",
                    "properties": {"event": {"const": name}, **spec["properties"]},
                    "required": ["event", *spec["required"]],
                },
            ]
        }
        defs[name] = branch
        branches.append({"$ref": f"#/$defs/{name}"})

    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "https://evalvitals.dev/schemas/run_log.schema.json",
        "title": "EvalVitals run_log.jsonl event",
        "description": (
            "One event per line in run_log.jsonl. Discriminated on `event`; "
            f"matches schema_version {RUN_LOG_SCHEMA_VERSION}. Permissive: extra "
            "(additive) fields are allowed."
        ),
        "$defs": defs,
        "oneOf": branches,
    }


def load_schema() -> dict[str, Any]:
    """Load the committed, rendered :data:`SCHEMA_PATH` (the shipped artifact)."""
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


def _validator(schema: dict[str, Any] | None = None):
    """Build a ``jsonschema`` validator with the date-time format checker on.

    Raises a clear :class:`ImportError` if ``jsonschema`` (an optional dep) is
    not installed, so the failure points at the fix instead of an obscure
    ``ModuleNotFoundError`` deep in a call stack.
    """
    try:
        import jsonschema
    except ImportError as exc:  # pragma: no cover - exercised via message only
        raise ImportError(
            "validating run_log.jsonl needs the 'jsonschema' package — "
            "install it with `pip install jsonschema` or `pip install evalvitals[dev]`."
        ) from exc
    schema = schema if schema is not None else build_schema()
    cls = jsonschema.validators.validator_for(schema)
    cls.check_schema(schema)
    return cls(schema, format_checker=cls.FORMAT_CHECKER)


def validate_event(event: dict[str, Any], *, schema: dict[str, Any] | None = None) -> None:
    """Validate a single decoded event; raise ``jsonschema.ValidationError`` if bad."""
    _validator(schema).validate(event)


def iter_log_errors(path: str | Path, *, schema: dict[str, Any] | None = None):
    """Yield ``(line_number, message)`` for every invalid line in a JSONL log.

    A convenience for CI / downstream consumers: parse-and-validate a whole
    ``run_log.jsonl`` in one call.  Lines that aren't valid JSON are reported
    too.  An empty iterator means the file fully conforms.
    """
    validator = _validator(schema)
    with open(path, encoding="utf-8") as fh:
        for i, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError as exc:
                yield i, f"invalid JSON: {exc}"
                continue
            for err in validator.iter_errors(event):
                yield i, err.message
