"""A run's logged events have a published schema that is drift-free.

These tests are the teeth behind log_schema.py: they prove the committed
``run_log.schema.json`` is (1) in sync with the schema-as-code builder, and
(2) a well-formed Draft 2020-12 schema. (3) — that what a run actually
emits for *every* event type validates against it — lives in
``test_run_logger_v2.py``'s
``test_persisted_event_identity_orders_concurrent_calls_and_validates``,
next to the producer it's checking.
"""

from __future__ import annotations

import json

import pytest

jsonschema = pytest.importorskip("jsonschema")


def test_committed_schema_matches_builder():
    """The shipped JSON file must equal build_schema() — re-render after edits."""
    from evalrx.eval_agent.log_schema import SCHEMA_PATH, build_schema

    committed = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    assert committed == build_schema(), (
        "run_log.schema.json is stale. Re-render it:\n"
        "  python -c \"import json; from evalrx.eval_agent.log_schema import "
        "build_schema, SCHEMA_PATH; SCHEMA_PATH.write_text(json.dumps(build_schema(), "
        "indent=2)+chr(10))\""
    )


def test_schema_is_valid_draft_2020_12():
    from evalrx.eval_agent.log_schema import build_schema

    schema = build_schema()
    cls = jsonschema.validators.validator_for(schema)
    cls.check_schema(schema)
    assert schema["$schema"].endswith("2020-12/schema")


def test_validation_rejects_malformed_events():
    """The schema must actually reject the breakage it claims to catch."""
    from evalrx.eval_agent.log_schema import RUN_LOG_SCHEMA_VERSION, validate_event

    base = {
        "event": "probe", "schema_version": RUN_LOG_SCHEMA_VERSION,
        "ts": "2026-06-22T18:57:33.296014+00:00", "trace_id": "t", "cycle": 0,
        "event_seq": 1, "stage": "M1", "span_id": "c0.m1",
        "analyzers": [], "findings": {}, "artifact_paths": {},
    }
    validate_event(base)  # the valid baseline must pass

    # unknown event name
    with pytest.raises(jsonschema.ValidationError):
        validate_event({**base, "event": "not_a_real_event"})
    # missing a required field (probe needs findings)
    bad = {k: v for k, v in base.items() if k != "findings"}
    with pytest.raises(jsonschema.ValidationError):
        validate_event(bad)
    # malformed timestamp
    with pytest.raises(jsonschema.ValidationError):
        validate_event({**base, "ts": "not-a-timestamp"})
    # wrong type for a core field
    with pytest.raises(jsonschema.ValidationError):
        validate_event({**base, "cycle": "zero"})
