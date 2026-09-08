"""Reliability contract for the Langfuse delivery outbox."""

from __future__ import annotations


def _event(seq: int = 1) -> dict[str, object]:
    from evalrx.observability.envelope import make_event_envelope

    return make_event_envelope(
        {"event": "analysis", "ts": "2026-08-20T00:00:00+00:00", "cycle": 0},
        trace_id="00000000-0000-0000-0000-000000000001",
        event_seq=seq,
    )


def test_outbox_is_idempotent_and_drains_in_sequence(tmp_path):
    from evalrx.observability.outbox import ObservabilityOutbox

    outbox = ObservabilityOutbox(tmp_path / "outbox.sqlite3")
    outbox.enqueue(_event(2))
    outbox.enqueue(_event(1))
    outbox.enqueue(_event(1))
    delivered: list[int] = []

    published, failed = outbox.drain(lambda event: delivered.append(int(event["event_seq"])))

    assert (published, failed) == (2, 0)
    assert delivered == [1, 2]
    assert outbox.pending_count() == 0


def test_failed_delivery_remains_durable(tmp_path):
    from evalrx.observability.outbox import ObservabilityOutbox

    outbox = ObservabilityOutbox(tmp_path / "outbox.sqlite3")
    outbox.enqueue(_event())

    published, failed = outbox.drain(lambda _event: (_ for _ in ()).throw(ConnectionError("offline")))

    assert (published, failed) == (0, 1)
    assert outbox.pending_count() == 1


def test_artifact_manifest_is_content_addressed(tmp_path):
    from evalrx.observability.envelope import artifact_manifests_for_event

    artifact = tmp_path / "artifacts" / "result.json"
    artifact.parent.mkdir()
    artifact.write_text('{"ok": true}', encoding="utf-8")

    manifests = artifact_manifests_for_event(
        {"event": "probe", "result_paths": {"probe": "artifacts/result.json"}}, run_dir=tmp_path,
    )

    assert manifests[0]["path"] == "artifacts/result.json"
    assert manifests[0]["artifact_id"].startswith("sha256:")
    assert manifests[0]["mime_type"] == "application/json"


def test_skipped_stage_keeps_its_pipeline_stage_in_the_envelope():
    from evalrx.observability.envelope import make_event_envelope

    envelope = make_event_envelope(
        {"event": "stage_skipped", "stage": "M5", "reason_code": "no_accepted_hypothesis"},
        trace_id="trace", event_seq=1,
    )

    assert envelope["stage"] == "M5"


def test_backfill_dry_run_preserves_existing_trace_and_order(tmp_path):
    from evalrx.eval_agent.run_logger_v2 import RunLoggerV2
    from evalrx.observability import backfill_run_to_langfuse

    trace_id = "00000000-0000-0000-0000-000000000001"
    logger = RunLoggerV2(tmp_path, trace_id=trace_id, observability_mode="offline")
    logger.log_run_start({})
    logger._append_stage("M2", "analysis", {})
    logger.close()

    summary = backfill_run_to_langfuse(tmp_path, dry_run=True)

    assert summary == {"trace_id": trace_id, "events": 2, "pending": 2, "published": 0}


def test_tracer_flushes_events_through_langfuse_client(tmp_path):
    from evalrx.observability.tracer import DiagnosticTracer

    class Client:
        def __init__(self):
            self.events = []

        def create_event(self, **kwargs):
            self.events.append(kwargs)

        def flush(self):
            pass

    tracer = DiagnosticTracer(tmp_path)
    client = Client()
    tracer._langfuse_client = client
    tracer.trace_id = "00000000-0000-0000-0000-000000000001"
    tracer.record_event({"event": "analysis", "ts": "2026-08-20T00:00:00+00:00"}, event_seq=1)
    tracer.flush()

    assert len(client.events) == 1
    assert client.events[0]["metadata"]["event_seq"] == 1
    assert tracer.outbox.pending_count() == 0


def test_tracer_nests_events_under_the_live_run_chain(tmp_path):
    from evalrx.observability.tracer import DiagnosticTracer

    class Event:
        def end(self):
            pass

    class Root:
        def __init__(self):
            self.calls = []

        def start_observation(self, **kwargs):
            self.calls.append(kwargs)
            return Event()

    class Client:
        def flush(self):
            pass

    tracer = DiagnosticTracer(tmp_path)
    tracer._langfuse_client = Client()
    tracer._live_root = Root()
    tracer.record_event({"event": "analysis", "ts": "2026-08-20T00:00:00+00:00"}, event_seq=1)
    tracer.flush()

    assert tracer._live_root.calls[0]["as_type"] == "event"
    assert tracer._live_root.calls[0]["metadata"]["event_seq"] == 1


def test_live_auto_sync_drains_the_outbox_at_event_time(tmp_path):
    from evalrx.observability.tracer import DiagnosticTracer

    class Client:
        def __init__(self):
            self.events = []

        def create_event(self, **kwargs):
            self.events.append(kwargs)

        def flush(self):
            pass

    tracer = DiagnosticTracer(tmp_path, auto_sync=True)
    tracer._langfuse_client = Client()
    tracer.record_event({"event": "analysis", "ts": "2026-08-20T00:00:00+00:00"}, event_seq=1)

    assert len(tracer._langfuse_client.events) == 1
    assert tracer.outbox.pending_count() == 0
