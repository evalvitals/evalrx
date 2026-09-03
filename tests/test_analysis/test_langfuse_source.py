"""Langfuse observations must rehydrate the renderer's event contract."""

from __future__ import annotations

import json
from types import SimpleNamespace


def _observation(seq: int, event: dict[str, object]):
    return SimpleNamespace(
        input=json.dumps({"event": event}),
        metadata={"event_seq": seq},
    )


def test_langfuse_source_paginates_and_orders_events(tmp_path):
    from evalrx.reporting.langfuse_source import LangfuseRunSource

    calls = []

    class Observations:
        def get_many(self, **kwargs):
            calls.append(kwargs)
            if kwargs["cursor"] is None:
                return SimpleNamespace(
                    data=[_observation(2, {"event": "analysis", "event_seq": 2})],
                    meta=SimpleNamespace(cursor="next"),
                )
            return SimpleNamespace(
                data=[_observation(1, {"event": "run_start", "event_seq": 1})],
                meta=SimpleNamespace(cursor=None),
            )

    client = SimpleNamespace(api=SimpleNamespace(observations=Observations()))
    source = LangfuseRunSource(client)

    events = source.events("00000000-0000-0000-0000-000000000001")
    root = source.materialize("00000000-0000-0000-0000-000000000001", tmp_path / "cache")

    assert [event["event"] for event in events] == ["run_start", "analysis"]
    assert calls[0]["trace_id"] == "00000000000000000000000000000001"
    assert [json.loads(line)["event"] for line in (root / "run_log.jsonl").read_text().splitlines()] == [
        "run_start", "analysis"
    ]


def test_langfuse_source_ignores_non_evalrx_observations():
    from evalrx.reporting.langfuse_source import LangfuseRunSource

    class Observations:
        def get_many(self, **_kwargs):
            return SimpleNamespace(data=[SimpleNamespace(input={"other": True}, metadata={})], meta=SimpleNamespace(cursor=None))

    source = LangfuseRunSource(SimpleNamespace(api=SimpleNamespace(observations=Observations())))
    assert source.events("trace") == []


def test_langfuse_source_materializes_media_only_inside_cache(tmp_path):
    from evalrx.reporting.langfuse_source import LangfuseRunSource

    attachment = {"artifact_id": "sha256:abc", "content": "@@@langfuseMedia:type=audio/wav|id=x@@@"}
    observation = SimpleNamespace(
        input={"event": {"event": "run_start"}, "artifacts": [attachment]},
        metadata={"event_seq": 1, "artifact_refs": [{"artifact_id": "sha256:abc", "path": "audio/test.wav"}]},
    )

    class Client:
        api = SimpleNamespace(observations=SimpleNamespace(
            get_many=lambda **_kwargs: SimpleNamespace(data=[observation], meta=SimpleNamespace(cursor=None))
        ))

        def resolve_media_references(self, *, obj, resolve_with):
            assert resolve_with == "base64_data_uri"
            assert obj == [attachment]
            return [{"artifact_id": "sha256:abc", "content": "data:audio/wav;base64,YXVkaW8="}]

    root = LangfuseRunSource(Client()).materialize("trace", tmp_path / "cache")
    assert (root / "audio" / "test.wav").read_bytes() == b"audio"


def test_materialized_langfuse_run_builds_the_existing_html_report(tmp_path):
    from evalrx.reporting.html_report import build_html_report
    from evalrx.reporting.langfuse_source import LangfuseRunSource

    observation = SimpleNamespace(
        input={"event": {"event": "run_start", "trace_id": "trace", "event_seq": 1, "protocol": {}}},
        metadata={"event_seq": 1},
    )
    client = SimpleNamespace(api=SimpleNamespace(observations=SimpleNamespace(
        get_many=lambda **_kwargs: SimpleNamespace(data=[observation], meta=SimpleNamespace(cursor=None))
    )))

    root = LangfuseRunSource(client).materialize("trace", tmp_path / "cache")
    report = build_html_report(root, no_audio=True)

    assert report.exists()
