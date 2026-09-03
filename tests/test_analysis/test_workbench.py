"""Regression coverage for the filesystem-backed web workbench primitives."""

from __future__ import annotations

import io
import json
import zipfile

import pytest

from evalrx.analysis.workbench import (
    EventSink,
    ThreadStore,
    UploadLimits,
    extract_archive,
    ingest_directory,
)


def _archive(entries: dict[str, str]) -> bytes:
    raw = io.BytesIO()
    with zipfile.ZipFile(raw, "w") as zf:
        for name, body in entries.items():
            zf.writestr(name, body)
    return raw.getvalue()


def test_extract_archive_skips_platform_junk_and_rejects_duplicate_targets(tmp_path):
    root = extract_archive(_archive({
        "dataset/rows.csv": "x,label\n1,PASS\n",
        "dataset/.DS_Store": "junk",
        "__MACOSX/dataset/._rows.csv": "junk",
    }), tmp_path / "data")
    assert root == tmp_path / "data" / "dataset"
    assert (root / "rows.csv").exists()
    assert not (root / ".DS_Store").exists()

    with pytest.raises(ValueError, match="duplicate"):
        extract_archive(_archive({"x": "a", "./x": "b"}), tmp_path / "duplicate")


def test_ingest_directory_normalizes_jsonl_csv_and_image_metadata(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "events.jsonl").write_text('{"value": 1, "label": "PASS"}\n{"value": 2, "label": "FAIL"}\n')
    (source / "extra.csv").write_text("group,cost\na,1.5\nb,2.5\n")
    pil = pytest.importorskip("PIL.Image")
    pil.new("RGB", (3, 2)).save(source / "sample.png")

    bundle = ingest_directory(source, tmp_path / "bundle")
    records = json.loads((tmp_path / "bundle" / "records.json").read_text())
    media = [json.loads(line) for line in (tmp_path / "bundle" / "media_units.jsonl").read_text().splitlines()]
    assert bundle.n_records == 4
    assert {row["_source_file"] for row in records} == {"events.jsonl", "extra.csv"}
    assert media[0]["kind"] == "image"
    assert media[0]["width"] == 3 and media[0]["height"] == 2
    assert (tmp_path / "bundle" / media[0]["bundle_path"]).exists()


def test_event_sink_and_thread_store_are_durable_and_ordered(tmp_path):
    thread = ThreadStore(tmp_path).create(name="study", provider="codex")
    sink = EventSink(thread / "events.jsonl", thread_id=thread.name, turn_id="initial")
    assert sink.emit("ingest", "started", "Reading files").seq == 1
    assert sink.emit("m2", "completed", "Done").seq == 2
    ThreadStore.append_message(thread, "user", "What stands out?", turn_id="initial")
    ThreadStore.set_current_turn(thread, "turn_001")

    events = [json.loads(line) for line in (thread / "events.jsonl").read_text().splitlines()]
    assert [event["seq"] for event in events] == [1, 2]
    assert json.loads((thread / "thread.json").read_text())["current_turn"] == "turn_001"


def test_extract_archive_applies_expansion_limits(tmp_path):
    with pytest.raises(ValueError, match="too many files"):
        extract_archive(
            _archive({f"f{i}.json": "{}" for i in range(3)}), tmp_path / "data",
            limits=UploadLimits(max_files=2),
        )
