"""label_quarantine: the run directory's files are off disk while the fix stage
runs and back, byte for byte, afterwards -- including under an exception, with
an append log a live FileHandler holds open, and when the fix stage re-creates
a path with new content."""

from __future__ import annotations

import json
import logging
import os

import pytest

from evalvitals.eval_agent.label_quarantine import MANIFEST_NAME, quarantine_run_dir


def _seed(root):
    (root / "logs" / "report").mkdir(parents=True)
    (root / "logs" / "contract").mkdir()
    (root / "explore").mkdir()
    (root / "baseline.json").write_text(json.dumps([{"id": "c-0", "expected": "Yes", "label": "fail"}]))
    (root / "logs" / "report" / "discovery_cases.json").write_text('[{"expected": "Yes"}]')
    (root / "logs" / "contract" / "c0.m1.json").write_text('{"gold_yes": 1}')
    (root / "explore" / "records.json").write_bytes(b"\x00binary\xff" * 10)
    return {p.relative_to(root).as_posix(): p.read_bytes() for p in root.rglob("*") if p.is_file()}


def test_everything_is_hidden_inside_and_restored_byte_identical_after(tmp_path):
    before = _seed(tmp_path)

    with quarantine_run_dir(tmp_path, write_manifest=False) as q:
        visible = [p for p in tmp_path.rglob("*") if p.is_file()]
        assert visible == [], visible                      # nothing readable
        assert (tmp_path / "logs" / "contract").is_dir()   # directories stay
        assert sorted(q.hidden) == sorted(before)
        assert q.n_bytes == sum(len(b) for b in before.values())
        # what the fix stage writes is untouched by the restore
        (tmp_path / "logs" / "fixes").mkdir()
        (tmp_path / "logs" / "fixes" / "outcome.md").write_text("NOT FIXED")

    after = {p.relative_to(tmp_path).as_posix(): p.read_bytes()
             for p in tmp_path.rglob("*") if p.is_file()}
    assert after.pop("logs/fixes/outcome.md") == b"NOT FIXED"
    assert after == before


def test_manifest_names_what_was_hidden(tmp_path):
    before = _seed(tmp_path)
    with quarantine_run_dir(tmp_path):
        pass
    manifest = json.loads((tmp_path / MANIFEST_NAME).read_text())
    assert manifest["n_hidden"] == len(before)
    assert set(manifest["hidden"]) == set(before)
    assert manifest["conflicts_saved_as_pre_fix"] == []
    assert manifest["restored_at"]


def test_append_log_held_open_by_a_filehandler_is_truncated_then_merged(tmp_path):
    (tmp_path / "logs").mkdir()
    log = tmp_path / "logs" / "run_log.jsonl"
    log.write_text('{"event": "case_record", "expected": "Yes"}\n')
    lg = logging.getLogger(f"quarantine-test-{os.getpid()}")
    lg.propagate = False
    handler = logging.FileHandler(log, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(message)s"))
    lg.addHandler(handler)
    lg.setLevel(logging.INFO)
    try:
        with quarantine_run_dir(tmp_path, append_logs=[log], write_manifest=False) as q:
            assert log.exists() and log.stat().st_size == 0   # truncated, not unlinked
            assert q.truncated == ["logs/run_log.jsonl"]
            lg.info('{"event": "fix"}')                       # the live handler keeps writing
            handler.flush()
            assert log.read_text() == '{"event": "fix"}\n'    # ...into the same file
        assert log.read_text() == ('{"event": "case_record", "expected": "Yes"}\n'
                                   '{"event": "fix"}\n')
        assert q.merged == ["logs/run_log.jsonl"]
    finally:
        lg.removeHandler(handler)
        handler.close()


def test_a_path_rewritten_during_the_block_keeps_the_new_content_and_saves_the_old(tmp_path):
    (tmp_path / "logs" / "contract").mkdir(parents=True)
    idx = tmp_path / "logs" / "contract" / "index.json"
    idx.write_text('{"payloads": ["old"]}')
    with quarantine_run_dir(tmp_path, write_manifest=False) as q:
        idx.write_text('{"payloads": ["new"]}')
    assert idx.read_text() == '{"payloads": ["new"]}'
    side = tmp_path / "logs" / "contract" / "index.json.pre_fix"
    assert side.read_text() == '{"payloads": ["old"]}'
    assert q.conflicts == ["logs/contract/index.json"]
    # the copy must not look like a contract payload to a `*.json` scan
    assert not side.name.endswith(".json")
    assert [p.name for p in (tmp_path / "logs" / "contract").glob("*.json")] == ["index.json"]


def test_restore_happens_even_when_the_fix_stage_raises(tmp_path):
    before = _seed(tmp_path)
    with pytest.raises(RuntimeError, match="coder died"):
        with quarantine_run_dir(tmp_path, write_manifest=False):
            assert not (tmp_path / "baseline.json").exists()
            raise RuntimeError("coder died")
    after = {p.relative_to(tmp_path).as_posix(): p.read_bytes()
             for p in tmp_path.rglob("*") if p.is_file()}
    assert after == before


def test_keep_prefixes_stay_visible(tmp_path):
    _seed(tmp_path)
    (tmp_path / "logs" / "fixes").mkdir()
    (tmp_path / "logs" / "fixes" / "01_prev.json").write_text("{}")
    with quarantine_run_dir(tmp_path, keep=("logs/fixes",), write_manifest=False) as q:
        assert (tmp_path / "logs" / "fixes" / "01_prev.json").exists()
        assert not (tmp_path / "baseline.json").exists()
        assert q.kept == ["logs/fixes/01_prev.json"]


def test_symlinks_are_left_alone_and_their_targets_untouched(tmp_path):
    outside = tmp_path.parent / f"{tmp_path.name}_outside.txt"
    outside.write_text("gold lives elsewhere")
    root = tmp_path / "run"
    root.mkdir()
    (root / "baseline.json").write_text("[]")
    (root / "link.txt").symlink_to(outside)
    with quarantine_run_dir(root, write_manifest=False) as q:
        assert (root / "link.txt").is_symlink()
        assert q.hidden == ["baseline.json"]
    assert outside.read_text() == "gold lives elsewhere"
    assert (root / "baseline.json").read_text() == "[]"


def test_missing_run_dir_is_a_no_op(tmp_path):
    with quarantine_run_dir(tmp_path / "never", write_manifest=True) as q:
        assert q.hidden == []
    assert not (tmp_path / "never").exists()
