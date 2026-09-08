"""Tests for dynamic HTML report generator and Langfuse exporter."""

import json
from pathlib import Path

from evalrx.reporting.html_report import build_html_report, embed_figures, extract_run_data
from evalrx.reporting.langfuse_exporter import export_to_langfuse_bundle


def _png(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"fake")


def test_embed_figures_finds_v1_figures_dir(tmp_path: Path):
    _png(tmp_path / "figures" / "m2_effects.png")
    out = embed_figures(None, tmp_path)
    assert "m2_effects" in out and out["m2_effects"].startswith("data:image/png;base64,")


def test_embed_figures_finds_v2_m2_artifacts(tmp_path: Path):
    """Regression test: embed_figures used to only glob logs_dir/figures, so
    RunContext.figures_dir's V2 mapping (M2/artifacts, run_context.py) meant
    every V2 run's static HTML export rendered zero embedded M2 charts."""
    _png(tmp_path / "M2" / "artifacts" / "m2_effects.png")
    out = embed_figures(None, tmp_path)
    assert "m2_effects" in out and out["m2_effects"].startswith("data:image/png;base64,")


def test_embed_figures_combines_v1_and_v2_sources_without_clobbering(tmp_path: Path):
    _png(tmp_path / "figures" / "shared.png")
    _png(tmp_path / "M2" / "artifacts" / "other.png")
    out = embed_figures(None, tmp_path)
    assert {"shared", "other"} <= set(out)


def test_html_report_generation(tmp_path: Path):
    # Test against real MMAU outputs
    run_dir = Path("examples/m1_m5/mmau_qwen2_audio/outputs")
    example_dir = Path("examples/m1_m5/mmau_qwen2_audio")
    out_file = tmp_path / "test_report.html"

    if run_dir.exists():
        data = extract_run_data(run_dir, example_dir)
        assert data["run"]["model"] is not None
        assert "m1" in data
        assert "m2" in data
        assert "m3" in data

        res_path = build_html_report(
            run_dir=run_dir,
            example_dir=example_dir,
            out_path=out_file,
            no_audio=True,
        )
        assert res_path.exists()
        content = res_path.read_text(encoding="utf-8")
        assert "EvalRX" in content
        assert "What this report says" in content
        assert "What we checked" in content
        assert "What patterns we found" in content
        assert "Possible explanation to test" in content
        assert "Repair attempt" in content
        assert "fonts.googleapis.com" not in content
        assert "fonts.gstatic.com" not in content


def test_langfuse_bundle_export(tmp_path: Path):
    run_dir = Path("examples/m1_m5/mmau_qwen2_audio/outputs")
    out_json = tmp_path / "langfuse_trace.json"

    if run_dir.exists():
        bundle = export_to_langfuse_bundle(run_dir, out_json)
        assert "trace" in bundle
        assert "spans" in bundle
        assert "scores" in bundle
        assert out_json.exists()
        loaded = json.loads(out_json.read_text(encoding="utf-8"))
        assert loaded["trace"]["name"].startswith("EvalRX:")


def test_report_uses_latest_trace_and_m4_status(tmp_path: Path):
    """An appended run must not inherit M1/M4 state from an earlier trace."""
    events = [
        {"event": "run_start", "trace_id": "old", "model": "old", "protocol": {}},
        {"event": "probe", "trace_id": "old", "cycle": 0, "analyzers": ["old_probe"]},
        {
            "event": "run_start",
            "trace_id": "new",
            "model": "new",
            "protocol": {},
            "n_cases": 1,
        },
        {"event": "probe", "trace_id": "new", "cycle": 2, "analyzers": ["new_probe"]},
        {
            "event": "surgery",
            "trace_id": "new",
            "module": "m4",
            "status": "supported",
            "fixed": False,
            "confidence_score": 0.8,
            "evidence": {
                "m4_test_name": "association",
                "m4_effect_size": 0.3,
                "m4_verdict": "supported by held-out evidence",
                "m4_evidence_grade": "causal",
            },
        },
    ]
    (tmp_path / "run_log.jsonl").write_text(
        "\n".join(json.dumps(event) for event in events), encoding="utf-8"
    )

    data = extract_run_data(tmp_path)

    assert data["run"]["raw_model"] == "new"
    assert data["m1"]["analyzers"] == ["new_probe"]
    assert data["m4"]["results"][0]["status"] == "supported"


def test_report_escapes_script_terminators_in_case_data(tmp_path: Path):
    """Manifest/model text must not be able to escape the CASES script block."""
    from evalrx.reporting.html_report import generate_html_report

    (tmp_path / "run_log.jsonl").write_text(
        json.dumps({"event": "run_start", "trace_id": "t", "protocol": {}}), encoding="utf-8"
    )
    data = extract_run_data(tmp_path)
    payload = "</script><script>globalThis.pwned = true</script>"
    data["cases"] = [{
        "id": "case-1",
        "status": "unchanged",
        "output": "",
        "instruction": payload,
        "choices": [],
        "expected": "",
        "duration": 0.0,
        "task": "",
        "probe_flags": [],
    }]

    page = generate_html_report(data, {}, {}, {})

    assert payload not in page
    assert "\\u003c/script\\u003e" in page
