"""Tests for dynamic HTML report generator and Langfuse exporter."""

import json
from pathlib import Path

from evalvitals.reporting.html_report import build_html_report, extract_run_data
from evalvitals.reporting.langfuse_exporter import export_to_langfuse_bundle


def test_html_report_generation(tmp_path: Path):
    # Test against real MMAU outputs
    run_dir = Path("examples/m1_m4/mmau_qwen2_audio/outputs")
    example_dir = Path("examples/m1_m4/mmau_qwen2_audio")
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
        assert "EvalVitals" in content
        assert "Checkup & Vital Signals" in content
        assert "Screening & Confirmatory Signals" in content
        assert "Root-Cause Diagnosis" in content
        assert "Targeted Repair & Paired Confirmation" in content


def test_langfuse_bundle_export(tmp_path: Path):
    run_dir = Path("examples/m1_m4/mmau_qwen2_audio/outputs")
    out_json = tmp_path / "langfuse_trace.json"

    if run_dir.exists():
        bundle = export_to_langfuse_bundle(run_dir, out_json)
        assert "trace" in bundle
        assert "spans" in bundle
        assert "scores" in bundle
        assert out_json.exists()
        loaded = json.loads(out_json.read_text(encoding="utf-8"))
        assert loaded["trace"]["name"].startswith("EvalVitals:")
