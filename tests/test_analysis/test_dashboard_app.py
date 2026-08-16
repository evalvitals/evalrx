"""The loop dashboard renders a data-rich Analysis tab (signal effects, e-BH
adjudication, explorer tables), not just the M2→M3→M5→Fix flow narrative.

Guarded on streamlit/pandas (the optional dashboard extras).
"""

from __future__ import annotations

import json
import pickle
import sys
from types import SimpleNamespace

import pytest

pytest.importorskip("streamlit")
pytest.importorskip("pandas")


def _build_loop_run(root):
    fused = root / "fused"
    (fused / "sandbox" / "tables").mkdir(parents=True)
    logs = root / "logs_m2_5"
    logs.mkdir(parents=True)

    (fused / "sandbox" / "tables" / "fail_by_signal.csv").write_text(
        "signal_value,fail_rate\nabsent,0.62\npresent,0.11\n", encoding="utf-8"
    )
    (fused / "fused_report.json").write_text(json.dumps({
        "observations": ["FAIL skews to small objects"],
        "visual_plan": [{
            "name": "fail_by_signal",
            "question": "Does the signal alter failure rate?",
            "data_shape": "categorical-vs-binary",
            "plot_kind": "bar",
            "fallback_kind": "bar",
            "required_columns": ["signal_value", "label"],
            "rationale": "A fail-rate bar is readable for two signal states.",
        }],
        "chart_readings": [{
            "chart": "Fail rate by signal",
            "reading": "The present state has a lower failure rate in this toy fixture.",
            "do_not_infer": "This chart alone is not causal.",
        }],
        "dashboard_storyboard": [{
            "id": "analysis",
            "title": "Analysis",
            "stages": ["M2"],
            "summary": "Agent-authored dashboard narrative for this run.",
            "items": ["Method: held-out comparison", "Takeaway: inspect the signal"],
            "artifact_refs": ["candidate_signals", "charts"],
        }],
        "claims": [{
            "id": "A1",
            "text": "The signal is lower risk when present.",
            "status": "descriptive",
            "evidence_ids": ["chart:fail_rate_by_signal"],
            "interpretation": "Use as an exploratory reading.",
            "do_not_infer": "No causal claim.",
        }],
        "critique": ["toy fixture; no causal interpretation"],
        "caveats": ["probe1 ~ label (circular)"],
        "charts": [{"name": "c", "kind": "bar", "data": "tables/fail_by_signal.csv",
                    "x": "signal_value", "y": "fail_rate", "title": "Fail rate by signal"}],
        "candidate_signals": [
            {"name": "probe1_flag", "source": "explorer", "effect": 0.86, "reject": True,
             "recipe": {"expr": "probe1 > 0"}},
            {"name": "diffuse_attention", "source": "explorer", "effect": None, "reject": None},
            {"name": "probe1_or_diffuse", "source": "explorer", "effect": 0.40, "reject": True},
        ],
        "adjudication": {"method": "e-BH", "alpha": 0.05, "n_signals_tested": 3,
                         "n_signals_rejected": 2, "split": "held_out"},
    }), encoding="utf-8")

    events = [
        {"event": "analysis", "cycle": 1, "severity": "none",
         "conclusion": "Specific over-detection failure mode.",
         "evidence_chain": ["attention near-uniform"],
         "stats_results": [{"summary": "signal vs FAIL: effect=+0.86 -> REJECT H0"}]},
        {"event": "diagnosis", "cycle": 1, "n_hypotheses": 2,
         "hypotheses": [
             {"statement": "language-prior hallucination", "failure_mode": "lp",
              "test_design": "relative_attention.max_relative_weight"},
             {"statement": "diffuse visual attention causes over-detection", "failure_mode": "diffuse_attention",
              "test_design": "prompt_contrast describe_first contrast"},
         ],
         "referenced_charts": ["Fail rate by signal"], "explore_context_used": True,
         "raw_judge_output": (
             "HYPOTHESIS: language-prior hallucination\n"
             "FAILURE_MODE: lp\n"
             "TEST: relative_attention.max_relative_weight\n\n"
             "HYPOTHESIS: diffuse visual attention causes over-detection\n"
             "FAILURE_MODE: diffuse_attention\n"
             "TEST: prompt_contrast describe_first contrast"
         )},
        # surgery.hypothesis matches the diagnosis statement -> the M5 verdict joins.
        {"event": "surgery", "cycle": 1, "module": "m5", "status": "supported",
         "hypothesis": "language-prior hallucination"},
        {"event": "fix", "cycle": 1},
    ]
    (logs / "run_log.jsonl").write_text("\n".join(json.dumps(e) for e in events), encoding="utf-8")
    return root


def _run_app(run_dir):
    from streamlit.testing.v1 import AppTest

    from evalvitals.analysis import dashboard_app

    sys.argv = ["dashboard_app.py", str(run_dir)]
    # Absolute: AppTest resolves a relative script path against the *calling*
    # file (this test module), not the repo root.
    at = AppTest.from_file(dashboard_app.__file__, default_timeout=30)
    at.run()
    return at


def test_loop_dashboard_renders_analysis_panel_without_error(tmp_path):
    _build_loop_run(tmp_path)
    at = _run_app(tmp_path)

    assert not at.exception
    assert [t.label for t in at.tabs] == [
        "1 Problem Setting",
        "2 Analysis",
        "3 Hypotheses & Artifacts",
    ]

    # The analysis tab renders claim-first evidence panels with nearby support.
    assert len(at.dataframe) >= 1
    blob = " ".join(str(m.value) for m in at.markdown)
    assert "Problem Setting" in blob
    assert "Bottom line" in blob
    # This fixture carries a real surgery/fix record (M5 "supported"), so the
    # run is confirmatory, not descriptive-only — evidence gets a verdict, and
    # the M4/M5 stage chips are real pipeline stages, not hidden.
    assert "Candidate signals" not in blob
    assert "Evidence you can use" in blob
    assert any("carry validity verdicts" in str(i.value) for i in at.info)
    assert "How to read this evidence" in blob
    assert "Takeaway" in blob
    assert "Supporting experiment" in blob
    assert "Method:" in blob
    assert "Takeaway:" in blob
    assert "Agent-authored dashboard narrative for this run." in blob
    assert "M1" in blob and "M2" in blob and "M3" in blob
    assert "Mechanism test" in blob and "Repair / surgery test" in blob
    # the supported M5 verdict reaches both the hero band and the hypothesis card.
    assert ">Supported</span>" in blob


def test_standalone_dashboard_hypotheses_tab_falls_back_gracefully_when_absent(tmp_path):
    source = "\n".join(f"line_{i} = {i}" for i in range(80)) + "\nFULL_SOURCE_SENTINEL = True\n"
    (tmp_path / "analysis.py").write_text(source, encoding="utf-8")
    (tmp_path / "stdout.txt").write_text("STDOUT_SENTINEL", encoding="utf-8")
    (tmp_path / "stderr.txt").write_text("STDERR_SENTINEL", encoding="utf-8")
    (tmp_path / "fused_report.json").write_text(json.dumps({
        "summary": "M2-only exploratory analysis.",
        "observations": ["Candidate pattern surfaced for review."],
        "candidate_signals": [{"name": "candidate_signal", "effect": 0.4}],
        "recommended_confirmatory_tests": ["Formulate an explicit hypothesis later."],
    }), encoding="utf-8")

    at = _run_app(tmp_path)

    assert not at.exception
    assert [t.label for t in at.tabs] == [
        "1 Problem Setting",
        "2 Exploratory Analysis",
        "3 Hypotheses",
        "4 Validation results",
        "5 Fix",
    ]
    blob = " ".join(str(m.value) for m in at.markdown)
    info = " ".join(str(i.value) for i in at.info)
    assert "No formal hypotheses were recorded for this report." in info
    assert "M3 step enabled" not in info
    assert "too thin" not in info
    assert "disabled" not in info.lower()
    assert "Review the candidate signals below as possible follow-ups" in blob
    assert "Candidate signals — possible follow-ups, not validated hypotheses" in blob
    assert any(
        e.label == "Candidate signals — possible follow-ups, not validated hypotheses"
        for e in at.expander
    )
    assert "Suggested next steps" in blob
    assert blob.index("Review the candidate signals below") < blob.index("Generated analysis.py preview")
    assert "Hypotheses & Artifacts" not in blob
    assert any(e.label == "Run artifacts and developer details" for e in at.expander)
    assert any(e.label == "stdout" for e in at.expander)
    assert any(e.label == "stderr" for e in at.expander)
    assert any(e.label == "Raw JSON report" for e in at.expander)
    assert any(b.label == "Download analysis.py" for b in at.download_button)
    assert any(c.label == "View full source" and c.value is False for c in at.checkbox)
    assert "Preview only" in " ".join(str(c.value) for c in at.caption)
    assert any("line_0 = 0" in str(c.value) for c in at.code)
    assert not any("FULL_SOURCE_SENTINEL" in str(c.value) for c in at.code)
    assert "STDOUT_SENTINEL" in " ".join(str(t.value) for t in at.text)
    assert "STDERR_SENTINEL" in " ".join(str(t.value) for t in at.text)

    at.checkbox[0].set_value(True)
    at.run()

    assert not at.exception
    assert any("FULL_SOURCE_SENTINEL" in str(c.value) for c in at.code)


def test_standalone_dashboard_uses_reader_friendly_header_and_details_expander(tmp_path):
    full_question = (
        "What predicts hallucination failures (label=fail) in this VLM "
        "object-presence probe? Compare attention-geometry scalars across "
        "three checkpoints and freeze threshold-explicit recipes."
    )
    (tmp_path / "fused_report.json").write_text(json.dumps({
        "ok": True,
        "question": full_question,
        "observations": ["77 of 221 adversarial cases failed."],
        "candidate_signals": [{"name": "attention_entropy_low"}],
        "charts": [{"name": "failrate_by_attention_entropy"}],
        "attempts": 1,
    }), encoding="utf-8")

    at = _run_app(tmp_path)

    assert not at.exception
    assert [t.label for t in at.tabs] == [
        "1 Problem Setting",
        "2 Exploratory Analysis",
        "3 Hypotheses",
        "4 Validation results",
        "5 Fix",
    ]
    header_html = next(str(m.value) for m in at.markdown if 'ev-report-overview' in str(m.value))
    assert "EvalVitals Analysis Report" in header_html
    assert "Hallucination failure patterns" in header_html
    assert full_question not in header_html
    assert str(tmp_path) not in header_html
    assert "fused_report" not in header_html
    assert any("View full research question and run details" in e.label for e in at.expander)
    assert full_question in " ".join(str(m.value) for m in at.markdown)
    assert str(tmp_path) in " ".join(str(c.value) for c in at.code)

    blob = " ".join(str(m.value) for m in at.markdown)
    assert "Candidate patterns" in blob
    assert "Findings" in blob
    assert "How to interpret this analysis" in blob
    assert 'ev-metric-label">Columns</div>' not in blob
    assert 'ev-metric-label">Rows</div>' not in blob
    assert 'ev-metric-value">-</div>' not in blob


def test_standalone_dashboard_overview_summarizes_validation_without_raw_metadata(tmp_path):
    (tmp_path / "fused_report.json").write_text(json.dumps({
        "ok": True,
        "question": "What predicts yield?",
        "data_profile": {"loaded_rows": 30},
        "observations": ["Pressure and temperature vary together."],
        "candidate_signals": [{"name": "high_pressure"}],
        "charts": [{"name": "yield_by_pressure"}],
        "attempts": 1,
    }), encoding="utf-8")
    (tmp_path / "confirm_report.json").write_text(json.dumps({
        "hypothesis_verdicts": [{"name": "high_pressure", "reject": True}],
    }), encoding="utf-8")

    at = _run_app(tmp_path)

    assert not at.exception
    header_html = next(str(m.value) for m in at.markdown if "ev-report-overview" in str(m.value))
    assert "Exploratory analysis with validation results" in header_html
    assert "30 records" in header_html
    assert str(tmp_path) not in header_html

    blob = " ".join(str(m.value) for m in at.markdown)
    assert 'ev-metric-label">Records</div>' in blob
    assert 'ev-metric-label">Validation</div>' in blob
    assert 'ev-metric-value">Yes</div>' in blob
    assert "Bundle:" in " ".join(str(c.value) for c in at.caption)


def test_standalone_dashboard_renders_m3_hypotheses_with_no_verdict(tmp_path):
    (tmp_path / "fused_report.json").write_text(json.dumps({
        "question": "What predicts yield?",
        "observations": ["Higher temperature batches yield more."],
        "hypotheses": [
            {
                "statement": "Higher temperature accelerates the reaction, raising yield.",
                "basis": "Higher temperature batches yield more (70.1% vs 88.4%).",
                "test_design": "Run a controlled temperature-ramp experiment holding pressure fixed.",
            },
            {"statement": "Catalyst B underperforms due to a side reaction.", "basis": "", "test_design": ""},
        ],
    }), encoding="utf-8")

    at = _run_app(tmp_path)

    assert not at.exception
    blob = " ".join(str(m.value) for m in at.markdown)
    assert "Higher temperature accelerates the reaction, raising yield." in blob
    assert "Run a controlled temperature-ramp experiment holding pressure fixed." in blob
    assert "Catalyst B underperforms due to a side reaction." in blob
    assert not any("No formal hypotheses were recorded" in str(i.value) for i in at.info)
    # proposal only — no support/tested verdict pill, since there's no
    # confirm/M4/M5 phase wired up for the standalone tool
    assert 'ev-pill" style="border-color' not in blob
    assert "not yet tested" not in blob.lower()
    assert any("Proposed only, not validated" in str(m.value) for m in at.markdown)


def test_standalone_dashboard_pairs_takeaway_with_its_chart_and_analysis(tmp_path):
    (tmp_path / "tables").mkdir()
    (tmp_path / "tables" / "yield_by_temp.csv").write_text(
        "temp_bin,mean_yield\nlow,70.1\nhigh,88.4\n", encoding="utf-8"
    )
    (tmp_path / "fused_report.json").write_text(json.dumps({
        "question": "What predicts yield?",
        "observations": ["30 batches, no missing values."],
        "takeaways": [{
            "title": "Higher temperature batches yield more (70.1% vs 88.4%).",
            "chart_names": ["yield_by_temp"],
            "table_names": ["yield_by_temp"],
            "analysis": "Mean yield rises from 70.1% in low-temperature batches to 88.4% in high-temperature ones.",
            "caveat": "Descriptive only; temperature and pressure are correlated.",
        }],
        "chart_readings": [{
            "chart": "yield_by_temp",
            "reading": "The high-temperature group has the higher plotted mean yield.",
            "do_not_infer": "This does not establish temperature as the cause.",
        }],
        "charts": [{
            "name": "yield_by_temp", "kind": "line",
            "data": "tables/yield_by_temp.csv", "x": "temp_bin", "y": "mean_yield",
            "title": "Mean yield by temperature bin",
        }],
        "tables": {"yield_by_temp": "tables/yield_by_temp.csv"},
    }), encoding="utf-8")

    at = _run_app(tmp_path)

    assert not at.exception
    blob = " ".join(str(m.value) for m in at.markdown)
    # the takeaway's numbered badge, headline, analysis, and caveat all render together
    assert 'ev-takeaway-badge">1<' in blob
    assert "Higher temperature batches yield more" in blob
    assert "Mean yield rises from 70.1%" in blob
    assert "Caveat" in blob and "Descriptive only" in blob
    assert "Findings overview" in blob
    assert "1 visual · 1 source table · caution noted · detail available" in blob
    assert "Evidence reading" in blob
    assert "How to read this visual" not in blob
    assert "high-temperature group has the higher plotted mean yield" in blob
    assert "Caution" in blob
    assert "This does not establish temperature as the cause." in blob
    assert "Details and source data for finding 1" in [e.label for e in at.expander]
    assert "Source table: Yield by temp" in [e.label for e in at.expander]
    # its supporting chart was found and rendered, not left orphaned
    assert "referenced evidence not found" not in blob
    assert "Mean yield by temperature bin" in blob  # the chart's own title rendered
    # no leftover M2/hypothesis-generation branding in the standalone view
    assert "Standalone M2" not in blob
    assert "M3 hypotheses" not in blob
    assert "hypothesis formation" not in blob


def test_standalone_dashboard_explains_unlinked_exploratory_material(tmp_path):
    (tmp_path / "tables").mkdir()
    (tmp_path / "tables" / "primary.csv").write_text(
        "group,value\na,1\nb,2\n", encoding="utf-8"
    )
    (tmp_path / "tables" / "context.csv").write_text(
        "group,value\na,3\nb,4\n", encoding="utf-8"
    )
    (tmp_path / "fused_report.json").write_text(json.dumps({
        "takeaways": [{
            "title": "Primary finding.", "chart_names": ["primary"],
            "table_names": [], "analysis": "The ranked finding.", "caveat": "",
        }],
        "visual_plan": [
            {"name": "primary", "question": "What is the main pattern?",
             "rationale": "Direct comparison.", "disposition": "primary"},
            {"name": "context", "question": "Does the distribution have an edge case?",
             "rationale": "A group comparison exposes imbalance.", "disposition": "supporting",
             "not_promoted_reason": "It is a diagnostic context check, not a ranked result."},
        ],
        "charts": [
            {"name": "primary", "kind": "bar", "data": "tables/primary.csv", "x": "group", "y": "value"},
            {"name": "context", "kind": "bar", "data": "tables/context.csv", "x": "group", "y": "value"},
        ],
    }), encoding="utf-8")

    at = _run_app(tmp_path)

    assert not at.exception
    labels = [e.label for e in at.expander]
    assert any("Supporting exploratory material (not a conclusion)" in label for label in labels)


def test_standalone_dashboard_renders_unlinked_serialized_plot_path(tmp_path):
    """Agent reports store plot paths as JSON strings, not ``Path`` objects."""
    (tmp_path / "fused_report.json").write_text(json.dumps({
        "takeaways": [{
            "title": "Primary finding.", "chart_names": [], "table_names": [],
            "analysis": "A descriptive finding.", "caveat": "",
        }],
        "plots": ["figures/unlinked_context.png"],
    }), encoding="utf-8")

    at = _run_app(tmp_path)

    assert not at.exception
    assert any("Supporting exploratory material (not a conclusion)" in e.label for e in at.expander)


def test_standalone_dashboard_renders_plain_title_as_headline_with_technical_detail(tmp_path):
    (tmp_path / "fused_report.json").write_text(json.dumps({
        "takeaways": [{
            "plain_title": "Small objects trip up the model far more often.",
            "title": "Small objects fail far more often (18% vs 4%, AUC 0.71).",
            "chart_names": [], "table_names": [], "analysis": "x", "caveat": "",
        }],
    }), encoding="utf-8")

    at = _run_app(tmp_path)

    assert not at.exception
    blob = " ".join(str(m.value) for m in at.markdown)
    assert 'ev-takeaway-title">Small objects trip up the model far more often.' in blob
    assert "Technical title" in blob
    assert "Small objects fail far more often (18% vs 4%, AUC 0.71)." in blob
    assert "Details and source data for finding 1" in [e.label for e in at.expander]


def test_standalone_dashboard_keeps_a_takeaway_with_only_plain_title(tmp_path):
    """A takeaway with plain_title but no technical 'title' must not be
    silently dropped — plain_title alone is a real finding."""
    (tmp_path / "fused_report.json").write_text(json.dumps({
        "takeaways": [{
            "plain_title": "Cases with the flag set almost always fail.",
            "chart_names": [], "table_names": [], "analysis": "x", "caveat": "",
        }],
    }), encoding="utf-8")

    at = _run_app(tmp_path)

    assert not at.exception
    blob = " ".join(str(m.value) for m in at.markdown)
    assert "Cases with the flag set almost always fail." in blob
    assert "no structured takeaways" not in blob.lower()


def test_standalone_dashboard_renders_plain_question_as_page_headline(tmp_path):
    """The page's <h1> is the report's biggest, most prominent text — it must
    not be the raw research question verbatim when a plain restatement
    exists (the technical one still shown, demoted)."""
    (tmp_path / "fused_report.json").write_text(json.dumps({
        "question": (
            "What predicts hallucination failures (label=fail)? Compare "
            "attention_entropy/focus_share DISTRIBUTIONS (violin/ECDF) between "
            "FAIL and PASS; signals are collinear."
        ),
        "plain_question": "Why does the model imagine objects that aren't in the picture?",
        "ok": True,
    }), encoding="utf-8")

    at = _run_app(tmp_path)

    assert not at.exception
    html = "".join(str(m.value) for m in at.markdown if isinstance(m.value, str))
    assert "<h1>Why does the model imagine objects that aren&#x27;t in the picture?</h1>" in html
    assert "Original question:" in html
    assert "collinear" in html  # the technical wording is demoted, not deleted


def test_standalone_dashboard_falls_back_to_raw_question_without_plain_question(tmp_path):
    """Older reports never had plain_question — must still render the raw
    question as the headline, with no dangling 'Original question:' line."""
    (tmp_path / "fused_report.json").write_text(json.dumps({
        "question": "What predicts yield?",
        "ok": True,
    }), encoding="utf-8")

    at = _run_app(tmp_path)

    assert not at.exception
    html = "".join(str(m.value) for m in at.markdown if isinstance(m.value, str))
    assert "<h1>What predicts yield?</h1>" in html
    assert "Original question:" not in html


def test_standalone_dashboard_falls_back_gracefully_with_no_takeaways(tmp_path):
    (tmp_path / "fused_report.json").write_text(json.dumps({
        "observations": ["No structured takeaways in this older report."],
        "charts": [],
    }), encoding="utf-8")

    at = _run_app(tmp_path)

    assert not at.exception
    blob = " ".join(str(i.value) for i in at.info)
    assert "no structured takeaways" in blob.lower()


def test_standalone_dashboard_shows_data_structure_panel(tmp_path):
    (tmp_path / "fused_report.json").write_text(json.dumps({
        "data_profile": {
            "n_rows": 30,
            "loaded_rows": 30,
            "grain": "case",
            "outcome": {"present": True, "column": "yield_pct", "kind": "continuous", "unique": 29},
            "columns": {
                "yield_pct": {
                    "role": "outcome", "dtype": "numeric", "unique": 29, "missing": 0,
                    "numeric_min": 63.8, "numeric_max": 95.9,
                },
                "temperature": {
                    "role": "predictor", "dtype": "numeric", "unique": 28, "missing": 0,
                    "numeric_min": 151.4, "numeric_max": 248.7,
                },
                "catalyst": {"role": "predictor", "dtype": "categorical", "unique": 3, "missing": 0},
            },
            "warnings": ["column 'batch_id' is constant and unlikely to be testable"],
            "folder_scan": {
                "root": "/data/runs/chestagentbench",
                "is_file": False,
                "n_files_total": 4,
                "n_dirs": 2,
                "extensions": {".json": 3, ".txt": 1},
                "json_files_found": 4,
                "json_files_used": 3,
                "entries": ["agent-a/", "agent-a/results.json", "notes.txt"],
                "truncated": False,
            },
        },
        "charts": [],
    }), encoding="utf-8")

    at = _run_app(tmp_path)

    assert not at.exception
    blob = " ".join(str(m.value) for m in at.markdown)
    # schema is stated up front: row/column counts, grain, and outcome kind
    assert "Data structure" in blob
    assert "case-level" in blob
    assert "Continuous outcome: Yield pct (29 distinct values)" in blob
    # what was literally found on disk is shown too, before the parsed schema
    assert "Folder contents" in blob
    assert "/data/runs/chestagentbench" in blob
    assert "4 files across 2 subdirectories" in blob
    assert "3 of 4 JSON file(s) sampled" in blob
    codes = " ".join(str(c.value) for c in at.code)
    assert "agent-a/results.json" in codes
    # the full column/role/type breakdown is a real table, not a truncated dict repr
    assert len(at.dataframe) >= 1
    schema_df = at.dataframe[0].value
    assert "temperature" in schema_df["Field"].str.lower().tolist()
    assert "outcome" in schema_df["Role"].tolist()
    # profiling notes (e.g. constant columns) surface too
    captions = " ".join(str(c.value) for c in at.caption)
    assert "constant" in captions.lower()


def test_standalone_dashboard_lets_you_browse_raw_records(tmp_path):
    (tmp_path / "fused_report.json").write_text(json.dumps({
        "data_profile": {"n_rows": 3, "columns": {}},
        "charts": [],
    }), encoding="utf-8")
    (tmp_path / "records.json").write_text(json.dumps([
        {"case_id": "c0", "region": "north", "yield_pct": 70.1},
        {"case_id": "c1", "region": "south", "yield_pct": 88.4},
        {"case_id": "c2", "region": "north", "yield_pct": 91.2},
    ]), encoding="utf-8")

    at = _run_app(tmp_path)

    assert not at.exception
    assert any(e.label == "Browse raw data (3 row(s) loaded)" for e in at.expander)
    assert len(at.dataframe) >= 1
    raw_df = at.dataframe[0].value
    assert set(raw_df["case_id"]) == {"c0", "c1", "c2"}

    at.text_input(key="raw_data_search").set_value("south").run()
    raw_df = at.dataframe[0].value
    assert list(raw_df["case_id"]) == ["c1"]


def test_standalone_dashboard_falls_back_to_sample_rows_without_records_json(tmp_path):
    (tmp_path / "fused_report.json").write_text(json.dumps({
        "data_profile": {
            "n_rows": 30,
            "columns": {},
            "sample_rows": [{"case_id": "c0", "yield_pct": 70.1}],
        },
        "charts": [],
    }), encoding="utf-8")

    at = _run_app(tmp_path)

    assert not at.exception
    assert any(e.label == "Browse raw data (1 row(s) loaded)" for e in at.expander)
    captions = " ".join(str(c.value) for c in at.caption)
    assert "only a small sample was saved" in captions.lower()


def test_loop_dashboard_informs_when_no_explore_report(tmp_path):
    logs = tmp_path / "logs_m2_5"
    logs.mkdir(parents=True)
    (logs / "run_log.jsonl").write_text(
        json.dumps({"event": "diagnosis", "cycle": 1, "n_hypotheses": 0, "hypotheses": []}) + "\n",
        encoding="utf-8",
    )
    at = _run_app(tmp_path)
    assert not at.exception
    # Missing explore artifacts is a normal, expected state (e.g. a
    # hypotheses-only rerun) — an info note, not an alarming yellow warning.
    assert any("explore report" in str(i.value).lower() for i in at.info)
    assert not any("explore report" in str(w.value).lower() for w in at.warning)


def _build_agentic_run(root):
    """An AgenticDiagnoseLoop run: run_start/loop_end lifecycle, a rejected
    early-stop dispatch (host discipline, not an error), then a real M1-M3-M5
    happy path that resolves supported."""
    logs = root / "logs_agentic"
    logs.mkdir(parents=True)
    events = [
        {"event": "run_start", "model": "FakeModel(...)", "decision_judge": "ClaudeModel(...)",
         "max_actions": 10, "n_cases": 4,
         "protocol": {"description": "Do FAIL cases share an attention signature?"},
         "label_distribution": {"FAIL": 2, "PASS": 2}},
        {"event": "probe", "cycle": 0, "analyzers": ["attention"], "findings": {}, "artifact_paths": {}},
        {"event": "agent_decision", "step": 0, "action": "run_probe", "params": {},
         "rationale": "start with M1", "valid": True},
        {"event": "agent_tool", "step": 0, "tool": "run_probe", "ok": True, "summary": "ran 1 analyzer(s)"},
        {"event": "agent_decision", "step": 1, "action": "stop",
         "params": {"resolved": True, "reason": "too early"}, "rationale": "check early exit", "valid": True},
        {"event": "agent_tool", "step": 1, "tool": "stop", "ok": False,
         "error": "no_supported_hypothesis", "summary": "cannot declare success yet"},
        {"event": "analysis", "cycle": 0, "conclusion": "signal found"},
        {"event": "agent_decision", "step": 2, "action": "propose_hypotheses", "params": {},
         "rationale": "generate hypotheses", "valid": True},
        {"event": "agent_tool", "step": 2, "tool": "propose_hypotheses", "ok": True, "summary": "1 hypothesis"},
        {"event": "diagnosis", "cycle": 0, "n_hypotheses": 1,
         "hypotheses": [{"statement": "attention causes failure", "failure_mode": "attention"}]},
        {"event": "agent_decision", "step": 3, "action": "test_hypothesis", "params": {},
         "rationale": "test it", "valid": True},
        {"event": "agent_tool", "step": 3, "tool": "test_hypothesis", "ok": True, "summary": "tested"},
        {"event": "surgery", "cycle": 0, "module": "m5", "status": "supported",
         "hypothesis": "attention causes failure"},
        {"event": "agent_decision", "step": 4, "action": "stop",
         "params": {"resolved": True, "reason": "supported"}, "rationale": "done", "valid": True},
        {"event": "agent_tool", "step": 4, "tool": "stop", "ok": True, "summary": "resolved"},
        {"event": "loop_end", "cycles": 1, "stopped_by": "agent_stop", "n_verified": 1,
         "total_duration_sec": 42.0},
    ]
    (logs / "run_log.jsonl").write_text("\n".join(json.dumps(e) for e in events), encoding="utf-8")
    return root


def test_agentic_run_dashboard_shows_hero_trajectory_and_verdicts(tmp_path):
    _build_agentic_run(tmp_path)
    at = _run_app(tmp_path)

    assert not at.exception
    assert [t.label for t in at.tabs] == [
        "1 Problem Setting", "2 Agent Trajectory", "3 Analysis", "4 Hypotheses & Artifacts",
    ]
    blob = " ".join(str(m.value) for m in at.markdown)
    assert "Agentic Diagnosis Run" in blob
    # what this run is investigating (run_start's protocol) is stated up front,
    # not just the bare verdict — a run_start-only field, never previously
    # surfaced anywhere in the dashboard.
    assert "Investigating: Do FAIL cases share an attention signature?" in blob
    assert "2 FAIL / 2 PASS" in blob
    assert "Resolved" in blob and "Supported" in blob
    assert ">Supported</span>" in blob
    assert "attention causes failure" in blob
    # the rejected early-stop dispatch is a visible discipline feature, not an error.
    assert "Rejected" in blob and "no_supported_hypothesis" in blob
    # accepted steps show their tool name and rationale.
    assert "run_probe" in blob and "test_hypothesis" in blob
    assert "start with M1" in blob
    # no raw run-directory path inside the hero band itself (demoted to a caption).
    assert "<h1>" not in blob
    captions = " ".join(str(c.value) for c in at.caption)
    assert str(tmp_path) in captions


def test_analysis_tab_tells_connected_story(tmp_path):
    _build_loop_run(tmp_path)
    at = _run_app(tmp_path)
    assert not at.exception
    heads = [m.value for m in at.markdown if isinstance(m.value, str) and m.value.startswith("###")]
    blob = " ".join(str(m.value) for m in at.markdown)
    # the three-panel narrative sections are present
    assert "ev-section-title\">Problem Setting<" in blob
    assert any("Analysis" in h for h in heads)
    assert any("Proposed Hypotheses" in h for h in heads)
    assert "Measurement" in blob
    assert "Exploratory analysis" in blob
    assert "Hypothesis generation" in blob
    # the hypothesis statement appears, and this fixture's matching M5 surgery
    # record ("supported") now surfaces as a real verdict — M4/M5 are live
    # pipeline stages once a run actually reaches them.
    assert "language-prior hallucination" in blob
    assert "M5" in blob and "M4" in blob
    assert ">Supported</span>" in blob
    assert any(e.label == "Why these charts were chosen" for e in at.expander)
    assert len(at.dataframe) >= 2  # signal table + visual-plan table


def test_hypotheses_tab_shows_multiple_hypotheses_and_how_each_was_derived(tmp_path):
    """M3 can propose 1-3 hypotheses per cycle (see diagnosis.py's _DIAGNOSE_PROMPT),
    and the LLM's own TEST: line for each is real evidence of how it was derived —
    both must actually reach the dashboard, not just the first hypothesis with no
    justification."""
    _build_loop_run(tmp_path)
    at = _run_app(tmp_path)

    assert not at.exception
    blob = " ".join(str(m.value) for m in at.markdown)
    # both hypotheses from this cycle are shown, not just the first
    assert "language-prior hallucination" in blob
    assert "diffuse visual attention causes over-detection" in blob
    # each hypothesis's own "how to check" line (from its test_design) is shown
    assert "relative_attention.max_relative_weight" in blob
    assert "prompt_contrast describe_first contrast" in blob
    # the full M3 reasoning (raw judge output) is available, not just the
    # parsed statement/failure_mode with no justification
    assert any("Full M3 reasoning" in e.label for e in at.expander)
    texts = " ".join(str(t.value) for t in at.text)
    assert "HYPOTHESIS: language-prior hallucination" in texts


def test_stale_case_matrix_never_silently_replaces_this_runs_own_charts(tmp_path):
    """A case matrix reconstructed from m1_state.pkl is a separate artifact from a
    specific attention-probe pipeline — if an output dir is reused across runs, a
    stale pickle must not silently stand in for (or hide) this run's own explorer
    charts, and it must be clearly labeled as a distinct, possibly-mismatched source."""
    pytest.importorskip("plotly", reason="case-matrix charts require Plotly dashboard extras")

    _build_loop_run(tmp_path)
    # This run's own group-stats/fail-rate tables (distinct signal name).
    tables = tmp_path / "fused" / "sandbox" / "tables"
    (tables / "groupstats_myown_signal.csv").write_text(
        "outcome,mean\nFAIL,0.8\nPASS,0.3\n", encoding="utf-8"
    )
    (tables / "failrate_by_myown_signal.csv").write_text(
        "bin,fail_rate\n[0-1),0.2\n[1-2),0.5\n", encoding="utf-8"
    )

    # A stale/unrelated case matrix (different signal, from a different pipeline).
    # Write the real pickle-shaped artifact instead of monkeypatching: AppTest
    # executes dashboard_app.py as a script, so module-level monkeypatches do not
    # reliably cross that file-run boundary.
    cases = [
        SimpleNamespace(id=f"c{i}", label=label, observed=observed, expected=expected)
        for i, (label, observed, expected) in enumerate([
            ("FAIL", "yes", "yes"),
            ("PASS", "no", "no"),
            ("FAIL", "yes", "no"),
            ("PASS", "no", "yes"),
            ("FAIL", "yes", "yes"),
            ("PASS", "no", "no"),
        ])
    ]
    relative_attention = SimpleNamespace(findings={"per_case": [
        {"id": f"c{i}", "max_relative_weight": v}
        for i, v in enumerate([0.9, 0.1, 0.8, 0.2, 0.7, 0.3])
    ]})
    with (tmp_path / "m1_state.pkl").open("wb") as fh:
        pickle.dump({"cases": cases, "probe_results": {"relative_attention": relative_attention}}, fh)

    at = _run_app(tmp_path)

    assert not at.exception
    expander_labels = [e.label for e in at.expander]
    # the stale matrix is opt-in and clearly labeled as a separate artifact —
    # never inline/unlabeled where it could be mistaken for this run's analysis
    assert any("case-matrix charts" in lbl and "m1_state.pkl" in lbl for lbl in expander_labels)
    assert any("Attention-probe statistical analysis" in lbl for lbl in expander_labels)


def test_hypotheses_join_with_m5_outcomes():
    from evalvitals.analysis.dashboard_app import _hypotheses_with_outcomes

    story = {
        "diagnoses": [{"cycle": 1, "referenced_charts": ["C"],
                       "hypotheses": [{"statement": "H one", "failure_mode": "fm"},
                                      {"statement": "H two", "failure_mode": "fm2"}]}],
        "surgeries": [{"module": "m5", "status": "supported", "hypothesis": "H one"},
                      {"module": "m4", "status": "fixed", "fixed": True, "hypothesis": "H one"}],
    }
    out = _hypotheses_with_outcomes(story)
    assert [h["statement"] for h in out] == ["H one", "H two"]
    assert len(out[0]["tests"]) == 2          # joined both tests by statement
    assert out[1]["tests"] == []              # H two was never tested


def _build_analysis_phase_run(root):
    """Same fused report as the confirmatory fixture, but the loop log is an
    analysis-phase pass (descriptive_only, no surgery) under logs_analysis/."""
    _build_loop_run(root)
    # Replace the confirmatory M2_5 log with an analysis-phase logs_analysis/ log.
    import shutil
    shutil.rmtree(root / "logs_m2_5")
    logs = root / "logs_analysis"
    logs.mkdir(parents=True)
    events = [
        {"event": "analysis", "cycle": 0, "descriptive_only": True,
         "conclusion": "Candidate signals described; verdict deferred."},
        {"event": "diagnosis", "cycle": 0, "n_hypotheses": 1,
         "hypotheses": [{"statement": "language-prior hallucination", "failure_mode": "lp"}],
         "referenced_charts": ["Fail rate by signal"]},
        {"event": "loop_end", "stopped_by": "analysis_complete"},
    ]
    (logs / "run_log.jsonl").write_text("\n".join(json.dumps(e) for e in events), encoding="utf-8")
    return root


def test_analysis_phase_dashboard_shows_no_supported_verdicts(tmp_path):
    # WS2 / decoupling: an analysis-phase run must present candidate signals
    # DESCRIPTIVELY — no "Supported" / "Not supported" verdict pills, since the
    # explorer's reject flags are not confirmation.
    _build_analysis_phase_run(tmp_path)
    at = _run_app(tmp_path)
    assert not at.exception
    assert [t.label for t in at.tabs] == [
        "1 Problem Setting",
        "2 Analysis",
        "3 Proposed Hypotheses",
    ]
    blob = " ".join(str(m.value) for m in at.markdown)
    assert "language-prior hallucination" in blob
    assert "Proposed Hypotheses" in blob
    # the descriptive evidence pills carry "Descriptive", never a verdict
    assert "Descriptive</span>" in blob
    assert ">Supported</span>" not in blob
    assert ">Not supported</span>" not in blob
    # the confirmatory method card ("tested signals survived") is replaced
    assert "tested signals survived" not in blob
    # an explicit analysis-phase banner orients the reader
    assert any("Analysis phase" in str(i.value) for i in at.info)


def test_analysis_phase_dashboard_recovers_proposed_hypotheses_file(tmp_path):
    _build_analysis_phase_run(tmp_path)
    logs = tmp_path / "logs_analysis"
    (logs / "run_log.jsonl").write_text(
        "\n".join([
            json.dumps({"event": "analysis", "cycle": 0, "descriptive_only": True,
                        "conclusion": "Candidate signals described; verdict deferred."}),
            json.dumps({"event": "loop_end", "stopped_by": "analysis_complete"}),
        ]) + "\n",
        encoding="utf-8",
    )
    analysis = tmp_path / "analysis"
    analysis.mkdir(exist_ok=True)
    (analysis / "proposed_hypotheses.json").write_text(json.dumps([
        {
            "hypothesis": "fallback hypothesis from proposed_hypotheses.json",
            "predicted_failure_mode": "fallback_mode",
        }
    ]), encoding="utf-8")

    at = _run_app(tmp_path)

    assert not at.exception
    assert [t.label for t in at.tabs] == [
        "1 Problem Setting",
        "2 Analysis",
        "3 Proposed Hypotheses",
    ]
    blob = " ".join(str(m.value) for m in at.markdown)
    assert "fallback hypothesis from proposed_hypotheses.json" in blob
    assert "fallback_mode" in blob


def _build_explore_run_with_pipeline(root, *, with_confirm=True, with_fix=True):
    """A standalone explore output plus the held-out pipeline artifacts
    (confirm_report.json / fix_report.json) next to the exploratory report."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "exploratory_report.json").write_text(json.dumps({
        "ok": True,
        "question": "what drives FAIL?",
        "observations": ["all failures adversarial"],
        "takeaways": [{"title": "Peaked attention rides with FAIL (d=1.3).",
                       "chart_names": [], "table_names": [],
                       "analysis": "focus share separates.", "caveat": ""}],
        "hypotheses": [
            {"statement": "Peaked attention marks hallucinations.",
             "basis": "d=1.3 on explore half", "test_design": "re-test recipe on holdout"},
            {"statement": "Peakedness is a downstream effect, not a cause.",
             "basis": "cannot tell from observational data", "test_design": "surgery"},
        ],
        "candidate_signals": [{"name": "risky_top_signal",
                               "recipe": {"name": "r", "kind": "expr", "expr": "focus_share >= 0.29"}}],
        "charts": [], "plots": [], "tables": {},
    }), encoding="utf-8")
    if with_confirm:
        (root / "confirm_report.json").write_text(json.dumps({
            "phase": "holdout_confirm", "split": "held_out",
            "n_validate_rows": 241, "n_validate_fail": 49, "alpha": 0.05,
            "adjudication": {"method": "e-BH", "alpha": 0.05, "split": "held_out",
                             "n_host_adjudicated": 1, "n_rejected": 1},
            "signal_verdicts": [{"name": "risky_top_signal", "status": "adjudicated",
                                 "reject": True, "effect": 0.45, "ci": [0.3, 0.6],
                                 "fail_rate_flagged": 0.62, "fail_rate_unflagged": 0.17,
                                 "n_holdout": 146}],
            "hypothesis_verdicts": [
                {"statement": "Peaked attention marks hallucinations.",
                 "verdict": "supported", "reasoning": "held-out fail rate 62% vs 17%.",
                 "needs_surgery": False},
                {"statement": "Peakedness is a downstream effect, not a cause.",
                 "verdict": "not_testable", "reasoning": "needs an intervention.",
                 "needs_surgery": True},
            ],
            "judge": {"model": "claude-opus-4-8", "effort": "low"},
        }), encoding="utf-8")
    if with_fix:
        (root / "fix_report.json").write_text(json.dumps({
            "phase": "surgery_fix", "model": "qwen3-vl-2b-instruct", "n_cases": 201,
            "hypotheses_in": ["Peaked attention marks hallucinations."],
            "m5_results": [{"statement": "Peaked attention marks hallucinations.",
                            "status": "supported", "confidence": 0.8,
                            "evidence_grade": "B", "holdout_verdict": "supported"}],
            "m4": "surgery summary text",
            "fix": {
                "max_tier": "L3b", "fixed": True, "best": "scan_then_decide",
                "ebh_survivors": ["scan_then_decide", "upscale_zoom"],
                "repair_rounds": 1, "recommendation": None,
                "refine_signal": {"kind": "heterogeneous_failure_mode",
                                  "candidate": "contrast_vote", "n_helped": 16,
                                  "n_hurt": 5,
                                  "message": "'contrast_vote' repaired 16 but broke 5 — gate the fix."},
                "routed": [],
                "attempted": [
                    {"tier": "L1", "name": "scan_then_decide", "kind": "template",
                     "n_pairs": 201, "n_fixed": 12, "n_broken": 0, "coverage": 1.0,
                     "e_value": 315.1, "effect": 0.06, "reject": True, "verdict": "fixed",
                     "summary": "e=315 -> REJECT H0 [fixed]"},
                    {"tier": "L2", "name": "upscale_zoom", "kind": "spec",
                     "n_pairs": 194, "n_fixed": 11, "n_broken": 0, "coverage": 0.98,
                     "e_value": 170.7, "effect": 0.057, "reject": True, "verdict": "fixed",
                     "summary": "e=171 -> REJECT H0 [fixed]"},
                    {"tier": "L3b", "name": "embedding_boost", "kind": "primitive",
                     "n_pairs": 201, "n_fixed": 5, "n_broken": 0, "coverage": 1.0,
                     "e_value": 5.3, "effect": 0.025, "reject": False, "verdict": "partial",
                     "summary": "e=5.3 -> inconclusive"},
                ],
            },
            "logs": "outputs_pipeline/3_surgery/logs",
        }), encoding="utf-8")
    return root


def test_explore_dashboard_renders_holdout_verdicts_and_fix(tmp_path):
    _build_explore_run_with_pipeline(tmp_path)
    at = _run_app(tmp_path)
    assert not at.exception
    assert [t.label for t in at.tabs] == [
        "1 Problem Setting", "2 Exploratory Analysis", "3 Hypotheses",
        "4 Validation results", "5 Fix",
    ]
    blob = " ".join(str(m.value) for m in at.markdown)
    # tab 3 stays the pure proposal view; verdict badges live in tab 4 only
    assert "Proposed only, not validated here" in blob
    assert "held-out: supported" in blob
    assert "held-out: not_testable" in blob
    assert "routed to surgery" in blob
    assert any("Signal recipes on the held-out split" in str(m.value) for m in at.markdown)
    # fix tab: reader-facing repair digest + full candidate table
    assert "Winner — L1 `scan_then_decide`" in blob
    assert "Survived e-BH across the whole candidate family" in blob
    assert "beat internals-write interventions" in blob
    assert "Refine signal:" in blob and "gate the fix" in blob
    assert "All repair candidates" in blob


def test_explore_dashboard_without_pipeline_artifacts_greys_out_tabs(tmp_path):
    """The layout is FIXED at five tabs: an M3-only run shows the held-out
    verdict and fix tabs as greyed "not available" placeholders instead of
    dropping them, so every explore-shaped result has the same shape."""
    _build_explore_run_with_pipeline(tmp_path, with_confirm=False, with_fix=False)
    at = _run_app(tmp_path)
    assert not at.exception
    assert [t.label for t in at.tabs] == [
        "1 Problem Setting", "2 Exploratory Analysis", "3 Hypotheses",
        "4 Validation results", "5 Fix",
    ]
    blob = " ".join(str(m.value) for m in at.markdown)
    assert blob.count("not available for this run") == 2
    assert "stopped at M3" in blob                    # verdict placeholder
    assert "No repair phase was run" in blob          # fix placeholder
    assert "held-out: supported" not in blob  # no badges without a confirm phase
    assert "All repair candidates" not in blob


def test_explore_dashboard_confirm_without_fix_greys_only_fix(tmp_path):
    """SKIP_FIX pipeline shape: real verdicts in tab 4, placeholder in tab 5."""
    _build_explore_run_with_pipeline(tmp_path, with_confirm=True, with_fix=False)
    at = _run_app(tmp_path)
    assert not at.exception
    blob = " ".join(str(m.value) for m in at.markdown)
    assert "held-out: supported" in blob
    assert blob.count("not available for this run") == 1
    assert "No repair phase was run" in blob


def test_fix_narrative_digest():
    from evalvitals.analysis.dashboard_app import _fix_narrative

    fix = {
        "best": "scan_then_decide",
        "ebh_survivors": ["scan_then_decide"],
        "refine_signal": {"message": "'x' repaired 16 but broke 5."},
        "attempted": [
            {"tier": "L1", "name": "scan_then_decide", "n_fixed": 12, "n_broken": 0,
             "e_value": 315.1, "reject": True, "verdict": "fixed"},
            {"tier": "L3b", "name": "boost", "n_fixed": 5, "n_broken": 0,
             "e_value": 5.3, "reject": False, "verdict": "partial"},
        ],
    }
    lines = " ".join(_fix_narrative(fix))
    assert "Winner — L1 `scan_then_decide`" in lines
    assert "repaired 12" in lines and "e-value 315.1" in lines
    assert "Survived e-BH" in lines
    assert "beat internals-write interventions" in lines and "e=5.3" in lines
    assert "Refine signal" in lines
    assert _fix_narrative({}) == []  # old reports without `attempted` degrade quietly


def test_resolve_scatter_axes_handles_real_named_and_legacy_columns():
    import pandas as pd

    from evalvitals.analysis.dashboard_app import _resolve_scatter_axes

    # (a) newer schema: real signal names as columns, no report hint available.
    d = pd.DataFrame({"attention_entropy": [0.8, 0.6], "center_offset": [0.2, 0.4],
                      "outcome": ["PASS", "FAIL"]})
    _d, xs, ys, oc = _resolve_scatter_axes(d, {}, "scatter_top_attention_pair.csv")
    assert (xs, ys, oc) == ("attention_entropy", "center_offset", "outcome")

    # (b) legacy schema: literal x/y renamed from the report's chart title.
    d2 = pd.DataFrame({"x": [1.0, 2.0], "y": [3.0, 4.0], "outcome": ["FAIL", "PASS"]})
    report = {"charts": [{"name": "scatter_pair",
                          "title": "focus_share vs edge_mass by outcome"}]}
    d2r, xs, ys, oc = _resolve_scatter_axes(d2, report, "scatter_pair.csv")
    assert (xs, ys, oc) == ("focus_share", "edge_mass", "outcome")
    assert {"focus_share", "edge_mass"} <= set(d2r.columns)  # columns were renamed

    # (c) unresolvable (no two value axes) → all-None sentinel, no crash.
    bad = pd.DataFrame({"outcome": ["FAIL", "PASS"]})
    _b, xs, ys, oc = _resolve_scatter_axes(bad, {}, "scatter_bad.csv")
    assert (xs, ys, oc) == (None, None, None)


def test_signals_dataframe_and_effect_figure_helpers():
    from evalvitals.analysis.dashboard_app import _signal_effect_figure, _signals_dataframe

    signals = [
        {"name": "a", "source": "explorer", "effect": 0.8, "reject": True},
        {"name": "b", "source": "catalog", "effect": 0.3, "reject": False},
        {"name": "c", "source": "explorer", "effect": None, "reject": None},
    ]
    df = _signals_dataframe(signals)
    assert list(df["verdict"]) == ["REJECT H0", "inconclusive", "descriptive"]

    fig = _signal_effect_figure(signals)  # only the 2 numeric ones plotted
    assert fig is not None


def test_verdict_sentence_states_the_outcome_in_words():
    """The hero must say what happened before showing any evidence.

    Confirming nothing is a legitimate result for this tool, so it gets a
    plain statement rather than an error tone -- and an in-sample confirmation
    must never be worded as if it survived a held-out test.
    """
    from evalvitals.analysis.dashboard_app import _verdict_sentence

    nothing, tone = _verdict_sentence(
        {"adjudication": {"n_candidates": 4, "n_rejected": 0, "split": "in_sample"}}
    )
    assert nothing == "Nothing confirmed — 0 of 4 candidate signals cleared adjudication."
    assert tone == "ev-verdict-null"

    in_sample, tone = _verdict_sentence(
        {"adjudication": {"n_candidates": 4, "n_rejected": 1, "split": "in_sample"}}
    )
    assert in_sample == "1 of 4 candidate signals confirmed, in-sample only."
    assert tone == "ev-verdict-partial"  # not the "passed" tone

    held_out, tone = _verdict_sentence(
        {"adjudication": {"n_candidates": 3, "n_rejected": 2, "split": "holdout"}}
    )
    assert held_out == "2 of 3 candidate signals confirmed, on held-out rows."
    assert tone == "ev-verdict-ok"

    # Singular, and the no-adjudication case must not read as "0 confirmed".
    one, _ = _verdict_sentence({"adjudication": {"n_candidates": 1, "n_rejected": 1}})
    assert "1 of 1 candidate signal confirmed" in one
    none, tone = _verdict_sentence({})
    assert none == "No candidate signals were adjudicated in this run."
    assert tone == "ev-verdict-neutral"
