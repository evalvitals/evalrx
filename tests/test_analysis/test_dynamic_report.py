from __future__ import annotations

import json
import os


def test_publish_report_contract_and_case_records(tmp_path):
    from evalrx.core import CaseBatch, FailureCase, Label
    from evalrx.eval_agent.run_logger_v2 import RunLoggerV2
    from evalrx.reporting.dynamic import load_published_report, publish_report

    logger = RunLoggerV2(tmp_path)
    logger.log_run_start({
        "model": "demo-model", "benchmark_name": "demo-set", "n_cases": 1,
        "label_distribution": {"fail": 1},
        "protocol": {"description": "Answer the question correctly."},
    })
    case = FailureCase.from_prompt(
        "What is visible?", id="sample-1", expected="cat", observed="dog", label=Label.FAIL,
    )
    logger.log_cases(CaseBatch([case]))
    logger.close()

    result = publish_report(tmp_path)
    data, envelope = load_published_report(tmp_path)
    assert result.generated_by == "deterministic"
    assert data["cases"][0]["id"] == "sample-1"
    assert data["cases"][0]["observed"] == "dog"
    assert envelope["format"] == "json-render"
    # bumped to @2 when CaseStudySheet joined the catalog: a cached layout
    # composed against the older catalog cannot name the new component.
    assert envelope["catalog_version"] == "evalrx-report@2"
    assert envelope["spec"]["elements"]["journey"]["type"] == "Journey"


def test_report_agent_repairs_once_then_validates():
    from evalrx.reporting.dynamic import ReportAgent

    valid = {
        "root": "page",
        "elements": {
            "page": {"type": "ReportPage", "props": {}, "children": ["hero", "journey", "outcome"]},
            "hero": {"type": "SettingHero", "props": {}},
            "journey": {"type": "Journey", "props": {}},
            "outcome": {"type": "OutcomeCard", "props": {}},
        },
    }

    class Model:
        def __init__(self):
            self.calls = 0

        def generate(self, _prompt):
            self.calls += 1
            return "not json" if self.calls == 1 else json.dumps(valid)

    model = Model()
    spec = ReportAgent(model).compose({"findings": [], "charts": [], "repairs": [], "cases": []})
    assert model.calls == 2
    assert spec == valid


def test_invalid_agent_output_falls_back(tmp_path):
    from evalrx.eval_agent.run_logger_v2 import RunLoggerV2
    from evalrx.reporting.dynamic import load_published_report, publish_report

    logger = RunLoggerV2(tmp_path)
    logger.log_run_start({"model": "demo", "n_cases": 0})
    logger.close()

    class BadModel:
        def generate(self, _prompt):
            return '{"root":"x","elements":{"x":{"type":"ArbitraryHTML","props":{}}}}'

    result = publish_report(tmp_path, model=BadModel())
    _, envelope = load_published_report(tmp_path)
    assert result.generated_by == "fallback"
    assert envelope["generated_by"]["mode"] == "fallback"
    assert envelope["spec"]["elements"]["setting"]["type"] == "SettingHero"


def test_dynamic_api_filters_and_serves_spa(tmp_path):
    from fastapi.testclient import TestClient

    from evalrx.core import CaseBatch, FailureCase, Label
    from evalrx.eval_agent.run_logger_v2 import RunLoggerV2
    from evalrx.reporting.dynamic import publish_report
    from evalrx.reporting.server import create_app

    logger = RunLoggerV2(tmp_path)
    logger.log_run_start({"model": "demo", "n_cases": 2})
    logger.log_cases(CaseBatch([
        FailureCase.from_prompt("alpha", id="a", label=Label.FAIL),
        FailureCase.from_prompt("beta", id="b", label=Label.PASS),
    ]))
    logger.close()
    publish_report(tmp_path)

    with TestClient(create_app(tmp_path)) as client:
        assert client.get("/api/report").status_code == 200
        response = client.get("/api/cases", params={"status": "fail", "limit": 1})
        assert response.json()["total"] == 1
        assert response.json()["items"][0]["id"] == "a"
        assert client.get("/").headers["content-type"].startswith("text/html")


def test_runs_panel_lists_and_opens_sibling_experiments(tmp_path):
    from fastapi.testclient import TestClient

    from evalrx.core import CaseBatch, FailureCase, Label
    from evalrx.eval_agent.run_logger_v2 import RunLoggerV2
    from evalrx.reporting.dynamic import publish_report
    from evalrx.reporting.server import create_app

    # Two sibling experiments under one root, as an example's outputs/ tree
    # holds them; only one is published, to exercise the unpublished fallback.
    run_a = tmp_path / "exp_a" / "outputs" / "logs"
    run_b = tmp_path / "exp_b" / "outputs" / "logs"
    for run, model in ((run_a, "model-a"), (run_b, "model-b")):
        logger = RunLoggerV2(run)
        logger.log_run_start({"model": model, "benchmark_name": "demo", "n_cases": 1})
        logger.log_cases(CaseBatch([FailureCase.from_prompt("q", id="c1", label=Label.FAIL)]))
        # discover_runs' marker is run.json + at least one M<n>/log.json.
        logger.log_stage_skipped("M1", "no_signal")
        logger.close()
    publish_report(run_a)

    with TestClient(create_app(run_a, runs_root=tmp_path)) as client:
        listing = client.get("/api/runs")
        assert listing.status_code == 200
        items = listing.json()["items"]
        assert {item["path"] for item in items} == {"exp_a/outputs/logs", "exp_b/outputs/logs"}
        published = next(item for item in items if item["published"])
        assert published["model"] == "model-a"
        unpublished = next(item for item in items if not item["published"])
        assert unpublished["model"] is None

        opened = client.post(f"/api/runs/{unpublished['id']}/open")
        assert opened.status_code == 200
        assert opened.json()["data"]["setting"]["model"] == "model-b"
        # The session actually swapped: /api/report now serves run_b too.
        assert client.get("/api/report").json()["data"]["setting"]["model"] == "model-b"

        assert client.post("/api/runs/not-a-real-id/open").status_code == 404


def test_runs_listing_keeps_the_most_recent_past_the_display_limit(tmp_path, monkeypatch):
    """discover_runs' own cutoff is a traversal safety valve, not recency
    order — the endpoint must sort the full find before truncating to what
    the panel shows, or a walk that reaches new runs late would drop them."""
    from fastapi.testclient import TestClient

    from evalrx.core import CaseBatch, FailureCase, Label
    from evalrx.eval_agent.run_logger_v2 import RunLoggerV2
    from evalrx.reporting import server as server_mod

    monkeypatch.setattr(server_mod, "RUNS_DISPLAY_LIMIT", 2)
    stamps = [1_700_000_000, 1_700_000_100, 1_700_000_200]  # oldest to newest
    runs = []
    for i, stamp in enumerate(stamps):
        run = tmp_path / f"exp_{i}" / "outputs" / "logs"
        logger = RunLoggerV2(run)
        logger.log_cases(CaseBatch([FailureCase.from_prompt("q", id="c1", label=Label.FAIL)]))
        # discover_runs' marker is run.json + at least one M<n>/log.json.
        logger.log_stage_skipped("M1", "no_signal")
        logger.close()
        # _run_mtime takes the max mtime across run.json and every M<n>/log.json
        # (RunLoggerV2 flushes an empty doc for every stage at close(), not
        # just the one actually used) — stamp every file, or an untouched
        # stage's real write time wins regardless.
        for f in run.rglob("*"):
            if f.is_file():
                os.utime(f, (stamp, stamp))
        runs.append(run)

    with TestClient(server_mod.create_app(runs[0], runs_root=tmp_path)) as client:
        listing = client.get("/api/runs").json()
        assert listing["truncated"] is True
        # The two most recent (exp_1, exp_2), never the oldest (exp_0) —
        # regardless of the walk's own traversal/alphabetical order.
        assert {item["path"] for item in listing["items"]} == {"exp_1/outputs/logs", "exp_2/outputs/logs"}


def test_runs_discovery_does_not_descend_into_a_found_runs_data_dir(tmp_path):
    from evalrx.core import CaseBatch, FailureCase, Label
    from evalrx.eval_agent.run_logger_v2 import RunLoggerV2
    from evalrx.reporting.server import discover_runs

    run = tmp_path / "exp" / "outputs" / "logs"
    logger = RunLoggerV2(run)
    logger.log_cases(CaseBatch([FailureCase.from_prompt("q", id="c1", label=Label.FAIL)]))
    logger.log_stage_skipped("M1", "no_signal")
    logger.close()
    # A run's own data/sandbox subtree must never be scanned for more runs,
    # even if it happens to contain another run.json + M<n>/log.json (e.g. a
    # nested coder sandbox that itself invoked evalrx).
    decoy_dir = run / "explore" / "sandbox" / "data"
    (decoy_dir / "M1").mkdir(parents=True)
    (decoy_dir / "run.json").write_text("{}\n", encoding="utf-8")
    (decoy_dir / "M1" / "log.json").write_text("{}\n", encoding="utf-8")

    assert discover_runs(tmp_path) == [run]


def test_external_case_media_is_copied_into_durable_run(tmp_path):
    from evalrx.core import CaseBatch, FailureCase, Inputs
    from evalrx.eval_agent.run_logger_v2 import RunLoggerV2
    from evalrx.reporting.run_events import read_v2_events

    media = tmp_path / "source.wav"
    media.write_bytes(b"RIFF-fake-audio")
    run = tmp_path / "run"
    logger = RunLoggerV2(run)
    logger.log_cases(CaseBatch([
        FailureCase(inputs=Inputs(prompt="listen", audio=str(media)), id="audio-1"),
    ]))
    logger.close()

    events = [e for e in read_v2_events(run) if e.get("event") == "case_record"]
    path = run / events[0]["media_paths"][0]
    assert path.parent == run / "media"
    assert path.read_bytes() == b"RIFF-fake-audio"


def test_unparsed_m3_response_is_recovered_for_audit_only():
    from evalrx.reporting.dynamic import _recover_unparsed_hypotheses

    response = """HYPOTHESIS: Formatting causes the failure.
PLAIN_STATEMENT: The answer changes when formatting changes.
FAILURE_MODE: brittleness
TEST: flip_rate HIGHER on failing cases
EXPECTED_ASSOCIATION: higher_on_failures"""
    proposals = _recover_unparsed_hypotheses(response)
    assert proposals[0]["statement"] == "Formatting causes the failure."
    assert proposals[0]["accepted_by_pipeline"] is False


def test_m1_examples_are_logged_and_reconstructed_from_case_evidence(tmp_path):
    from evalrx.core import CaseBatch, FailureCase, Label, Result
    from evalrx.eval_agent.run_logger_v2 import RunLoggerV2
    from evalrx.reporting.dynamic import build_report_data

    case = FailureCase.from_prompt(
        "Choose the animal", id="case-1", expected="cat", observed="dog", label=Label.FAIL,
    )
    logger = RunLoggerV2(tmp_path)
    logger.log_run_start({"model": "demo", "n_cases": 1})
    logger.log_cases(CaseBatch([case]))
    logger.log_probe(0, {"format_check": Result(
        analyzer="format_check", model="demo",
        findings={"per_case": [{"sample_id": "case-1", "answer_flipped": 1}]},
    )}, cases=CaseBatch([case]))
    logger.close()

    data = build_report_data(tmp_path)
    example = data["stage_detail"]["m1"]["examples"][0]
    assert example["case_id"] == "case-1"
    assert example["baseline_output"] == "dog"
    assert example["check_result"]["answer flipped"] == 1


def test_legacy_m4_example_is_labeled_as_an_aggregate_validation_test():
    from evalrx.reporting.dynamic import _m4_examples

    example = _m4_examples([{
        "hypothesis": "Formatting causes the failure.", "status": "refuted",
        "effect_size": -0.5, "verdict": "The independent test refuted it.",
        "evidence": {"chosen_tool": "signal_label_assoc", "ci": [-0.7, -0.2]},
    }], {}, [])
    assert example[0]["kind"] == "validation_test"
    assert "full validation set" in example[0]["plain_reading"]


def test_report_uses_plain_language_for_internal_signal_names():
    from evalrx.reporting.dynamic import _plain_signal

    label = _plain_signal("format_sensitivity.format_flip_rate")
    assert "answer order" in label.lower()
    assert "format_sensitivity" not in label


def test_indexed_media_can_be_served_from_the_adjacent_example_data_dir(tmp_path):
    from evalrx.reporting.server import _resolve_indexed_media

    root = tmp_path / "example" / "outputs" / "logs"
    root.mkdir(parents=True)
    media = tmp_path / "example" / "data" / "clip.wav"
    media.parent.mkdir()
    media.write_bytes(b"RIFF-fake-audio")

    assert _resolve_indexed_media(root, str(media)) == media.resolve()


def test_explicit_run_log_wins_over_a_nested_logs_subrun(tmp_path):
    from evalrx.reporting.server import _resolve_report_root

    (tmp_path / "run.json").write_text("{}\n")
    (tmp_path / "logs").mkdir()
    (tmp_path / "logs" / "run.json").write_text("{}\n")
    assert _resolve_report_root(tmp_path) == tmp_path


def test_read_only_run_can_be_served_without_publishing(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from evalrx.eval_agent.run_logger_v2 import RunLoggerV2
    from evalrx.reporting.server import create_app

    logger = RunLoggerV2(tmp_path)
    logger.log_run_start({"model": "read-only-demo", "n_cases": 0})
    logger.close()
    monkeypatch.setattr("evalrx.reporting.server.report_is_current", lambda _root: False)
    monkeypatch.setattr(
        "evalrx.reporting.server.publish_report",
        lambda _root: (_ for _ in ()).throw(PermissionError("read only")),
    )
    with TestClient(create_app(tmp_path)) as client:
        payload = client.get("/api/report").json()
    assert payload["layout"]["generated_by"]["mode"] == "deterministic-read-only"
    assert payload["data"]["setting"]["model"] == "read-only-demo"


def test_a_pre_ref_run_still_numbers_its_repairs_the_emitters_way():
    """The AVLM run's stored payload predates `ref`, and must still be readable.

    Left to itself the frontend numbers each list from one, so the frozen
    candidate is #1 on its card and whatever position it happened to occupy in
    the sweep — two numbers, one repair, nothing on screen connecting them.
    """
    from evalrx.reporting.dynamic import _backfill_repair_identity

    payload = {
        "selection": [
            {"name": "visual_grounding", "tier": "L1"},
            {"name": "audio_evidence_then_answer", "tier": "L1"},
            {"name": "coded_pipeline", "tier": "L2"},
        ],
        "attempted": [{"name": "coded_pipeline", "tier": "L2"}],
    }
    _backfill_repair_identity(payload)

    assert [r["ref"] for r in payload["selection"]] == ["R1", "R2", "R3"]
    # Same repair, same number, in both lists.
    assert [r["ref"] for r in payload["attempted"]] == ["R3"]
    # The host authored two of these three and can say what they do.
    assert payload["selection"][0]["headline"].startswith("Tells the model to read")
    assert payload["selection"][2]["headline"].startswith("Runs a short program")
    # The judge invented the middle one under a prompt that never asked for a
    # description, so nothing is invented on its behalf now.
    assert payload["selection"][1]["headline"] == ""


def test_backfill_never_overwrites_what_the_producer_wrote():
    from evalrx.reporting.dynamic import _backfill_repair_identity

    payload = {
        "selection": [
            {"name": "coded_pipeline", "tier": "L2", "ref": "R9",
             "headline": "Re-asks the model and takes the majority answer."},
        ],
        "attempted": [],
    }
    _backfill_repair_identity(payload)

    assert payload["selection"][0]["ref"] == "R9"
    assert payload["selection"][0]["headline"].startswith("Re-asks the model")


def test_stage_figures_embed_everywhere_they_are_cited(tmp_path):
    # M3 re-cites M2's figure files as its own evidence_figures entries; a
    # portable export must inline those too, not just stage_detail.m2.figures
    # (a report/ served over HTTP has /api/artifact, a single file does not).
    from evalrx.reporting.static_export import _embed_stage_figures

    figure_file = tmp_path / "explore" / "figures" / "00_class_balance.png"
    figure_file.parent.mkdir(parents=True)
    figure_file.write_bytes(b"\x89PNG fake")
    run_root = tmp_path / "logs"
    run_root.mkdir()
    cited = "../explore/figures/00_class_balance.png"
    data = {
        "stage_detail": {
            "m2": {"figures": [{"path": cited, "title": "Class balance"}]},
            "m3": {"evidence_figures": [{"path": cited, "title": "FAIL/PASS case balance"}]},
        }
    }
    _embed_stage_figures(run_root, data)
    m2_uri = data["stage_detail"]["m2"]["figures"][0].get("data_uri")
    m3_uri = data["stage_detail"]["m3"]["evidence_figures"][0].get("data_uri")
    assert m2_uri and m2_uri.startswith("data:image/png;base64,")
    assert m3_uri == m2_uri
