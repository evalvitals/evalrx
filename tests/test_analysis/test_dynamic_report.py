from __future__ import annotations

import json


def test_publish_report_contract_and_case_records(tmp_path):
    from evalvitals.core import CaseBatch, FailureCase, Label
    from evalvitals.eval_agent.run_logger import RunLogger
    from evalvitals.reporting.dynamic import load_published_report, publish_report

    logger = RunLogger(tmp_path)
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
    assert envelope["catalog_version"] == "evalvitals-report@1"
    assert envelope["spec"]["elements"]["journey"]["type"] == "Journey"


def test_report_agent_repairs_once_then_validates():
    from evalvitals.reporting.dynamic import ReportAgent

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
    from evalvitals.eval_agent.run_logger import RunLogger
    from evalvitals.reporting.dynamic import load_published_report, publish_report

    logger = RunLogger(tmp_path)
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

    from evalvitals.core import CaseBatch, FailureCase, Label
    from evalvitals.eval_agent.run_logger import RunLogger
    from evalvitals.reporting.dynamic import publish_report
    from evalvitals.reporting.server import create_app

    logger = RunLogger(tmp_path)
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


def test_external_case_media_is_copied_into_durable_run(tmp_path):
    from evalvitals.core import CaseBatch, FailureCase, Inputs
    from evalvitals.eval_agent.run_logger import RunLogger

    media = tmp_path / "source.wav"
    media.write_bytes(b"RIFF-fake-audio")
    run = tmp_path / "run"
    logger = RunLogger(run)
    logger.log_cases(CaseBatch([
        FailureCase(inputs=Inputs(prompt="listen", audio=str(media)), id="audio-1"),
    ]))
    logger.close()

    events = [json.loads(line) for line in (run / "run_log.jsonl").read_text().splitlines()]
    path = run / events[0]["media_paths"][0]
    assert path.parent == run / "artifacts" / "case_media"
    assert path.read_bytes() == b"RIFF-fake-audio"
