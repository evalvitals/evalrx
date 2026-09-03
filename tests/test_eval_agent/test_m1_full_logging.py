"""M1 (ProbeAgent) full-output logging.

RunLogger.log_probe must capture *everything* M1 generates so a run is fully
observable afterwards: per-analyzer COMPLETE results (metadata + n_cases +
rendered summary, not just the inlined findings), the heavy artifacts, and the
analyzers that were selected but errored at runtime.
"""

from __future__ import annotations

import json


def test_log_probe_persists_full_results_and_failed_analyzers(tmp_path):
    from evalrx.core.result import Result
    from evalrx.eval_agent.run_logger import RunLogger

    logger = RunLogger(run_dir=tmp_path / "run")
    res = Result(
        analyzer="self_consistency",
        model="FakeVLM()",
        findings={"consistency": 0.2, "n_samples": 5},
        metadata={"strategy": "vote", "note": "diagnostic-only"},
    )
    logger.log_probe(
        0, {"self_consistency": res},
        failed_analyzers={"logit_lens": "RuntimeError: model exposes no hidden states"},
    )
    logger.close()

    events = [json.loads(line) for line in
              (tmp_path / "run" / "run_log.jsonl").read_text().splitlines()]
    probe = [e for e in events if e["event"] == "probe"][-1]

    # selected-but-errored analyzers are observable, not silently absent
    assert probe["failed_analyzers"] == {
        "logit_lens": "RuntimeError: model exposes no hidden states"}

    # the COMPLETE result is written to disk and pointed to from the event
    rel = probe["result_paths"]["self_consistency"]
    doc = json.loads((tmp_path / "run" / rel).read_text())
    assert doc["analyzer"] == "self_consistency"
    assert doc["metadata"] == {"strategy": "vote", "note": "diagnostic-only"}
    assert doc["findings"]["consistency"] == 0.2
    # summary() is rendered into the file even though Result.to_dict() omits it
    assert "self_consistency" in doc["summary"]


def test_log_probe_without_failures_omits_failed_analyzers(tmp_path):
    """No failures -> no failed_analyzers key (kept optional/additive)."""
    from evalrx.core.result import Result
    from evalrx.eval_agent.run_logger import RunLogger

    logger = RunLogger(run_dir=tmp_path / "run")
    logger.log_probe(0, {"pope": Result(analyzer="pope", model="m", findings={"acc": 0.9})})
    logger.close()

    events = [json.loads(line) for line in
              (tmp_path / "run" / "run_log.jsonl").read_text().splitlines()]
    probe = [e for e in events if e["event"] == "probe"][-1]
    assert "failed_analyzers" not in probe
    assert "pope" in probe["result_paths"]
