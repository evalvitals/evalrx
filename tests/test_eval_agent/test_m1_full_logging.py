"""M1 (ProbeAgent) full-output logging.

RunLoggerV2.log_probe must capture *everything* M1 generates so a run is
fully observable afterwards: per-analyzer COMPLETE results (metadata +
n_cases + rendered summary, not just the inlined findings), the heavy
artifacts, and the analyzers that were selected but errored at runtime.
"""

from __future__ import annotations

import json


def test_log_probe_persists_full_results_and_failed_analyzers(tmp_path):
    from evalrx.core.result import Result
    from evalrx.eval_agent.run_logger_v2 import RunLoggerV2

    logger = RunLoggerV2(run_dir=tmp_path / "run", observability_mode="offline")
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

    m1 = json.loads((tmp_path / "run" / "M1" / "log.json").read_text())
    probe = m1["probe"][-1]

    # selected-but-errored analyzers are observable, not silently absent
    assert probe["failed_analyzers"] == {
        "logit_lens": "RuntimeError: model exposes no hidden states"}

    # the COMPLETE result is inlined into the probe entry, not a sibling file
    doc = probe["results"]["self_consistency"]
    assert doc["analyzer"] == "self_consistency"
    assert doc["metadata"] == {"strategy": "vote", "note": "diagnostic-only"}
    assert doc["findings"]["consistency"] == 0.2
    # summary() is rendered into the doc even though Result.to_dict() omits it
    assert "self_consistency" in doc["summary"]


def test_log_probe_without_failures_omits_failed_analyzers(tmp_path):
    """No failures -> no failed_analyzers key (kept optional/additive)."""
    from evalrx.core.result import Result
    from evalrx.eval_agent.run_logger_v2 import RunLoggerV2

    logger = RunLoggerV2(run_dir=tmp_path / "run", observability_mode="offline")
    logger.log_probe(0, {"pope": Result(analyzer="pope", model="m", findings={"acc": 0.9})})
    logger.close()

    m1 = json.loads((tmp_path / "run" / "M1" / "log.json").read_text())
    probe = m1["probe"][-1]
    assert "failed_analyzers" not in probe
    assert "pope" in probe["results"]
