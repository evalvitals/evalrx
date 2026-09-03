"""Analyzers measure every case by default.

Until 2026-08 every per-case analyzer shipped its own ``max_cases`` cap (16 for
``coverage_verification_gap``, 32 for ``selfcheck_consistency``, 64 for
``termination_audit`` ...). The explore/confirm split hands M1 and the held-out
M5 a whole partition each, and those caps silently cut both down to a few dozen
rows: on chartqa (128/128) the M5 "confirmation" of a selfcheck-based lead ran
on 32 cases. The cap is now opt-in (``max_cases=N``); ``0`` means every case.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path

from evalrx.core.capability import Capability
from evalrx.core.case import CaseBatch, FailureCase, Inputs, Label
from evalrx.core.model import Model
from evalrx.core.registry import registry


def _batch(n_fail: int, n_pass: int) -> CaseBatch:
    cases = [
        FailureCase(id=f"f{i}", inputs=Inputs(prompt=f"q{i}"), observed="Answer: 1",
                    expected="1", label=Label.FAIL)
        for i in range(n_fail)
    ] + [
        FailureCase(id=f"p{i}", inputs=Inputs(prompt=f"q{i}"), observed="Answer: 2",
                    expected="2", label=Label.PASS)
        for i in range(n_pass)
    ]
    return CaseBatch(cases)


class _Echo(Model):
    capabilities = frozenset({Capability.GENERATE})
    modalities = frozenset({"text"})

    def generate(self, inputs, **kwargs):
        return "Answer: 1"

    def logprobs(self, inputs, **kwargs):  # pragma: no cover
        raise NotImplementedError

    def forward(self, inputs, capture, spec=None):  # pragma: no cover
        raise NotImplementedError


def test_every_registered_analyzer_defaults_max_cases_to_every_case():
    """The cap is opt-in everywhere: no analyzer silently subsamples the batch."""
    seen = []
    for name, cls in sorted(registry.analyzers.all().items()):
        try:
            params = inspect.signature(cls.__init__).parameters
        except (TypeError, ValueError):
            continue
        spec = params.get("max_cases")
        if spec is None or spec.default is inspect.Parameter.empty:
            continue
        seen.append(name)
        assert spec.default in (0, None), (
            f"{name} caps its cases at {spec.default} by default; 0 (= every case) "
            "is the library default — pass max_cases=N to opt into a cap")
    assert len(seen) >= 15, seen  # the cap knob still exists where it used to


def test_stratified_head_zero_is_every_case_in_document_order():
    batch = _batch(60, 240)
    assert [c.id for c in batch.stratified_head(0)] == [c.id for c in batch]
    # an explicit cap still stratifies (FAIL gets up to half the budget)
    capped = batch.stratified_head(32)
    assert len(capped) == 32 and sum(c.label is Label.FAIL for c in capped) == 16


def test_answer_extraction_audit_measures_a_batch_larger_than_the_old_cap():
    from evalrx.analyzers.reasoning.answer_extraction_audit import AnswerExtractionAudit

    batch = _batch(50, 200)  # 250 > the old default of 200
    result = AnswerExtractionAudit().run(_Echo(), batch)
    assert len(result.findings["per_case"]) == 250
    # opting in still works exactly as before
    result = AnswerExtractionAudit(max_cases=40).run(_Echo(), batch)
    assert len(result.findings["per_case"]) == 40


def test_probe_generator_collects_every_case_unless_capped(tmp_path):
    from evalrx.agent_runtime.sandbox import ExperimentSandbox
    from evalrx.eval_agent.stages.probe_generator import _INPUT_FILENAME, ProbeGenerator

    batch = _batch(3, 4)

    def _collected(**kw) -> int:
        sandbox = ExperimentSandbox(workdir=tmp_path / f"sb{len(kw)}", cleanup=False)
        gen = ProbeGenerator(sandbox=sandbox, **kw)
        gen._collect_outputs(_Echo(), batch)
        recorded = json.loads((Path(sandbox.workdir) / _INPUT_FILENAME).read_text())
        return len(recorded["cases"])

    assert _collected() == 7
    assert _collected(max_cases=2) == 2
