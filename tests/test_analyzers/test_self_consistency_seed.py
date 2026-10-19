"""self_consistency draws a case's samples from a seed of its content.

The same case gets the same samples in every run and whatever else ran
before it; another seed draws others; the caller's random state is left as
it was; ``seed=None`` keeps the old behaviour (the global random state).
"""

from __future__ import annotations

import pytest

from evalrx.core.capability import Capability
from evalrx.core.case import CaseBatch, FailureCase, Inputs, Label
from evalrx.core.model import Model

torch = pytest.importorskip("torch")


class _Sampler(Model):
    """Answers drawn from torch's global random state, like a sampling local model."""

    capabilities = frozenset({Capability.GENERATE})
    modalities = frozenset({"text"})

    def generate(self, inputs, **kwargs):
        return ["yes", "no", "maybe", "unsure"][int(torch.randint(0, 4, (1,)))]

    def forward(self, inputs, capture, spec=None):
        raise NotImplementedError


def _case(prompt):
    return FailureCase(inputs=Inputs(prompt=prompt), expected="yes", observed="no", label=Label.FAIL)


def _samples(analyzer, case):
    return analyzer.run(_Sampler(), CaseBatch([case])).artifacts["samples"]


def test_a_case_gets_the_same_samples_in_every_run_whatever_ran_before():
    from evalrx.analyzers.uncertainty.self_consistency import SelfConsistencyAnalyzer

    a, b = _case("is it red?"), _case("is it blue?")
    first = _samples(SelfConsistencyAnalyzer(n=8), a)
    torch.rand(100)                                  # someone else used the random state
    _samples(SelfConsistencyAnalyzer(n=8), b)        # another case ran first
    assert _samples(SelfConsistencyAnalyzer(n=8), a) == first
    # a case with another id but the same content draws the same; another seed draws others
    assert _samples(SelfConsistencyAnalyzer(n=8), _case("is it red?")) == first
    assert _samples(SelfConsistencyAnalyzer(n=8, seed=1), a) != first
    assert len(set(first)) > 1                       # still n samples, not one repeated


def test_the_callers_random_state_is_left_as_it_was_and_none_keeps_the_global_state():
    from evalrx.analyzers.uncertainty.self_consistency import SelfConsistencyAnalyzer

    torch.manual_seed(123)
    expected = torch.rand(3)
    torch.manual_seed(123)
    _samples(SelfConsistencyAnalyzer(n=4), _case("is it red?"))
    assert torch.equal(torch.rand(3), expected)
    unseeded = SelfConsistencyAnalyzer(n=8, seed=None)
    torch.manual_seed(5)
    one = _samples(unseeded, _case("is it red?"))
    torch.manual_seed(6)
    assert _samples(unseeded, _case("is it red?")) != one
