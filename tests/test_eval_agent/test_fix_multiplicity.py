"""Fix module: e-BH multiplicity control over the best-of-N candidate family.

Selecting the best of many candidates validated against the SAME baseline is a
multiple-comparisons hazard: each cleared only its own per-candidate gate
(e >= 1/alpha). A candidate is a winner only if it ALSO survives e-BH FDR across
the whole tested family (sole-survivor bar rises to m/alpha). These tests pin
that an underpowered best-of-N is correctly culled, while a single candidate and
a well-powered winner are unaffected.
"""

from __future__ import annotations

from evalrx.core.capability import Capability
from evalrx.core.case import CaseBatch, FailureCase, Inputs, Label
from evalrx.core.model import Model
from evalrx.eval_agent.hypothesis import Hypothesis
from evalrx.eval_agent.stages.fix_agent import FixAgent


def _hyp(s: str) -> Hypothesis:
    return Hypothesis(statement=s, target_model="m", predicted_failure_mode="")


def _yes_batch(n: int) -> CaseBatch:
    yes = {"all_of": ["yes"], "none_of": ["no"]}
    return CaseBatch([
        FailureCase(id=f"c{i}", inputs=Inputs(prompt=f"Is there a lesion {i}?"),
                    expected=yes, label=Label.FAIL)
        for i in range(n)
    ])


class _CarefulFixModel(Model):
    """Answers 'No' unless the prompt says 'carefully' -> 'Yes'."""

    capabilities = frozenset({Capability.GENERATE})
    modalities = frozenset({"text"})

    def generate(self, inputs, **kwargs):
        p = str(getattr(inputs, "prompt", inputs)).lower()
        return "Yes." if "carefully" in p else "No."

    def forward(self, inputs, capture, spec=None):
        raise NotImplementedError


class _MultiCandidateJudge(Model):
    """Proposes several L1 templates; exactly one ('careful') actually fixes."""

    capabilities = frozenset({Capability.GENERATE})
    modalities = frozenset({"text"})

    def generate(self, inputs, **kwargs) -> str:
        import json
        return json.dumps([
            {"name": "polite", "prompt_template": "Please answer. {prompt}"},
            {"name": "terse", "prompt_template": "Briefly: {prompt}"},
            {"name": "careful", "prompt_template": "Look carefully. {prompt}"},
        ])

    def forward(self, inputs, capture, spec=None):
        raise NotImplementedError


def test_underpowered_best_of_N_is_culled_by_ebh():
    """One real fix (8/8, e~28) among a 3-candidate family: clears its own gate
    but NOT the family bar (m/alpha = 60) -> outcome is NOT fixed, and the
    recommendation names the e-BH multiplicity cull."""
    agent = FixAgent(judge=_MultiCandidateJudge(), max_tier="L1")
    out = agent.propose_and_validate(_CarefulFixModel(), _yes_batch(8), [_hyp("x")])

    careful = next(v for v in out.attempted if v.candidate.name == "careful")
    assert careful.fixed is True                      # cleared its OWN gate
    assert out.fixed is False                          # but not the family e-BH
    assert "careful" not in out.ebh_survivors
    assert out.recommendation and "e-BH" in out.recommendation["reason"]


def test_well_powered_best_of_N_survives_ebh():
    """Same family, but 24 cases: the real fix reaches a large e-value that
    clears the m/alpha family bar -> validated and selected."""
    agent = FixAgent(judge=_MultiCandidateJudge(), max_tier="L1")
    out = agent.propose_and_validate(_CarefulFixModel(), _yes_batch(24), [_hyp("x")])

    assert out.fixed is True
    assert out.best is not None and out.best.candidate.name == "careful"
    assert "careful" in out.ebh_survivors
    assert out.recommendation is None


def test_single_candidate_is_not_penalised():
    """m=1: the e-BH bar reduces to the per-candidate gate (1/alpha), so a lone
    8/8 winner is unaffected by the multiplicity machinery."""
    import json

    class _OneFix(Model):
        capabilities = frozenset({Capability.GENERATE})
        modalities = frozenset({"text"})

        def generate(self, inputs, **kwargs) -> str:
            if "EXECUTION CONTRACT" in str(inputs) or "JSON array" not in str(inputs):
                pass
            return json.dumps([{"name": "careful", "prompt_template": "Look carefully. {prompt}"}])

        def forward(self, inputs, capture, spec=None):
            raise NotImplementedError

    agent = FixAgent(judge=_OneFix(), max_tier="L1")
    out = agent.propose_and_validate(_CarefulFixModel(), _yes_batch(8), [_hyp("x")])
    assert out.fixed is True
    assert out.best is not None and out.best.candidate.name == "careful"
    assert out.ebh_survivors == ["careful"]
