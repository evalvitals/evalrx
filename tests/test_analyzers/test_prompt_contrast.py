"""PromptContrastAnalyzer: paired prompt interventions → by_strategy + repair flags.

P2 of the depth roadmap: intervention-grade (causal) evidence for decision-layer
failure hypotheses, flowing into the paired McNemar/Friedman stats machinery.
"""

from __future__ import annotations

from evalvitals.analyzers.perturbation.prompt_contrast import (
    PromptContrastAnalyzer,
    _default_score,
)
from evalvitals.core.capability import Capability
from evalvitals.core.case import CaseBatch, FailureCase, Inputs, Label
from evalvitals.core.model import Model


class InterventionModel(Model):
    """Baseline answers "no" to everything; interventions repair it.

    - prompts containing "describe" → describes then answers correctly ("yes")
    - prompts containing "subtle"   → answers "yes" (sensitivity instruction works)
    - plain baseline                → always "no"
    """

    capabilities = frozenset({Capability.GENERATE})
    modalities = frozenset({"text", "image"})

    def generate(self, inputs, **kwargs):
        p = str(getattr(inputs, "prompt", inputs)).lower()
        if "describe" in p:
            return "The region shows an irregular opacity. Yes."
        if "subtle" in p:
            return "Yes."
        return "No."

    def forward(self, inputs, capture, spec=None):
        raise NotImplementedError


def _batch() -> CaseBatch:
    yes = {"all_of": ["yes"], "none_of": ["no"]}
    no = {"all_of": ["no"], "none_of": ["yes"]}
    return CaseBatch([
        FailureCase(id="y0", inputs=Inputs(prompt="Is there an effusion?"),
                    expected=yes, label=Label.FAIL),
        FailureCase(id="y1", inputs=Inputs(prompt="Is the heart enlarged?"),
                    expected=yes, label=Label.FAIL),
        FailureCase(id="n0", inputs=Inputs(prompt="Is there a fracture?"),
                    expected=no, label=Label.PASS),
    ])


def test_by_strategy_success_matrix():
    res = PromptContrastAnalyzer().run(InterventionModel(), _batch())
    bs = res.findings["by_strategy"]
    assert set(bs) == {"baseline", "describe_first", "sensitive"}
    # baseline: "no" everywhere → wrong on gold-yes, right on gold-no
    assert bs["baseline"] == {"y0": 0.0, "y1": 0.0, "n0": 1.0}
    # interventions: "yes" everywhere → repair gold-yes, break gold-no
    assert bs["sensitive"] == {"y0": 1.0, "y1": 1.0, "n0": 0.0}
    assert res.findings["success_rate_baseline"] == round(1 / 3, 4)
    assert res.findings["success_rate_sensitive"] == round(2 / 3, 4)


def test_repair_and_breakage_flags():
    res = PromptContrastAnalyzer().run(InterventionModel(), _batch())
    by_id = {e["sample_id"]: e for e in res.artifacts["per_case"]}
    assert by_id["y0"]["fixed_by_sensitive"] is True
    assert by_id["y0"]["broken_by_sensitive"] is False
    assert by_id["n0"]["fixed_by_sensitive"] is False
    assert by_id["n0"]["broken_by_sensitive"] is True
    assert res.findings["n_fixed_by_sensitive"] == 2
    assert res.findings["n_broken_by_sensitive"] == 1


def test_custom_strategies_baseline_auto_added():
    a = PromptContrastAnalyzer(strategies={"cot": "Think step by step. {prompt}"})
    assert "baseline" in a.strategies and "cot" in a.strategies
    res = a.run(InterventionModel(), _batch())
    assert set(res.findings["by_strategy"]) == {"baseline", "cot"}


def test_unscored_cases_counted():
    cases = CaseBatch([FailureCase(id="u", inputs=Inputs(prompt="describe this"))])
    res = PromptContrastAnalyzer().run(InterventionModel(), cases)
    # no expected rubric → every strategy call unscored
    assert res.findings["n_unscored"] == 3
    assert res.findings["by_strategy"]["baseline"] == {}


def test_default_score_word_boundary():
    case = FailureCase(inputs=Inputs(prompt="q"),
                       expected={"all_of": ["no"], "none_of": ["yes"]})
    assert _default_score(case, "No abnormality.") is True
    assert _default_score(case, "Nothing seen, all normal.") is False  # no bare "no"
    case2 = FailureCase(inputs=Inputs(prompt="q"), expected="axial")
    assert _default_score(case2, "This is an axial CT slice.") is True


def test_stats_layer_runs_paired_contrasts():
    from evalvitals.analysis.stats_tools import (
        build_stats_input,
        default_plan,
        run_stats_tool,
    )

    batch = _batch()
    res = PromptContrastAnalyzer().run(InterventionModel(), batch)
    inp = build_stats_input({"prompt_contrast": res}, batch)
    assert set(inp.groups) == {"baseline", "describe_first", "sensitive"}

    plan = default_plan(inp)
    tools = [t for t, _, _ in plan]
    assert "friedman_nemenyi" in tools
    # Pairwise paired contrasts against the baseline strategy.
    pairs = [c["strategies"] for t, c, _ in plan if t == "mcnemar_evalue"]
    assert ["baseline", "describe_first"] in pairs
    assert ["baseline", "sensitive"] in pairs

    r = run_stats_tool("mcnemar_evalue", inp, {"strategies": ["baseline", "sensitive"]})
    assert r.ok and r.effect is not None
    # sensitive repairs 2 and breaks 1 → net positive effect
    assert r.effect > 0

# ── WS3: non-tautological per-case signal surfaced to M2 (prompt_sensitivity) ──

class _ConsistentModel(Model):
    """Answers the same thing under every prompt (prompt-robust)."""
    capabilities = frozenset({Capability.GENERATE})
    modalities = frozenset({"text", "image"})

    def generate(self, inputs, **kwargs):
        return "Yes."

    def forward(self, inputs, capture, spec=None):
        raise NotImplementedError


def test_prompt_sensitivity_surfaced_to_findings_per_case():
    res = PromptContrastAnalyzer().run(InterventionModel(), _batch())
    pc = res.findings["per_case"]
    assert pc and all("prompt_sensitivity" in e and "sample_id" in e for e in pc)
    # baseline "no" vs describe/subtle "yes" → answer flips → sensitivity 1/3.
    assert all(abs(e["prompt_sensitivity"] - 0.3333) < 1e-3 for e in pc)
    # The tautological repair flags stay OUT of findings["per_case"].
    assert all("fixed_by_describe_first" not in e for e in pc)


# ── audio: dataclasses.replace must preserve non-image modality fields ──

class _AudioEchoModel(Model):
    """Answers "yes" iff .audio survived onto the re-asked Inputs."""

    capabilities = frozenset({Capability.GENERATE})
    modalities = frozenset({"text", "audio"})

    def generate(self, inputs, **kwargs):
        return "Yes." if getattr(inputs, "audio", None) is not None else "No."

    def forward(self, inputs, capture, spec=None):
        raise NotImplementedError


def test_audio_input_preserved_across_strategies():
    yes = {"all_of": ["yes"], "none_of": ["no"]}
    cases = CaseBatch([
        FailureCase(id="a0", inputs=Inputs(prompt="What sound is this?", audio="clip.wav"),
                    expected=yes, label=Label.FAIL),
    ])
    res = PromptContrastAnalyzer().run(_AudioEchoModel(), cases)
    # Every strategy re-generate must still carry .audio through, not just baseline.
    for strat, scores in res.findings["by_strategy"].items():
        assert scores["a0"] == 1.0, f"strategy {strat!r} dropped .audio on re-ask"


def test_prompt_robust_model_has_zero_sensitivity():
    res = PromptContrastAnalyzer().run(_ConsistentModel(), _batch())
    pc = res.findings["per_case"]
    assert pc and all(e["prompt_sensitivity"] == 0.0 for e in pc)


def test_prompt_sensitivity_is_a_clean_stats_signal():
    from evalvitals.analysis.stats_tools import build_stats_input

    batch = _batch()
    res = PromptContrastAnalyzer().run(InterventionModel(), batch)
    inp = build_stats_input({"prompt_contrast": res}, batch)
    # Enters the tested family; not isolated as a label leak.
    assert "prompt_contrast.prompt_sensitivity" in inp.per_case
    assert "prompt_contrast.prompt_sensitivity" not in inp.sanity
