"""Fix module — validation-end fixes and proposer context (2026-08-18).

What these pin down (each was a measured cause of a "regressed"/"unsafe"
verdict on the qwen3.5-2b bbh_tracking7 / bbh_word_sorting runs):

* a judge-proposed ``max_tokens`` BELOW the baseline budget is raised to it
  (a shorter budget truncates the chain -> wrong answer -> "regressed" by
  decoding, not by idea), the proposal is kept for the record, and the
  bridge enforces the same floor for coded pipelines;
* ``n_samples`` applies to EVERY strategy (a ``least_to_most, n_samples=3``
  used to run once) and samples vote on the extracted final answer, not on
  the whole chain text (where no two CoT samples ever match);
* what each candidate PRODUCED per case is captured (``FixValidation.outputs``)
  and persisted as ``outputs.jsonl`` beside the record, and calls that hit
  the decode cap are counted (``n_truncated``);
* ``run_fix`` drops hypotheses M4's experiment REFUTED and tells the proposer;
* the proposer sees full example cases (prompt + baseline output + expected
  answer) from a DISJOINT split, the scoring rule, the baseline decoding, the
  M2/M5/explore evidence — and no image-tool catalog on a text-only batch;
* the built-in ``self_consistency_5`` is a FLOOR of the tested family for
  text-only batches, not a fallback for a silent judge.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from evalvitals.core.capability import Capability
from evalvitals.core.case import CaseBatch, FailureCase, Inputs, Label
from evalvitals.core.model import Model
from evalvitals.eval_agent.hypothesis import Hypothesis, HypothesisStatus
from evalvitals.eval_agent.stages.fix_agent import (
    FixAgent,
    FixCandidate,
    FixContext,
    _format_examples,
)
from evalvitals.eval_agent.stages.fix_tiers import FixTier
from evalvitals.eval_agent.stages.fix_tools import (
    MAX_TOKENS_CAP,
    PipelineSpec,
    answer_key,
    run_pipeline,
)


def _hyp(statement: str, mode: str = "") -> Hypothesis:
    return Hypothesis(statement=statement, target_model="m", predicted_failure_mode=mode)


def _mc_batch(n_fail: int = 3, n_pass: int = 3) -> CaseBatch:
    """Multiple-choice text cases scored by 'the last Answer: line == gold'."""
    cases = []
    for i in range(n_fail + n_pass):
        gold = "(B)"
        fail = i < n_fail
        cases.append(
            FailureCase(
                id=f"q{i}",
                inputs=Inputs(prompt=f"Question {i}: pick one.\nOptions:\n(A) x\n(B) y\n(C) z\n"
                                     "Put the final answer as 'Answer: (X)'."),
                expected=gold,
                observed=("Reasoning... Answer: (C)" if fail else "Reasoning... Answer: (B)"),
                label=Label.FAIL if fail else Label.PASS,
                metadata={"gold": gold},
            )
        )
    return CaseBatch(cases)


def _mc_score(case, output):
    """Last 'Answer:' span, punctuation-insensitive letter match."""
    import re

    m = re.findall(r"answer:\s*\(?([a-z])\)?", str(output).lower())
    if not m:
        return False
    return m[-1] == str(case.metadata["gold"]).strip("()").lower()


class CountingModel(Model):
    """Answers (B) and records every kwargs it was called with."""

    capabilities = frozenset({Capability.GENERATE})
    modalities = frozenset({"text"})

    def __init__(self, reply: str = "Chain of thought.\nAnswer: (B)") -> None:
        self.calls: list[dict] = []
        self.reply = reply
        self.n_truncated = 0

    def generate(self, inputs, **kwargs):
        self.calls.append(dict(kwargs))
        return self.reply

    def forward(self, inputs, capture, spec=None):
        raise NotImplementedError


class ScriptedJudge(Model):
    capabilities = frozenset({Capability.GENERATE})
    modalities = frozenset({"text"})

    def __init__(self, reply: str) -> None:
        self._reply = reply
        self.prompts: list[str] = []

    def generate(self, inputs, **kwargs) -> str:
        self.prompts.append(str(getattr(inputs, "prompt", inputs)))
        return self._reply

    def forward(self, inputs, capture, spec=None):
        raise NotImplementedError


# ── run_pipeline: n_samples for every strategy + label-free vote key ──────────


def test_n_samples_applies_to_multi_call_strategies():
    """least_to_most with n_samples=3 = three end-to-end passes (2 calls each),
    not one pass with n_samples silently ignored."""
    model = CountingModel()
    case = list(_mc_batch(1, 0))[0]
    spec = PipelineSpec(name="l2m", strategy="least_to_most", n_samples=3)
    capture: dict = {}
    assert run_pipeline(model, case, spec, _mc_score, capture=capture) is True
    assert len(model.calls) == 6
    assert capture["n_calls"] == 6 and len(capture["outputs"]) == 3
    assert capture["winner"].endswith("Answer: (B)")


def test_vote_uses_extracted_final_answer_not_whole_text():
    """CoT samples never match byte-for-byte; the vote must be on the final
    answer, or n_samples=5 degenerates to 'first sample'."""
    replies = iter([
        "long chain one ... Answer: (C)",
        "a different chain ... Answer: (B)",
        "yet another chain ... Answer: (B)",
    ])

    class Sampler(CountingModel):
        def generate(self, inputs, **kwargs):
            return next(replies)

    case = list(_mc_batch(1, 0))[0]
    spec = PipelineSpec(name="sc", n_samples=3)
    capture: dict = {}
    assert run_pipeline(Sampler(), case, spec, _mc_score, capture=capture) is True
    assert capture["winner"].endswith("Answer: (B)")
    # the key itself is label-free and format-normalised
    assert answer_key("blah\nAnswer: (B)") == answer_key("other\nAnswer: B")
    assert answer_key("FINAL: 4 and more", r"FINAL:\s*(\d+)") == "4"


def test_label_free_consensus_override_preserves_baseline_on_any_countervote():
    case = list(_mc_batch(1, 0))[0]  # recorded baseline answer is (C)

    class Sampler(Model):
        capabilities = frozenset({Capability.GENERATE})
        modalities = frozenset({"text"})

        def __init__(self, replies):
            self.replies = iter(replies)

        def generate(self, inputs, **kwargs):
            return next(self.replies)

        def forward(self, inputs, capture, spec=None):
            raise NotImplementedError

    spec = PipelineSpec(
        name="guarded", n_samples=3, baseline_override_min_support=2
    )
    capture: dict = {}
    mixed = ["Answer: (B)", "Answer: (B)", "Answer: (C)"]
    assert run_pipeline(Sampler(mixed), case, spec, _mc_score, capture=capture) is False
    assert capture["winner"] == case.observed

    unanimous = ["Answer: (B)", "Answer: (B)", "Answer: (B)"]
    assert run_pipeline(Sampler(unanimous), case, spec, _mc_score) is True
    roundtrip = PipelineSpec.from_dict(spec.to_dict())
    assert roundtrip is not None and roundtrip.baseline_override_min_support == 2


def test_max_tokens_cap_is_above_long_form_baselines():
    assert MAX_TOKENS_CAP >= 20480
    spec = PipelineSpec.from_dict({"name": "long", "generation_kwargs": {"max_tokens": 20480}})
    assert spec.generation_kwargs == {"max_tokens": 20480}


# ── the baseline budget is a floor ────────────────────────────────────────────


def test_judge_max_tokens_below_baseline_is_raised_to_the_floor():
    judge = ScriptedJudge(json.dumps([{
        "name": "short_budget", "prompt_template": "{prompt}",
        "generation_kwargs": {"max_tokens": 900, "temperature": 0.7}, "n_samples": 1,
    }]))
    agent = FixAgent(judge=judge, max_tier="L2", allow_codegen=False,
                     baseline_generation_kwargs={"max_tokens": 4096, "temperature": 0.6},
                     floor_candidates=())
    model = CountingModel()
    out = agent.propose_and_validate(model, _mc_batch(), [_hyp("h")], context=FixContext())
    cand = next(v.candidate for v in out.attempted if v.candidate.name == "short_budget")
    assert cand.payload["generation_kwargs"]["max_tokens"] == 4096
    assert cand.payload["generation_kwargs_proposed"]["max_tokens"] == 900
    # and the model was really called with the floor
    assert all(c.get("max_tokens") == 4096 for c in model.calls)
    # the proposer was told about the budget and the floor rule
    assert "BASELINE DECODING: max_tokens=4096" in judge.prompts[-1]
    assert "raised to 4096" in judge.prompts[-1]


def test_floor_is_inferred_from_the_model_when_not_given():
    class Budgeted(CountingModel):
        max_tokens = 2048

    agent = FixAgent(judge=None, max_tier="L2", allow_codegen=False, floor_candidates=())
    assert agent._resolve_max_tokens_floor(Budgeted(), _mc_batch()) == 2048
    # case metadata as the last resort
    batch = _mc_batch()
    for c in batch:
        c.metadata["generation_config"] = {"max_tokens": 1024}
    assert agent._resolve_max_tokens_floor(CountingModel(), batch) == 1024
    assert agent._resolve_max_tokens_floor(CountingModel(), _mc_batch()) is None


def test_bridge_enforces_floor_and_passes_generation_kwargs(tmp_path):
    from evalvitals.eval_agent.stages.fix_pipeline import run_coded_pipeline

    code = '''
import json
cases = json.load(open("fix_cases.json"))["cases"]
out = []
for c in cases:
    assert "baseline_output" in c
    a = model_generate(c["id"], generation_kwargs={"max_tokens": 100, "temperature": 0.2, "bogus": 1})
    out.append({"sample_id": c["id"], "output": a})
print("FIX_PIPELINE_RESULT_JSON=" + json.dumps({"per_case": out}))
'''
    model = CountingModel()
    res = run_coded_pipeline(code, model, _mc_batch(2, 0), workdir=tmp_path,
                             timeout_sec=60, max_tokens_floor=4096)
    assert res.ok, res.error
    assert model.calls and all(
        c == {"max_tokens": 4096, "temperature": 0.2} for c in model.calls
    )


# ── outputs + truncation telemetry ────────────────────────────────────────────


def test_candidate_outputs_and_truncation_are_recorded(tmp_path):
    from evalvitals.eval_agent.run_logger import RunLogger

    class Truncating(CountingModel):
        def generate(self, inputs, **kwargs):
            self.n_truncated += 1  # every call "hits the cap"
            return super().generate(inputs, **kwargs)

    judge = ScriptedJudge(json.dumps([
        {"name": "rewrite", "prompt_template": "Think step by step. {prompt}"},
    ]))
    logger = RunLogger(run_dir=tmp_path / "logs")
    agent = FixAgent(judge=judge, max_tier="L1", run_logger=logger)
    batch = _mc_batch(2, 2)
    out = agent.propose_and_validate(Truncating(), batch, [_hyp("h")])
    v = next(v for v in out.attempted if v.candidate.name == "rewrite")
    assert set(v.outputs) == {c.id for c in batch}
    assert all(o.endswith("Answer: (B)") for o in v.outputs.values())
    assert v.n_truncated == len(batch)
    assert "hit the decode cap" in v.summary
    # persisted beside the record, not inline in the event
    logger.close()
    fix_events = [json.loads(l) for l in (tmp_path / "logs" / "run_log.jsonl").read_text().splitlines()
                  if '"event": "fix"' in l]
    assert fix_events
    att = fix_events[-1]["attempted"][0]
    assert "outputs" not in att and att["n_outputs"] == len(batch)
    assert att["n_truncated"] == len(batch)
    rows = [json.loads(l) for l in
            next((tmp_path / "logs" / "fixes").glob("*rewrite*/outputs.jsonl")).read_text().splitlines()]
    assert {r["case_id"] for r in rows} == {c.id for c in batch}
    assert {r["status"] for r in rows} <= {"fixed", "broken", "unchanged"}
    # the feedback block for a next round names truncation as the cause
    assert "hit the decode cap" in FixAgent._format_prior([v])


# ── proposer context ──────────────────────────────────────────────────────────


def test_examples_show_outputs_and_gold_only_from_disjoint_cases():
    batch = _mc_batch(2, 1)
    with_gold = _format_examples(batch, with_gold=True)
    without = _format_examples(batch, with_gold=False)
    assert "MODEL OUTPUT" in with_gold and "Answer: (C)" in with_gold
    assert "EXPECTED: (B)" in with_gold
    assert "EXPECTED" not in without and "withheld" in without
    assert "PASS case" in with_gold  # a PASS contrast is included


def test_proposer_sees_context_examples_scoring_and_no_image_catalog_for_text():
    judge = ScriptedJudge("[]")
    agent = FixAgent(judge=judge, max_tier="L2", allow_codegen=False,
                     scoring_note="last 'Answer:' line, letter match",
                     baseline_generation_kwargs={"max_tokens": 4096, "temperature": 0.6})
    explore = _mc_batch(2, 1)
    ctx = FixContext(example_cases=explore, evidence="  M2: signal foo vs FAIL effect +0.9",
                     refuted=["grading-mismatch hypothesis [experiment: verdict=0]"],
                     task_note="BBH tracking of shuffled objects")
    agent.propose_and_validate(CountingModel(), _mc_batch(2, 2), [_hyp("h")], context=ctx)
    l2_prompt = judge.prompts[-1]
    assert "EXPECTED: (B)" in l2_prompt              # full examples from the explore split
    assert "last 'Answer:' line" in l2_prompt        # scoring rule
    assert "BASELINE DECODING: max_tokens=4096" in l2_prompt
    assert "M2: signal foo" in l2_prompt              # evidence
    assert "REFUTED BY AN INTERVENTION EXPERIMENT" in l2_prompt
    assert "BBH tracking" in l2_prompt
    assert "text-only" in l2_prompt and "zoom_center" not in l2_prompt
    assert "self_consistency_5" in l2_prompt          # told what is already in the family


def test_without_disjoint_examples_gold_is_withheld():
    judge = ScriptedJudge("[]")
    agent = FixAgent(judge=judge, max_tier="L1")
    agent.propose_and_validate(CountingModel(), _mc_batch(2, 2), [_hyp("h")])
    assert "EXPECTED" not in judge.prompts[-1] and "withheld" in judge.prompts[-1]


def test_floor_candidate_is_tested_next_to_judge_proposals():
    judge = ScriptedJudge(json.dumps([
        {"name": "judge_idea", "prompt_template": "Careful. {prompt}"},
    ]))
    agent = FixAgent(judge=judge, max_tier="L2", allow_codegen=False,
                     baseline_generation_kwargs={"temperature": 0.6})
    out = agent.propose_and_validate(CountingModel(), _mc_batch(), [_hyp("h")])
    names = [v.candidate.name for v in out.attempted]
    assert "judge_idea" in names and "self_consistency_5" in names
    sc = next(v.candidate for v in out.attempted if v.candidate.name == "self_consistency_5")
    assert sc.source == "floor" and sc.payload["n_samples"] == 5
    assert sc.payload["generation_kwargs"]["temperature"] == 0.6  # inherits a stochastic baseline

    off = FixAgent(judge=judge, max_tier="L2", allow_codegen=False, floor_candidates=())
    out2 = off.propose_and_validate(CountingModel(), _mc_batch(), [_hyp("h")])
    assert "self_consistency_5" not in [v.candidate.name for v in out2.attempted]


def test_floor_skipped_for_image_yes_no_batches():
    pytest.importorskip("PIL")
    from PIL import Image

    img = Image.new("RGB", (32, 32), "white")
    batch = CaseBatch([
        FailureCase(id=f"i{i}", inputs=Inputs(prompt=f"Is there a dot {i}?", image=img),
                    expected="yes", observed="no", label=Label.FAIL,
                    metadata={"task": "yes_no"})
        for i in range(3)
    ])
    agent = FixAgent(judge=None, max_tier="L2", allow_codegen=False)
    assert agent._floor_names(has_images=True, tasks={"yes_no"}) == ()
    assert agent._floor_names(has_images=True, tasks={"multiple_choice"}) == ("self_consistency_5",)
    assert agent._floor_names(has_images=False, tasks={""}) == ("self_consistency_5",)
    del batch


# ── run_fix: refuted hypotheses never reach the proposer as verified ─────────


def test_run_fix_drops_m4_refuted_hypothesis_and_tells_the_proposer():
    from evalvitals.eval_agent import VLDiagnoseLoop, VLDiagnoseReport
    from evalvitals.eval_agent.stages.hypothesis_tester import HypothesisTestResult
    from evalvitals.eval_agent.stages.protocol import ExperimentProtocol
    from evalvitals.eval_agent.stages.surgery import InterventionResult

    h_bad = _hyp("labels come from a grading mismatch")
    h_bad.id = "h-bad"
    h_ok = _hyp("the model mis-binds the option letter")
    h_ok.id = "h-ok"

    def _tr(h):
        return HypothesisTestResult(hypothesis=h, status=HypothesisStatus.SUPPORTED,
                                    test_name="t", effect_size=0.5,
                                    is_consistent_with_protocol=True, confidence=0.9,
                                    verdict="signal vs FAIL effect=+0.9")

    report = VLDiagnoseReport(cycles=1, stopped_by="criteria_met",
                              verified_hypotheses=[_tr(h_bad), _tr(h_ok)],
                              final_hypotheses=[h_bad, h_ok])
    report.fix_proposal = InterventionResult(
        hypothesis=h_bad, status=HypothesisStatus.REFUTED, fixed=False,
        evidence={"verdict": 0.0, "alt_ordering_rescue_rate": 0.09},
    )

    class Recorder:
        run_logger = None

        def __init__(self):
            self.hypotheses = None
            self.context = None

        def propose_and_validate(self, model, data, hypotheses, prior_attempts=None, context=None):
            self.hypotheses = list(hypotheses)
            self.context = context
            return object()

    stub = Recorder()
    loop = VLDiagnoseLoop(model=CountingModel(), protocol=ExperimentProtocol(description="BBH"),
                          fix_agent=stub, confirm_split=0.5)
    loop.run_fix(report, _mc_batch(4, 4))
    assert [h.id for h in stub.hypotheses] == ["h-ok"]
    ctx = stub.context
    assert isinstance(ctx, FixContext)
    assert ctx.refuted and "grading mismatch" in ctx.refuted[0] and "verdict=0" in ctx.refuted[0]
    assert ctx.example_cases is not None            # the EXPLORE split, in full
    assert "mis-binds" in ctx.evidence and "effect=+0.9" in ctx.evidence
    assert ctx.task_note == "BBH"
    # the example cases are disjoint from the validation half
    _, confirm = loop._split_explore_confirm(_mc_batch(4, 4))
    assert {c.id for c in ctx.example_cases}.isdisjoint({c.id for c in confirm})


def test_run_fix_with_no_verified_hypothesis_does_not_call_minimal_agent():
    """Even a legacy minimal proposer cannot bypass the M5 evidence gate."""
    from evalvitals.eval_agent import VLDiagnoseLoop, VLDiagnoseReport
    from evalvitals.eval_agent.stages.protocol import ExperimentProtocol

    class Minimal:
        run_logger = None
        seen = None

        def propose_and_validate(self, model, data, hypotheses):
            self.seen = list(hypotheses)
            return object()

    stub = Minimal()
    loop = VLDiagnoseLoop(model=CountingModel(), protocol=ExperimentProtocol(description="d"),
                          fix_agent=stub)
    report = VLDiagnoseReport(cycles=1, stopped_by="max_cycles", final_hypotheses=[_hyp("x")])
    outcome = loop.run_fix(report, _mc_batch())
    assert stub.seen is None
    assert outcome.stage_status == "skipped"


# ── recommendation names a promising-but-inconclusive candidate ──────────────


def test_recommendation_surfaces_inconclusive_positive_candidate():
    from evalvitals.eval_agent.stages.fix_agent import FixValidation

    agent = FixAgent(judge=None, max_tier="L2", allow_codegen=False, floor_candidates=())
    good = FixValidation(candidate=FixCandidate(tier=FixTier.L2_SCAFFOLD, name="good",
                                                payload={}, kind="spec"),
                         n_pairs=80, n_fixed=11, n_broken=2, effect=0.1125, e_value=7.5,
                         reject=False, verdict="partial")
    batch = _mc_batch(11, 69)
    rec = agent._no_fix_recommendation([good], [FixTier.L1_PROMPT], batch, CountingModel())
    assert rec is not None and rec.get("promising", {}).get("candidate") == "good"
    assert "INCONCLUSIVE, not refuted" in rec["reason"]
    assert "baseline_repeats" in rec["reason"]


# ── noise model: per-case pass rates on both arms ─────────────────────────────


def test_bounded_mean_evalue_is_valid_and_directional():
    from evalvitals.stats import compare_paired_rates, evalue_bounded_mean

    assert evalue_bounded_mean([]) == 1.0
    assert evalue_bounded_mean([0.0] * 50) == 1.0
    up = evalue_bounded_mean([1.0] * 11)
    assert up > 20  # 11 clean repairs, nothing broken, clears 1/alpha
    assert evalue_bounded_mean([-1.0] * 11) < 1.0  # wrong direction: no evidence
    # a fractional repair is worth less than a full one
    assert evalue_bounded_mean([0.4] * 11) < up
    # sampling noise (symmetric +/-1) accumulates no evidence in expectation:
    # empirical false-rejection rate well under alpha
    import random

    rng = random.Random(0)
    rejects = sum(
        evalue_bounded_mean([rng.choice([-1.0, 1.0]) for _ in range(60)]) >= 20
        for _ in range(500)
    )
    assert rejects <= 5  # <= 1% observed at alpha 5%

    r = compare_paired_rates([0.4] * 10 + [0.9] * 70, [1.0] * 10 + [0.9] * 70)
    assert r.effect == pytest.approx(0.075)
    assert r.e_value > 1.0 and r.details["e_value_regression"] < 1.0
    assert r.details["n_positive"] == 10 and r.details["n_negative"] == 0


class _CoinModel(CountingModel):
    """Baseline is a coin per case (deterministic script); 'carefully' is always right."""

    def __init__(self, script):
        super().__init__()
        self._script = list(script)  # per baseline call: True -> right answer
        self._i = 0

    def generate(self, inputs, **kwargs):
        self.calls.append(dict(kwargs))
        p = str(getattr(inputs, "prompt", inputs))
        if "carefully" in p.lower():
            return "Chain.\nAnswer: (B)"
        self._i += 1
        ok = self._script[(self._i - 1) % len(self._script)]
        return "Chain.\nAnswer: (B)" if ok else "Chain.\nAnswer: (C)"


def test_paired_rates_uses_frozen_sample_plus_fresh_and_weighs_unstable():
    """baseline_repeats=k: the frozen observed output is sample 1, k-1 are
    generated; each case's baseline is a rate; unstable cases stay in."""
    batch = _mc_batch(4, 4)  # 4 FAIL (observed wrong) + 4 PASS (observed right)
    # fresh baseline samples: alternate right/wrong -> every case ends unstable
    model = _CoinModel([True, False])
    agent = FixAgent(judge=None, max_tier="L1", baseline_repeats=3, floor_candidates=())
    baseline, unstable = agent._baseline(model, batch)
    assert len(model.calls) == 2 * len(batch)  # k-1 fresh per case; frozen counted
    assert all(0.0 < agent._baseline_rates[c.id] < 1.0 for c in batch)
    assert len(unstable) == len(batch)
    cand = FixCandidate(tier=FixTier.L1_PROMPT, name="careful", kind="template",
                        payload={"prompt_template": "Carefully. {prompt}"})
    v = agent._validate(cand, model, batch, baseline, unstable)
    assert v.noise_model == "paired_rates"
    assert v.n_pairs == len(batch) and v.n_unstable == len(batch)  # weighed, not dropped
    assert v.candidate_rate == 1.0 and 0.0 < v.baseline_rate < 1.0
    assert v.effect == pytest.approx(1.0 - v.baseline_rate)
    assert v.n_broken == 0


def test_paired_rates_certifies_a_real_fix_and_not_noise():
    """A candidate that lifts every unstable case to certainty is certified;
    an identity-like candidate that is just as flaky is not."""
    batch = _mc_batch(12, 12)
    real = FixAgent(judge=ScriptedJudge(json.dumps([
        {"name": "careful", "prompt_template": "Carefully. {prompt}"}])),
        max_tier="L1", baseline_repeats=5, floor_candidates=())
    out = real.propose_and_validate(_CoinModel([True, False, False]), batch, [_hyp("h")])
    v = out.attempted[0]
    assert v.noise_model == "paired_rates" and v.n_baseline_samples == 5
    assert v.fixed is True and out.fixed is True

    class Flaky(_CoinModel):
        def generate(self, inputs, **kwargs):
            # the "fix" prompt is exactly as flaky as the baseline
            self.calls.append(dict(kwargs))
            self._i += 1
            ok = self._script[(self._i - 1) % len(self._script)]
            return "Chain.\nAnswer: (B)" if ok else "Chain.\nAnswer: (C)"

    noise = FixAgent(judge=ScriptedJudge(json.dumps([
        {"name": "same", "prompt_template": "Rephrased. {prompt}"}])),
        max_tier="L1", baseline_repeats=5, floor_candidates=())
    out2 = noise.propose_and_validate(Flaky([True, False, False]), batch, [_hyp("h")])
    assert out2.fixed is False
    assert out2.attempted[0].verdict in {"partial", "unsafe", "no_effect"}


def test_candidate_repeats_apply_to_declarative_candidates_only(tmp_path):
    batch = _mc_batch(2, 2)
    agent = FixAgent(judge=None, max_tier="L2", candidate_repeats=3, floor_candidates=())
    model = CountingModel()
    baseline, unstable = agent._baseline(model, batch)
    tmpl = FixCandidate(tier=FixTier.L1_PROMPT, name="t", kind="template",
                        payload={"prompt_template": "Carefully. {prompt}"})
    v = agent._validate(tmpl, model, batch, baseline, unstable)
    assert v.n_candidate_samples == 3 and len(model.calls) == 3 * len(batch)
    assert v.noise_model == "paired_rates"
    code = FixCandidate(tier=FixTier.L2_SCAFFOLD, name="coded_pipeline", kind="code",
                        payload={"code": (
                            'import json\ncases=json.load(open("fix_cases.json"))["cases"]\n'
                            'out=[{"sample_id":c["id"],"output":model_generate(c["id"])} for c in cases]\n'
                            'print("FIX_PIPELINE_RESULT_JSON="+json.dumps({"per_case":out}))\n')})
    agent._sandbox = None
    agent._run_context = None
    import evalvitals.eval_agent.stages.fix_agent as fa
    from evalvitals.agent_runtime.sandbox import ExperimentSandbox

    agent._sandbox = ExperimentSandbox(workdir=tmp_path / "sb", cleanup=False)
    v2 = agent._validate(code, model, batch, baseline, unstable)
    assert v2.n_candidate_samples == 1  # coded pipelines never repeat
    del fa


def test_rates_mode_power_ceiling_uses_baseline_rates():
    from evalvitals.eval_agent.stages.fix_agent import FixValidation

    agent = FixAgent(judge=None, max_tier="L2", baseline_repeats=5, allow_codegen=False,
                     floor_candidates=())
    batch = _mc_batch(2, 40)
    agent._baseline_rates = {c.id: (0.9 if c.label == Label.PASS else 0.2) for c in batch}
    weak = FixValidation(candidate=FixCandidate(tier=FixTier.L1_PROMPT, name="w", payload={},
                                                kind="template"),
                         n_pairs=42, n_fixed=2, n_broken=0, effect=0.02, e_value=1.5,
                         reject=False, verdict="partial")
    rec = agent._no_fix_recommendation([weak], [FixTier.L1_PROMPT], batch, CountingModel())
    assert rec is not None  # ceiling with 42 cases of headroom is high -> not 'underpowered'
    tiny = _mc_batch(1, 1)
    agent._baseline_rates = {c.id: 0.9 for c in tiny}
    rec2 = agent._no_fix_recommendation([weak], [FixTier.L1_PROMPT], tiny, CountingModel())
    assert rec2 is not None and rec2.get("action") == "gather_more_failures"


# ── a template with literal braces must not take the stage down ─────────────


def test_spec_template_with_literal_braces_renders_and_never_aborts_the_stage():
    """qwen3.5-2b/bbh_tracking7 run8: the judge's L2 template contained a
    literal "{1,2,3,4,5,6,7}"; str.format raised KeyError outside the per-case
    try and the whole fix stage (2 h in) died. The spec path must render with
    safe_format like the L1 path, and a per-case exception of any kind must
    score None for that case, not propagate."""
    spec = PipelineSpec(name="braces",
                        prompt_template="{prompt}\nUse the set {1,2,3,4,5,6,7} and \\frac{a}{b}.")
    model = CountingModel()
    case = list(_mc_batch(1, 0))[0]
    capture: dict = {}
    assert run_pipeline(model, case, spec, _mc_score, capture=capture) is True
    assert "{1,2,3,4,5,6,7}" in capture["prompt"] and "\\frac{a}{b}" in capture["prompt"]

    class Exploding(CountingModel):
        def generate(self, inputs, **kwargs):
            if "q1" in str(getattr(inputs, "prompt", "")):
                raise RuntimeError("adapter hiccup")
            return super().generate(inputs, **kwargs)

    agent = FixAgent(judge=None, max_tier="L1", concurrency=3, floor_candidates=())

    def raising_strategy(model, case):
        if case.id == "q1":
            raise KeyError("boom")
        return True

    agent._strategy = lambda candidate: raising_strategy  # type: ignore[assignment]
    cand = FixCandidate(tier=FixTier.L1_PROMPT, name="t", kind="template",
                        payload={"prompt_template": "x {prompt}"})
    scores = agent._candidate_scores(cand, Exploding(), _mc_batch(2, 1))
    assert scores["q1"] is None and scores["q0"] is True and scores["q2"] is True


# ── no verified hypothesis: M4 + fix on the best unverified lead (opt-in) ────


def _inconclusive_report():
    from evalvitals.eval_agent import VLDiagnoseReport
    from evalvitals.eval_agent.stages.hypothesis_tester import HypothesisTestResult

    hs = [_hyp("lead A (weak)"), _hyp("lead B (best)"), _hyp("lead C (refuted)")]
    for i, h in enumerate(hs):
        h.id = f"h{i}"
    trs = [
        HypothesisTestResult(hypothesis=hs[0], status=HypothesisStatus.INCONCLUSIVE, test_name="t",
                             effect_size=0.05, is_consistent_with_protocol=True, confidence=0.1,
                             verdict="weak"),
        HypothesisTestResult(hypothesis=hs[1], status=HypothesisStatus.INCONCLUSIVE, test_name="t",
                             effect_size=0.2, is_consistent_with_protocol=True, confidence=0.4,
                             verdict="best"),
        HypothesisTestResult(hypothesis=hs[2], status=HypothesisStatus.REFUTED, test_name="t",
                             effect_size=-0.3, is_consistent_with_protocol=True, confidence=0.5,
                             verdict="refuted"),
    ]
    return VLDiagnoseReport(cycles=1, stopped_by="max_cycles", verified_hypotheses=[],
                            all_test_results=trs, final_hypotheses=hs), hs


def test_run_m4_default_still_requires_verified_but_allow_unverified_uses_best_lead():
    from evalvitals.eval_agent import VLDiagnoseLoop
    from evalvitals.eval_agent.stages.protocol import ExperimentProtocol

    report, hs = _inconclusive_report()
    loop = VLDiagnoseLoop(model=CountingModel(), protocol=ExperimentProtocol(description="d"))
    assert loop.run_m4(report, _mc_batch()) is None                       # unchanged default
    iv = loop.run_m4(report, _mc_batch(), allow_unverified=True)
    assert iv is not None and iv.hypothesis is hs[1]                     # best non-refuted lead
    assert iv.evidence.get("hypothesis_was_verified") is False
    assert report.fix_proposal is iv


def test_run_fix_without_verified_skips_repair_authoring():
    from evalvitals.eval_agent import VLDiagnoseLoop
    from evalvitals.eval_agent.stages.protocol import ExperimentProtocol

    report, hs = _inconclusive_report()

    class Recorder:
        run_logger = None
        hypotheses = None
        context = None

        def propose_and_validate(self, model, data, hypotheses, prior_attempts=None, context=None):
            self.hypotheses = list(hypotheses)
            self.context = context
            return object()

    stub = Recorder()
    loop = VLDiagnoseLoop(model=CountingModel(), protocol=ExperimentProtocol(description="d"),
                          fix_agent=stub)
    outcome = loop.run_fix(report, _mc_batch())
    assert stub.hypotheses is None
    assert outcome.stage_status == "skipped"
    assert outcome.skip_reason == "no_accepted_hypothesis"


# ── coded-pipeline bridge: concurrent, request-id tagged ─────────────────────


class EchoPromptModel(Model):
    """Replies with the prompt it was asked, after a short sleep: a threaded
    pipeline whose replies were mis-routed would get another case's text."""

    capabilities = frozenset({Capability.GENERATE})
    modalities = frozenset({"text"})

    def __init__(self, delay: float = 0.15) -> None:
        self.delay = delay
        self.n_truncated = 0
        self._lock = __import__("threading").Lock()
        self.max_in_flight = 0
        self._in_flight = 0

    def generate(self, inputs, **kwargs):
        import time as _t
        with self._lock:
            self._in_flight += 1
            self.max_in_flight = max(self.max_in_flight, self._in_flight)
        _t.sleep(self.delay)
        with self._lock:
            self._in_flight -= 1
        return "echo:" + str(inputs.prompt)

    def forward(self, inputs, capture, spec=None):
        raise NotImplementedError


_THREADED_PIPELINE = """
import json
from concurrent.futures import ThreadPoolExecutor
cases = json.load(open("fix_cases.json"))["cases"]
def solve(c):
    outs = [model_generate(c["id"], prompt="P-%s-%d" % (c["id"], k)) for k in range(3)]
    return {"sample_id": c["id"], "output": "|".join(outs)}
with ThreadPoolExecutor(max_workers=8) as ex:
    res = list(ex.map(solve, cases))
print("FIX_PIPELINE_RESULT_JSON=" + json.dumps({"per_case": res}))
"""


@pytest.mark.parametrize("concurrency", [1, 6])
def test_threaded_coded_pipeline_gets_its_own_replies(tmp_path, concurrency):
    """Eight sandbox threads in flight: every reply must land in the thread
    that asked for it (old bridge: whichever thread read stdin next got it),
    and with concurrency>1 the host really services calls in parallel."""
    import time as _t

    from evalvitals.eval_agent.stages.fix_pipeline import run_coded_pipeline

    model = EchoPromptModel(delay=0.15)
    batch = _mc_batch(n_fail=6, n_pass=6)
    t0 = _t.monotonic()
    result = run_coded_pipeline(_THREADED_PIPELINE, model, batch, workdir=tmp_path,
                                timeout_sec=60, concurrency=concurrency)
    elapsed = _t.monotonic() - t0
    assert result.ok, result.error
    assert result.n_calls == 36
    for case in batch:
        assert result.outputs[case.id] == "|".join(
            f"echo:P-{case.id}-{k}" for k in range(3)), result.outputs[case.id]
    if concurrency > 1:
        assert model.max_in_flight > 1
        assert elapsed < 36 * 0.15            # parallel: well under the serial sum
    else:
        assert model.max_in_flight == 1       # serial host: arrival order


def test_bridge_reply_carries_rid_and_errors_route_to_the_caller(tmp_path):
    """An unknown case id errors in the calling thread only; the other
    thread's call still succeeds (replies are routed by id, not by order)."""
    from evalvitals.eval_agent.stages.fix_pipeline import run_coded_pipeline

    code = """
import json
from concurrent.futures import ThreadPoolExecutor
cases = json.load(open("fix_cases.json"))["cases"]
def bad(_):
    try:
        model_generate("no-such-case", prompt="x")
        return "no error"
    except RuntimeError as e:
        return "err:" + str(e)
def good(c):
    return model_generate(c["id"], prompt="ok-" + c["id"])
with ThreadPoolExecutor(max_workers=2) as ex:
    fb = ex.submit(bad, None); fg = ex.submit(good, cases[0])
    b, g = fb.result(), fg.result()
print("FIX_PIPELINE_RESULT_JSON=" + json.dumps({"per_case": [
    {"sample_id": cases[0]["id"], "output": g + "//" + b}]}))
"""
    batch = _mc_batch(n_fail=1, n_pass=0)
    result = run_coded_pipeline(code, EchoPromptModel(delay=0.05), batch,
                                workdir=tmp_path, timeout_sec=30, concurrency=4)
    assert result.ok, result.error
    out = result.outputs[batch[0].id]
    assert out.startswith(f"echo:ok-{batch[0].id}//err:")
    assert "unknown case_id" in out


def test_fix_agent_passes_concurrency_to_the_coded_bridge(monkeypatch):
    from evalvitals.eval_agent.stages import fix_agent as fa
    from evalvitals.eval_agent.stages.fix_pipeline import CodedPipelineResult

    seen = {}

    def fake_run(code, model, data, **kw):
        seen.update(kw)
        return CodedPipelineResult(outputs={c.id: "Answer: (B)" for c in data}, ok=True)

    monkeypatch.setattr(fa, "run_coded_pipeline", fake_run)
    agent = FixAgent(judge=ScriptedJudge("[]"), max_tier="L2", concurrency=5)
    cand = fa.FixCandidate(tier=FixTier.L2_SCAFFOLD, name="coded", payload={"code": "x"})
    agent._run_coded(cand, CountingModel(), _mc_batch())
    assert seen["concurrency"] == 5


# ── allow_unverified=True: the exploratory fix path, opt-in ──────────────────
#
# Without an M5-verified hypothesis run_fix records a skipped stage (above).
# The examples we run pass allow_unverified=True so the fix still executes on
# the best unverified leads — the candidate validation on CONFIRM is the gate.


def test_run_fix_allow_unverified_keeps_a_minimal_fix_agent_working():
    """A stub that only accepts (model, data, hypotheses) keeps working and
    receives the final proposals when nothing was verified."""
    from evalvitals.eval_agent import VLDiagnoseLoop, VLDiagnoseReport
    from evalvitals.eval_agent.stages.protocol import ExperimentProtocol

    class Minimal:
        run_logger = None
        seen = None

        def propose_and_validate(self, model, data, hypotheses):
            self.seen = list(hypotheses)
            return object()

    stub = Minimal()
    loop = VLDiagnoseLoop(model=CountingModel(), protocol=ExperimentProtocol(description="d"),
                          fix_agent=stub)
    report = VLDiagnoseReport(cycles=1, stopped_by="max_cycles", final_hypotheses=[_hyp("x")])
    loop.run_fix(report, _mc_batch(), allow_unverified=True)
    assert stub.seen and stub.seen[0].statement == "x"


def test_run_fix_allow_unverified_uses_unverified_leads_and_says_so():
    from evalvitals.eval_agent import VLDiagnoseLoop
    from evalvitals.eval_agent.stages.protocol import ExperimentProtocol

    report, hs = _inconclusive_report()

    class Recorder:
        run_logger = None
        hypotheses = None
        context = None

        def propose_and_validate(self, model, data, hypotheses, prior_attempts=None, context=None):
            self.hypotheses = list(hypotheses)
            self.context = context
            return object()

    stub = Recorder()
    loop = VLDiagnoseLoop(model=CountingModel(), protocol=ExperimentProtocol(description="d"),
                          fix_agent=stub)
    outcome = loop.run_fix(report, _mc_batch(), allow_unverified=True)
    assert getattr(outcome, "stage_status", "completed") != "skipped"
    assert [h.id for h in stub.hypotheses] == ["h1", "h0"]               # best first, refuted dropped
    assert stub.context.hypotheses_note.startswith("UNVERIFIED")
    # the proposer sees the caveat right under the hypotheses heading
    judge = ScriptedJudge("[]")
    agent = FixAgent(judge=judge, max_tier="L1")
    agent.propose_and_validate(CountingModel(), _mc_batch(), stub.hypotheses, context=stub.context)
    assert "UNVERIFIED: M5 found no statistically significant evidence" in judge.prompts[-1]
