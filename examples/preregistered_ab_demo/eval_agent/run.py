"""Eval agent — pre-registered A/B comparison with selective-inference discipline.

Demonstrates the EvalOrchestrator: a closed-loop eval workflow that enforces
selective-inference safety — mine on `explore`, pre-register a falsifiable
hypothesis, test ONCE on `validate`, lock `confirm` for the final report.

This prevents the textbook mistake of mining data for a pattern and then testing
that same pattern on the same data (inflated false discovery rate).

Usage (inside Docker):
    python run.py               # synthetic demo + M1→M4 loop (requires GEMINI_API_KEY)
    python run.py --n-cases 120

Expected output:
    Pre-registration hash: a3f7c2...
    [EvalOrchestrator] strategy B better (effect=+0.13), reject=True
    ...
    [AutoDiagnoseLoop] cycles=1  resolved=False
      severity : medium
      narrative: The model shows moderate self-consistency ...
      hypothesis  : The model exhibits inconsistent responses ...

Also demonstrates:
  CounterfactualReplay — identifies which agent steps most influence the outcome.
  AutoDiagnoseLoop     — M1 probe → M2 analysis → M3 Gemini diagnosis → M4 surgery.
"""

from __future__ import annotations

import argparse
import os
import random
from pathlib import Path

import yaml

from evalvitals.analyzers.agent.counterfactual import CounterfactualReplay
from evalvitals.core.case import CaseBatch, FailureCase, Inputs, Label, Step, Trajectory
from evalvitals.eval_agent import (
    DataSplit,
    EvalOrchestrator,
    PreregisteredHypothesis,
    RunContext,
)

CONFIG = Path(__file__).parent / "config.yaml"


def _make_synthetic_cases(n: int, seed: int = 42) -> list:
    rng = random.Random(seed)
    return [
        FailureCase(
            id=f"case_{i:04d}",
            inputs=Inputs(prompt=f"Multi-step question #{i}: solve step by step."),
            label=Label.PASS if rng.random() < 0.6 else Label.FAIL,
        )
        for i in range(n)
    ]


def strategy_a(case: FailureCase) -> bool:
    """Baseline strategy — direct answer, ~65% success."""
    return random.random() < 0.65


def strategy_b(case: FailureCase) -> bool:
    """Chain-of-thought prompt — ~78% success on multi-step questions."""
    return random.random() < 0.78


def demo_orchestrator(cfg: dict) -> None:
    n = cfg.get("n_cases", 60)
    alpha = cfg.get("alpha", 0.05)
    min_effect = cfg.get("min_effect", 0.03)

    cases = _make_synthetic_cases(n)

    hyp = PreregisteredHypothesis(
        predicate="multi-step questions",
        statement="chain-of-thought prompt improves accuracy on multi-step questions",
        direction="B>A",
        min_effect=min_effect,
        alpha=alpha,
        split="validate",
    )

    orch = EvalOrchestrator(split=DataSplit(explore_frac=0.5, validate_frac=0.3))
    report = orch.run(cases, hyp, strategy_a, strategy_b)

    print(f"Pre-registration hash: {report['prereg_hash']}")
    print(f"[EvalOrchestrator] decision={report['decision']}")
    print(f"Hypothesis: {hyp.statement!r}")
    print(f"Split tested: {report['split']}")
    e = report["e_value"]
    e_str = f"{e:.1f}" if e is not None else "N/A"
    print(f"Effect: {report['effect']:+.3f}  CI: {report['ci']}  "
          f"e-value: {e_str}  reject: {report['decision'] == 'REJECT H0'}")


def demo_counterfactual() -> None:
    """Show CounterfactualReplay ranking steps by causal influence."""

    def rerun_fn(trajectory, step_idx, seed):
        # Synthetic: step 1 is always causal (flips outcome when replayed)
        rng = random.Random(seed + step_idx)
        return step_idx == 1 and rng.random() < 0.8

    traj = Trajectory(
        sample_id="demo",
        steps=[
            Step(idx=0, tool_call={"name": "search"}, observation="found docs"),
            Step(idx=1, tool_call={"name": "extract_answer"}, observation="extracted"),
            Step(idx=2, tool_call={"name": "format_output"}, observation="formatted"),
        ],
        outcome=Label.PASS,
    )
    case = FailureCase(id="cf_demo", inputs=Inputs(prompt="solve this task"), trajectory=traj)

    analyzer = CounterfactualReplay(rerun_fn=rerun_fn, n_replays=10)
    result = analyzer._run(None, CaseBatch([case]))
    pc = result.findings["per_case"][0]
    print(f"\n[Counterfactual] most influential step: {pc['most_influential_step']}")
    print(f"  all steps: {pc['steps']}")


def demo_auto_diagnose() -> None:
    """Run the M1→M4 AutoDiagnoseLoop with Gemini as both subject model and judge.

    M1 probes the model with self_consistency + verbalized_confidence.
    M2 summarises the findings into a structured report.
    M3 DiagnosisAgent (Gemini) reads the report and proposes hypotheses.
    M4 SurgeryAgent attempts to verify each hypothesis.

    Requires GEMINI_API_KEY in the environment.
    """
    import evalvitals  # noqa: F401 — side-effect: registers all analyzers

    from evalvitals.eval_agent import AutoDiagnoseLoop, DiagnosisAgent, SurgeryAgent
    from evalvitals.eval_agent.stages.probe import ModelKind, StrategyProbe
    from evalvitals.eval_agent.stages.probe_agent import ProbeAgent
    from evalvitals.models.blackbox.gemini import GeminiModel

    model = GeminiModel()

    cases = CaseBatch([
        FailureCase(
            id="fc_0",
            inputs=Inputs(prompt="What is 17 multiplied by 19?"),
            label=Label.FAIL,
        ),
        FailureCase(
            id="fc_1",
            inputs=Inputs(prompt="Name the longest river in Africa."),
            label=Label.FAIL,
        ),
        FailureCase(
            id="fc_2",
            inputs=Inputs(prompt="Is Paris the capital of Germany?"),
            label=Label.FAIL,
        ),
    ])

    # Run only GENERATE-compatible analyzers that work on plain text cases.
    text_probe = StrategyProbe(priority_override={
        kind: ["self_consistency", "verbalized_confidence"]
        for kind in ModelKind
    })

    # Rooted under outputs/ (mounted to the host by docker-compose.yml) so the
    # run survives container teardown, like every other example's RunContext.
    ctx = RunContext(Path(__file__).parent / "outputs", verbose=True)
    print(f"  logging to: {ctx.root}")

    # SurgeryAgent uses the same Gemini model as judge to write + execute
    # targeted diagnostic scripts via ExperimentWriter (M4 code-execution path).
    surgery = SurgeryAgent(judge=model, run_context=ctx)

    loop = AutoDiagnoseLoop(
        model=model,
        probe_agent=ProbeAgent(probe=text_probe, max_analyzers=2),
        diagnosis_agent=DiagnosisAgent(),
        surgery_agent=surgery,
        max_cycles=2,
        run_logger=ctx.logger,
    )
    report = loop.run(cases)

    print(f"\n[AutoDiagnoseLoop] cycles={report.cycles}  resolved={report.resolved}")
    if report.final_analysis:
        print(f"  severity : {report.final_analysis.severity}")
        print(f"  narrative: {report.final_analysis.narrative[:300]}")
    for h in report.final_hypotheses:
        print(f"  hypothesis  : {h.statement}")
        print(f"    mode      : {h.predicted_failure_mode}  status: {h.status}")

    ctx.finalize()


def demo_cli_agent() -> None:
    """M4 using Claude Code (or another CLI agent) as the experiment writer.

    Instead of the single-pass LLM path, SurgeryAgent routes through an
    agentic CLI tool that has bash/file access and self-repairs.

    Requires:
        - ``claude`` binary on PATH and ANTHROPIC_API_KEY set  (claude_code)
        - OR set EVALVITALS_CLI_PROVIDER to another provider name
          (codex / opencode / gemini_cli / kimi_cli) with the corresponding binary.

    Enable with: EVALVITALS_CLI_DEMO=1 python run.py
    """
    import evalvitals  # noqa: F401 — registers all analyzers

    from evalvitals.eval_agent import (
        AutoDiagnoseLoop,
        CliAgentConfig,
        DiagnosisAgent,
        ExperimentWriterConfig,
        SurgeryAgent,
    )
    from evalvitals.eval_agent.stages.probe import ModelKind, StrategyProbe
    from evalvitals.eval_agent.stages.probe_agent import ProbeAgent
    from evalvitals.models.blackbox.gemini import GeminiModel

    provider = os.getenv("EVALVITALS_CLI_PROVIDER", "claude_code")
    model = GeminiModel()

    cases = CaseBatch([
        FailureCase(id="cli_0", inputs=Inputs(prompt="What is 17 × 19?"), label=Label.FAIL),
        FailureCase(id="cli_1", inputs=Inputs(prompt="Longest river in Africa?"), label=Label.FAIL),
    ])

    writer_cfg = ExperimentWriterConfig(
        cli_agent=CliAgentConfig(
            provider=provider,
            model="sonnet" if provider == "claude_code" else "",
            max_budget_usd=1.0,
            timeout_sec=120,
        ),
        exec_fix_timeout_sec=30,
    )

    text_probe = StrategyProbe(priority_override={
        kind: ["self_consistency", "verbalized_confidence"]
        for kind in ModelKind
    })

    surgery = SurgeryAgent(judge=model, writer_config=writer_cfg)
    loop = AutoDiagnoseLoop(
        model=model,
        probe_agent=ProbeAgent(probe=text_probe, max_analyzers=2),
        diagnosis_agent=DiagnosisAgent(),
        surgery_agent=surgery,
        max_cycles=1,
    )
    report = loop.run(cases)

    print(f"\n[CLI Agent Demo — {provider}] cycles={report.cycles}")
    for h in report.final_hypotheses:
        print(f"  hypothesis: {h.statement}")
        print(f"    status  : {h.status}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(CONFIG))
    parser.add_argument("--n-cases", type=int, default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if args.n_cases:
        cfg["n_cases"] = args.n_cases

    print("=== EvalOrchestrator + pre-registered hypothesis ===")
    demo_orchestrator(cfg)

    print("\n=== CounterfactualReplay (causal step attribution) ===")
    demo_counterfactual()

    print("\n=== AutoDiagnoseLoop M1→M4 (requires GEMINI_API_KEY) ===")
    demo_auto_diagnose()

    if os.getenv("EVALVITALS_CLI_DEMO"):
        provider = os.getenv("EVALVITALS_CLI_PROVIDER", "claude_code")
        print(f"\n=== CLI Agent Demo ({provider}) ===")
        demo_cli_agent()


if __name__ == "__main__":
    main()
