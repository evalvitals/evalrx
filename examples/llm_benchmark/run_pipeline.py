"""M1 -> M2 -> M3 -> M5 -> M4 over a frozen CaseBatch from build_cases.py.

    python run_pipeline.py --model qwen3.5-9b --dataset supergpqa_law
    python run_pipeline.py --analysis-only        # M1->M2->M3, stop before M5/M4
    python -m evalvitals.cli dashboard outputs/qwen3.5-9b/supergpqa_law

Stages (evalvitals.eval_agent.loop.VLDiagnoseLoop -- the class name says VL, but
it takes a plain Model plus an ExperimentProtocol whose target_modalities is
{"text"} here, and nothing in it is vision-specific):

    M1 ProbeAgent          selects and runs analyzers against the batch
    M2 StatsAnalysisAgent  protocol-aware statistics over M1's per-case signals
    M3 DiagnosisAgent      proposes hypotheses from the stats
    M5 HypothesisTester    tests each hypothesis + checks protocol consistency
    M4 SurgeryAgent        proposes a fix for the best VERIFIED hypothesis

M4 runs OUTSIDE the loop, on the held-out confirm split (config `confirm_split`),
so the repair is validated on cases the loop never mined for its hypothesis. With
confirm_split at 0 the fix is scored on the same data that produced the
hypothesis, which is how a diagnosis loop flatters itself.

The judge/coder are `claude -p` CLI calls and are NOT the model under test. The
model under test is only loaded for generation: already done in build_cases.py,
and again by M4 if the fix needs to be executed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "llm_band_probe"))
sys.path.insert(0, str(HERE.parent.parent))

import band_locate as B  # noqa: E402
import datasets as CATALOG  # noqa: E402

CFG = yaml.safe_load((HERE / "config.yaml").read_text())


# ---------------------------------------------------------------- model
class EndpointModel:
    """The model under test, over an OpenAI-compatible endpoint.

    Deliberately thin: M1/M4 call ``generate`` and nothing else. Sampling is
    pinned to the Qwen thinking recipe because greedy decoding sends these models
    into verbatim self-verification loops that never terminate.
    """

    def __init__(self, model_id: str, base_url: str, max_tokens: int, sampling: dict):
        from evalvitals.core.capability import Capability

        self.capabilities = frozenset({Capability.GENERATE})
        self.modalities = frozenset({"text"})
        self.model_id = model_id
        self.base_url = base_url
        self.max_tokens = max_tokens
        self.sampling = sampling
        self.n_calls = 0
        self.n_truncated = 0

    def generate(self, inputs, **kwargs) -> str:
        B.MODEL_ID, B.BASE_URL = self.model_id, self.base_url
        self.n_calls += 1
        text, reason = B.generate(
            str(getattr(inputs, "prompt", inputs)),
            kwargs.get("max_tokens", self.max_tokens),
            kwargs.get("temperature", self.sampling["temperature"]),
            sampling={k: v for k, v in self.sampling.items() if k != "temperature"},
        )
        if reason == "length":
            self.n_truncated += 1
        return text

    def logprobs(self, inputs, **kwargs):  # pragma: no cover
        raise NotImplementedError("endpoint exposes no logprobs")

    def forward(self, inputs, capture, spec=None):  # pragma: no cover
        raise NotImplementedError("endpoint exposes no internals")

    def __repr__(self) -> str:
        return f"EndpointModel({self.model_id})"


# ---------------------------------------------------------------- inputs
def load_batch(model_id: str, dataset: str):
    from evalvitals.core.case import CaseBatch, FailureCase, Inputs, Label

    path = HERE / "outputs" / model_id / dataset / "cases.json"
    if not path.exists():
        raise SystemExit(
            f"{path} not found — run:\n"
            f"  python build_cases.py --model {model_id} --dataset {dataset}"
        )
    report = json.loads(path.read_text())
    cases = [
        FailureCase(
            inputs=Inputs(prompt=c["prompt"]),
            observed=c["output"],
            expected=c["gold"] if not isinstance(c["gold"], list) else c["gold"][0],
            label=Label.PASS if c["label"] == "PASS" else Label.FAIL,
        )
        for c in report["cases"]
    ]
    return CaseBatch(cases), report


def build_protocol(dataset: str):
    """The human prior handed to M1/M2/M5.

    States the OBSERVATION and the grading rule only. It must NOT name a
    suspected mechanism -- proposing the mechanism is M3's job, and supplying one
    here leaks the answer into the loop that is meant to find it.
    """
    from evalvitals.eval_agent.stages.protocol import ExperimentProtocol

    entry = CATALOG.get(dataset)
    return ExperimentProtocol(
        description=(
            f"A text-only LLM answers items from {entry.source}. "
            f"{entry.slicing} Failure cases are items the model answered "
            f"incorrectly; the batch also contains, as controls, items from the "
            f"same slice it answered correctly. Each answer is scored against its "
            f"own gold answer."
        ),
        task_domain=entry.chapter,
        success_criteria=entry.grading,
        failure_patterns="",
        target_modalities=frozenset({"text"}),
        metadata={
            "dataset": entry.name,
            "items_in_slice": entry.items,
            "reference_accuracy_qwen35_9b": entry.accuracy_9b,
        },
    )


def build_judge(model_name: str, effort: str):
    from evalvitals.eval_agent import ClaudeModel

    judge = ClaudeModel(model=model_name, effort=effort)
    if not judge.generate("Reply with exactly the word OK").strip():
        raise SystemExit(
            f"judge probe: claude --model {model_name} returned empty "
            f"(rate-limited?) — try --judge-model sonnet, or a lower --judge-effort"
        )
    print(f"judge: claude model={model_name} effort={effort or 'default'}")
    return judge


def build_codegen(backend: str):
    from evalvitals.eval_agent import CliAgentConfig

    provider = {"claude": "claude_code", "codex": "codex", "agy": "antigravity"}.get(
        backend, backend)
    is_claude = provider == "claude_code"
    effort = str(CFG.get("codegen_effort", "") or "")
    return CliAgentConfig(
        provider=provider,
        model=str(CFG.get("codegen_model", "claude-opus-5")) if is_claude else "",
        max_budget_usd=float(CFG.get("codegen_budget_usd", 2.0)),
        timeout_sec=int(CFG.get("codegen_timeout_sec", 480)),
        extra_args=(("--effort", effort) if (effort and is_claude) else ()),
    )


# ---------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=CFG["model"])
    ap.add_argument("--base-url", default=CFG["base_url"])
    ap.add_argument("--dataset", default=CFG["dataset"])
    ap.add_argument("--judge-model", default=CFG["judge_model"])
    ap.add_argument("--judge-effort", default=CFG["judge_effort"])
    ap.add_argument("--backend", default="claude", choices=["claude", "codex", "agy"])
    ap.add_argument("--max-cycles", type=int, default=CFG["max_cycles"])
    ap.add_argument("--confirm-split", type=float, default=CFG["confirm_split"])
    ap.add_argument("--analysis-only", action="store_true",
                    help="M1->M2->M3 and stop: propose hypotheses, skip M5 and M4")
    ap.add_argument("--skip-m4", action="store_true",
                    help="run M1->M5 but do not attempt a fix")
    args = ap.parse_args()

    from evalvitals.analysis.stats_agent import StatsAnalysisAgent
    from evalvitals.eval_agent import (
        ExperimentWriterConfig,
        FixAgent,
        RunLogger,
        SurgeryAgent,
        VLDiagnoseLoop,
    )
    from evalvitals.eval_agent.stages.diagnosis import DiagnosisAgent
    from evalvitals.eval_agent.stages.hypothesis_tester import HypothesisTester
    from evalvitals.eval_agent.stages.probe_agent import ProbeAgent

    batch, report_in = load_batch(args.model, args.dataset)
    out = HERE / "outputs" / args.model / args.dataset
    out.mkdir(parents=True, exist_ok=True)

    print(f"[batch] {args.dataset} n={report_in['n']} "
          f"PASS={report_in['n_pass']} FAIL={report_in['n_fail']} "
          f"acc={report_in['accuracy']:.3f}")

    model = EndpointModel(args.model, args.base_url, CFG["max_tokens"],
                          {"temperature": float(CFG["temperature"]),
                           "top_p": float(CFG["top_p"]), "top_k": int(CFG["top_k"])})
    judge = build_judge(args.judge_model, args.judge_effort)
    codegen = build_codegen(args.backend)
    logger = RunLogger(run_dir=out / "logs", verbose=True)

    # Every stage takes its judge/coder through its CONSTRUCTOR. Assigning
    # `stage.judge` afterwards would leave each stage on its own default and the
    # run would complete looking normal while none of the configured judge
    # reached M1/M2/M3/M5.
    loop = VLDiagnoseLoop(
        model=model,
        protocol=build_protocol(args.dataset),
        probe_agent=ProbeAgent(judge=judge, allow_codegen=True,
                               codegen_config=codegen),
        stats_agent=StatsAnalysisAgent(judge=judge, allow_codegen=True,
                                       codegen_config=codegen),
        diagnosis_agent=DiagnosisAgent(judge=judge),
        hypothesis_tester=HypothesisTester(judge=judge),
        surgery_agent=SurgeryAgent(
            judge=judge, writer_config=ExperimentWriterConfig(cli_agent=codegen)),
        fix_agent=FixAgent(
            judge=judge,
            max_tier=str(CFG.get("fix_max_tier", "L3b")),
            cli_config=codegen,
            run_logger=logger,
            max_validation_cases=int(CFG.get("fix_validation_cases", 0)),
            exec_timeout_sec=int(CFG.get("fix_exec_timeout_sec", 1800)),
        ),
        max_cycles=args.max_cycles,
        run_logger=logger,
        confirm_split=args.confirm_split,
    )

    if args.analysis_only:
        report = loop.run_analysis(batch)
        print(f"[M1-M3] proposed {len(report.hypotheses)} hypotheses")
    else:
        report = loop.run(batch)
        print(f"[M1-M5] cycles={report.cycles} stopped_by={report.stopped_by} "
              f"verified={len(report.verified_hypotheses)}/"
              f"{len(report.all_test_results)}")
        for t in report.all_test_results:
            stmt = getattr(t.hypothesis, "statement", str(t.hypothesis))
            print(f"  - [{t.status}] conf={t.confidence:.2f} "
                  f"grade={t.evidence_grade} {stmt[:100]}")
        if not args.skip_m4:
            fix = loop.run_m4(report, batch)
            print(f"[M4] {'fix proposed' if fix else 'no verified hypothesis to fix'}")
            if fix is not None:
                outcome = loop.run_fix(report, batch)
                print("[fix]", getattr(outcome, "recommendation", None) or outcome)

    summary = {
        "model": args.model,
        "dataset": args.dataset,
        "batch": {k: report_in[k] for k in
                  ("n", "accuracy", "n_pass", "n_fail", "truncated_rate")},
        "confirm_split": args.confirm_split,
        "analysis_only": args.analysis_only,
        "n_hypotheses": len(getattr(report, "hypotheses", []) or []),
        "n_verified": len(getattr(report, "verified_hypotheses", []) or []),
        "model_calls": model.n_calls,
        "model_truncated": model.n_truncated,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nwrote {out/'summary.json'}")
    print(f"dashboard: python -m evalvitals.cli dashboard {out}")


if __name__ == "__main__":
    main()
