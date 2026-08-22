"""Shared M1→M5→Fix runner for the benchmark-backed VLM examples.

The benchmark-specific ``run.py`` files only define the task protocol and
manifest path.  Keeping the execution and statistical split identical makes
the two examples directly comparable.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class BenchmarkConfig:
    name: str
    manifest: str
    task_domain: str
    description: str
    success_criteria: str
    # Spec key of the model under test (``--model`` overrides it per run).
    model: str = "qwen2.5-vl-7b-instruct"


_ARTICLES = re.compile(r"\b(a|an|the)\b", re.IGNORECASE)
_NUMBER = re.compile(r"[-+]?\d+(?:,\d{3})*(?:\.\d+)?%?")
_ANSWER_TAG = re.compile(
    r"(?:final\s+answer|answer)\s*(?:[:=]|\bis\b)\s*(.+)",
    re.IGNORECASE,
)


def normalize_answer(value: Any) -> str:
    text = str(value).strip().lower()
    text = text.replace("−", "-").replace("–", "-")
    text = _ARTICLES.sub(" ", text)
    text = re.sub(r"[^\w.%+\-]+", " ", text)
    return " ".join(text.split())


def _number(value: Any) -> float | None:
    match = _NUMBER.fullmatch(normalize_answer(value).replace(" ", ""))
    if not match:
        return None
    raw = match.group(0).replace(",", "")
    try:
        number = float(raw.rstrip("%"))
    except ValueError:
        return None
    # ChartQA treats a trailing percent sign as answer formatting: ``6.8`` and
    # ``6.8%`` denote the same benchmark answer, not 6.8 versus 0.068.
    return number


def answer_matches(observed: str, expected: Any, *, numeric_tolerance: float) -> bool:
    """Official-style relaxed answer match used by discovery and FixAgent."""
    golds = expected if isinstance(expected, list) else [expected]
    raw = str(observed or "")
    tagged = [m.group(1).splitlines()[0].strip() for m in _ANSWER_TAG.finditer(raw)]
    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    # Score a label-blind final-answer extraction, not the first line. FixAgent
    # candidates are allowed to reason before emitting ``Final answer: ...``;
    # first-line scoring turned every such correct repair into a regression.
    extracted = tagged[-1] if tagged else (lines[-1] if lines else raw)
    candidate = normalize_answer(extracted)
    aliases = {"yes": "true", "no": "false"}
    candidate = aliases.get(candidate, candidate)
    for gold in golds:
        target = normalize_answer(gold)
        target = aliases.get(target, target)
        if candidate == target:
            return True
        candidate_number, target_number = _number(candidate), _number(target)
        if candidate_number is None or target_number is None:
            continue
        if target_number == 0:
            if abs(candidate_number) <= 1e-9:
                return True
        elif abs(candidate_number - target_number) <= numeric_tolerance * abs(target_number):
            return True
    return False


def load_cases(manifest_path: Path, limit: int):
    from evalvitals.core.case import CaseBatch, FailureCase, Inputs, Provenance, Source

    rows = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"empty or invalid manifest: {manifest_path}")
    selected = rows[:limit]
    cases = []
    for row in selected:
        image = (manifest_path.parent / row["image"]).resolve()
        if not image.is_file():
            raise FileNotFoundError(f"missing image for {row['id']}: {image}")
        metadata = dict(row.get("metadata", {}))
        metadata.update({
            "dataset": row.get("dataset", "unknown"),
            "task": "exact_or_numeric",
            "numeric_tolerance": float(row.get("numeric_tolerance", 0.0)),
        })
        cases.append(FailureCase(
            id=str(row["id"]),
            inputs=Inputs(prompt=str(row["prompt"]), image=str(image)),
            expected=row["answers"],
            metadata=metadata,
            provenance=Provenance(source=Source.DATASET, metadata={"manifest": str(manifest_path)}),
        ))
    return CaseBatch(cases), selected


def score_bool(case, observed: str) -> bool:
    return answer_matches(
        observed,
        case.expected,
        numeric_tolerance=float(case.metadata.get("numeric_tolerance", 0.0)),
    )


def score_label(case, observed: str):
    from evalvitals.core.case import Label
    return Label.PASS if score_bool(case, observed) else Label.FAIL


def _self_test(config: BenchmarkConfig, manifest: Path, limit: int) -> None:
    cases, _ = load_cases(manifest, limit)
    first = list(cases)[0]
    gold = first.expected[0] if isinstance(first.expected, list) else first.expected
    assert score_bool(first, str(gold))
    assert not score_bool(first, "__definitely_wrong__")
    assert answer_matches("105", ["100"], numeric_tolerance=0.05)
    assert not answer_matches("106", ["100"], numeric_tolerance=0.05)
    print(f"Smoke test passed: {config.name}, {len(cases)} manifest cases and scorer checks.")


def main(config: BenchmarkConfig) -> None:
    parser = argparse.ArgumentParser(description=f"EvalVitals M1-M5+Fix on {config.name}")
    parser.add_argument("--model", default=config.model)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--manifest", default=config.manifest)
    parser.add_argument("--limit", type=int, default=128)
    parser.add_argument("--max-cycles", type=int, default=1)
    parser.add_argument("--judge-provider", choices=["agy", "claude", "codex"], default="agy")
    parser.add_argument("--judge-model", default="")
    parser.add_argument("--judge-effort", default="high")
    parser.add_argument("--fix-tier", choices=["L1", "L2", "L3a"], default="L3a")
    parser.add_argument(
        "--allow-codegen", action=argparse.BooleanOptionalAction, default=True,
        help="Let the configured CLI coding agent synthesize and execute an L2 repair pipeline.",
    )
    parser.add_argument(
        "--auto-escalate", action=argparse.BooleanOptionalAction, default=False,
        help="Automatically continue from L2 into L3a when lower tiers do not validate.",
    )
    parser.add_argument(
        "--code-only", action="store_true",
        help="restrict the fix pool to the coder-written L2 pipeline "
             "(candidate_allowlist={'coded_pipeline'}, one judge candidate) — the "
             "autonomous-code-repair protocol these examples were first written "
             "for. Default: every admissible candidate competes.",
    )
    parser.add_argument(
        "--explore", action=argparse.BooleanOptionalAction, default=True,
        help="In-cycle free-form EDA between M1 and M2 (same coder CLI as the "
             "repair pipeline); charts + tables land under <run-dir>/explore.",
    )
    parser.add_argument("--run-dir", default="outputs")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    manifest = Path(args.manifest)
    if args.smoke_test:
        _self_test(config, manifest, args.limit)
        return

    import evalvitals
    from evalvitals.eval_agent import (
        CaseDiscoveryAgent,
        CliAgentConfig,
        DiagnosisAgent,
        ExperimentProtocol,
        FixAgent,
        HypothesisTester,
        ProbeAgent,
        RunContext,
        StatsAnalysisAgent,
        StrategyProbe,
        VLDiagnoseLoop,
    )

    protocol = ExperimentProtocol(
        description=config.description,
        task_domain=config.task_domain,
        success_criteria=config.success_criteria,
        target_modalities=frozenset({"text", "image"}),
    )
    candidates, rows = load_cases(manifest, args.limit)
    print(f"Loading {args.model} on {args.device}; benchmark={config.name}, n={len(candidates)}")
    model = evalvitals.load(
        args.model,
        backend="hf_local",
        device=args.device,
        dtype=args.dtype,
        max_new_tokens=64,
        want=["attention"],
    )
    if args.judge_provider == "agy":
        from evalvitals.agent_runtime.judges import AgyModel

        judge = AgyModel(model=args.judge_model, timeout_sec=300)
        coder_provider = "antigravity"
        coder_extra_args = ()
    elif args.judge_provider == "claude":
        from evalvitals.eval_agent import ClaudeModel

        judge = ClaudeModel(
            model=args.judge_model or "sonnet",
            effort=args.judge_effort,
            timeout_sec=300,
        )
        coder_provider = "claude_code"
        coder_extra_args = (("--effort", args.judge_effort) if args.judge_effort else ())
    else:
        from evalvitals.agent_runtime.judges import CodexModel

        # Terra is the default for this agentic benchmark path.  The same
        # explicitly named model is also handed to the exploratory and repair
        # coding turns below, so no AGY/Claude call is mixed into the run.
        judge = CodexModel(
            model=args.judge_model or "gpt-5.6-terra",
            timeout_sec=600,
        )
        coder_provider = "codex"
        coder_extra_args = ()
    probe = judge.generate("Reply with exactly OK")
    if not probe.strip():
        raise RuntimeError(f"{args.judge_provider} judge returned an empty availability probe")

    discovery = CaseDiscoveryAgent(
        scorer=score_label,
        generation_kwargs={"max_new_tokens": 64, "do_sample": False},
        include_unknown=False,
    ).discover(model, candidates, protocol=protocol)
    cases = discovery.cases
    print(
        f"Baseline: PASS={discovery.n_pass}, FAIL={discovery.n_fail}, "
        f"UNKNOWN={discovery.n_unknown}, accuracy={discovery.n_pass / max(1, len(cases)):.3f}"
    )
    if not discovery.has_m5_groups:
        raise RuntimeError("M5 needs both PASS and FAIL cases; adjust the frozen sample")
    # These benchmarks explicitly request one short answer. A terse response
    # that ended well below the 64-token cap is a normal EOS stop, not the
    # text-shape truncation heuristic used for free-form reasoning tasks.
    for case in cases:
        case.metadata["finish_reason"] = "stop"

    run_dir = Path(args.run_dir)
    # llm_benchmark layout: the run log + artifacts under <run-dir>/logs, the
    # explore report as its sibling <run-dir>/explore (where the dashboard's
    # logs*/run_log.jsonl merge and explore lookup both expect them).
    ctx = RunContext(run_dir / "logs", verbose=True, config={
        "benchmark": config.name,
        "model": args.model,
        "n_cases": len(cases),
        "manifest": str(manifest),
        "manifest_seed": rows[0].get("sample_seed") if rows else None,
        "confirm_split": 0.5,
        "fix_tier": args.fix_tier,
        "allow_codegen": args.allow_codegen,
        "auto_escalate": args.auto_escalate,
    })
    pinned = [
        "answer_extraction_audit",
        "selfcheck_consistency",
        "coverage_verification_gap",
    ]
    probe_agent = ProbeAgent(
        probe=StrategyProbe(priority_override={key: pinned for key in ("vlm", "agent", "llm")}),
        judge=None,
        max_analyzers=len(pinned),
    )
    stats_agent = StatsAnalysisAgent(
        judge=judge,
        figure_dir=str(ctx.figures_dir),
        max_signal_tools=16,
        allow_codegen=False,
    )
    fix_agent = FixAgent(
        judge=judge,
        max_tier=args.fix_tier,
        score_fn=score_bool,
        run_logger=ctx.logger,
        cli_config=(
            CliAgentConfig(
                provider=coder_provider,
                timeout_sec=420,
                model=(args.judge_model or "gpt-5.6-terra") if args.judge_provider == "codex" else args.judge_model,
                extra_args=coder_extra_args,
            )
            if args.allow_codegen else None
        ),
        allow_codegen=args.allow_codegen,
        run_context=ctx,
        # Keep enough paired cases to detect a conservative subtype-gated
        # repair: the 512-case launch uses 256 EXPLORE / 256 CONFIRM.
        max_validation_cases=256,
        alpha=0.05,
        # Full candidate family by default (judge L1/L2 prompts and specs,
        # the self_consistency floor, and the coded pipeline) — the
        # llm_benchmark shape. --code-only restores the single-candidate
        # autonomous-code-repair protocol: the agent may run two
        # feedback-driven code revisions on EXPLORE; run_fix then freezes
        # the best positive-net candidate and evaluates that one on
        # CONFIRM. No confirmation feedback enters authoring either way.
        candidate_allowlist={"coded_pipeline"} if args.code_only else None,
        max_repair_rounds=2,
        **({"max_judge_candidates": 1} if args.code_only else {}),
        # A conservative VLM repair commonly makes one baseline call plus
        # three enhanced passes.  At 128 confirmation cases ChartQA can exceed
        # twenty minutes on a 7B model, so budget execution per whole batch
        # rather than inheriting the text-agent default.
        exec_timeout_sec=2400,
    )
    from evalvitals.eval_agent import SurgeryAgent
    from evalvitals.eval_agent.stages.experiment_writer import ExperimentWriterConfig

    coder_cfg = CliAgentConfig(
        provider=coder_provider,
        model=(args.judge_model or "gpt-5.6-terra") if args.judge_provider == "codex" else args.judge_model,
        timeout_sec=900,
        extra_args=coder_extra_args,
    )
    explorer = None
    if args.explore:
        from evalvitals.agent_runtime.sandbox import ExperimentSandbox
        from evalvitals.analysis import ExploratoryAnalysisAgent

        explorer = ExploratoryAnalysisAgent(
            cli_config=coder_cfg,
            sandbox=ExperimentSandbox(
                workdir=run_dir / "explore" / "sandbox", cleanup=False),
            timeout_sec=900,
            max_attempts=2,
        )
    loop = VLDiagnoseLoop(
        model=model,
        protocol=protocol,
        probe_agent=probe_agent,
        stats_agent=stats_agent,
        diagnosis_agent=DiagnosisAgent(judge=judge),
        hypothesis_tester=HypothesisTester(judge=judge, min_effect=0.05),
        fix_agent=fix_agent,
        max_cycles=args.max_cycles,
        run_logger=ctx.logger,
        confirm_split=0.5,
        confirm_split_seed=20260818,
        surgery_agent=SurgeryAgent(
            judge=judge, writer_config=ExperimentWriterConfig(cli_agent=coder_cfg)),
        explorer=explorer,
        explore_dir=run_dir / "explore",
        verbose=True,
    )
    report = loop.run(cases)
    discovery_rows = [{
        "id": case.id,
        "prompt": case.inputs.prompt,
        "expected": case.expected,
        "observed": str(case.observed),
        "label": case.label.value,
    } for case in cases]
    ctx.write_diagnose_report(report, cases, discovery=discovery_rows)
    print(
        f"Diagnosis: stopped_by={report.stopped_by}, cycles={report.cycles}, "
        f"verified={len(report.verified_hypotheses)}"
    )
    fix_proposal = loop.run_m4(report, cases, allow_unverified=True)
    if fix_proposal is not None:
        tag = "verified" if report.verified_hypotheses else "UNVERIFIED (best lead)"
        print(f"M4 experiment on the {tag} hypothesis: status={fix_proposal.status}")
    else:
        print("M4: no hypothesis to experiment on")
    outcome = loop.run_fix(
        report,
        cases,
        max_tier=args.fix_tier,
        auto_escalate=args.auto_escalate,
        # No M5-verified hypothesis still gets a fix attempt on the best
        # unverified leads (the candidate validation on CONFIRM is the gate).
        allow_unverified=True,
    )
    for validation in outcome.attempted:
        effect = "n/a" if validation.effect is None else f"{validation.effect:+.3f}"
        print(
            f"Fix [{validation.candidate.tier.label}] {validation.candidate.name}: "
            f"fixed={validation.fixed}, repairs={validation.n_fixed}, "
            f"breaks={validation.n_broken}, effect={effect}"
        )
    ctx.finalize()
    print(f"Done. Full artifact guide: {(ctx.root / 'README.txt').resolve()}")
