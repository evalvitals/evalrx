"""One execution path for every cell: Stage 0 discovery -> M1..M5 -> M4 -> fix.

Mirrors the validated wiring of ``examples/m1_m4/vlm_benchmark_common.py`` (the
ChartQA/Spatial457 chains) with the per-modality settings of the audio examples
and ``llm_benchmark`` folded in as task attributes: pinned M1 set, scorer,
protocol, generation budget. The model under test is ``hf_local`` by default
(white-box capture + paper-method candidates stay available); ``--backend
endpoint`` swaps in an OpenAI-compatible server for the black-box path.
"""

from __future__ import annotations

import json
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

from . import tasks as T
from .models import Resolved


def model_label(resolved: Resolved) -> str:
    from evalvitals.specs import get_spec

    spec = get_spec(resolved.spec_key)
    return f"{resolved.label}, spec {spec.key} = {spec.hf_repo}"


def generation_settings(task: T.Task, args) -> dict:
    """Stage 0 / analyzer generation kwargs. Short-answer tasks (vlm, alm) are
    greedy; the llm tasks sample at T=0.6 / top_p 0.95 / top_k 20 like
    ``llm_benchmark`` — under greedy decoding Qwen3.5 (thinking off too) falls
    into verbatim self-verification loops that run to the token cap on
    free-form reasoning prompts (measured: 8/8 causal-judgement items hit 2048
    tokens without an answer line). ``--temperature`` overrides per run."""
    temperature = args.temperature if args.temperature is not None else (0.6 if task.modality == "llm" else 0.0)
    max_new = int(args.max_new_tokens or task.max_new_tokens)
    if temperature > 0:
        return {"max_new_tokens": max_new, "do_sample": True, "temperature": float(temperature),
                "top_p": float(args.top_p), "top_k": int(args.top_k)}
    return {"max_new_tokens": max_new, "do_sample": False}


def load_model(resolved: Resolved, args, task: T.Task):
    """``(model, generation_kwargs, spec)`` for the chosen backend. Thinking stays OFF
    unless ``--enable-thinking`` (the specs send enable_thinking=False)."""
    from evalvitals.core.capability import Capability
    from evalvitals.models.compose import compose
    from evalvitals.specs import get_spec

    spec = get_spec(resolved.spec_key)
    if getattr(args, "model_path", None):
        # Only the location changes. The spec still decides the auto class, the
        # chat template kwargs and the modalities, so a local checkout is the
        # same model under test rather than a differently-configured one.
        spec = replace(spec, hf_repo=str(args.model_path))
    if args.enable_thinking:
        spec = replace(spec, chat_template_kwargs={**spec.chat_template_kwargs, "enable_thinking": True})
    gen = generation_settings(task, args)
    max_new = gen["max_new_tokens"]
    if resolved.backend == "endpoint":
        from evalvitals.models.backends.openai_compat import openai_runtime

        sampling = {"temperature": gen.get("temperature", 0.0), "max_tokens": max_new}
        if gen.get("do_sample"):
            sampling["top_p"] = gen["top_p"]
        runtime = openai_runtime(base_url=args.base_url, api_key=args.api_key, **sampling)
        # the endpoint's generate_fn carries the sampling itself; per-call kwargs are not forwarded
        return compose(spec, "api", runtime, set()), {}, spec
    from evalvitals.models.backends.base import RuntimeConfig

    device = args.device or (resolved.size.default_device if resolved.size is not None else "cuda")
    attn_choice = args.attn_impl or (resolved.size.attn_impl if resolved.size is not None else "sdpa")
    attn = None if attn_choice == "auto" else attn_choice
    # apply_chat_template: a TEXT spec's prompt is one user turn through the chat
    # template (with the spec's enable_thinking=False). Without it hf_local
    # tokenises the raw prompt — completion mode — and Qwen3.5 opens its own
    # <think> block and runs to the cap (the llm smokes of 2026-08-21).
    runtime = RuntimeConfig(device=device, dtype=args.dtype, attn_impl=attn, max_new_tokens=max_new,
                            apply_chat_template=True)
    want = {Capability.ATTENTION} if attn == "eager" else set()
    model = compose(spec, "hf_local", runtime, want)
    return model, gen, spec


def load_weights(model, resolved: Resolved, args, task: T.Task):
    """``model.load()`` with one fallback: model code without an SDPA dispatch
    (the remote NemotronH classes) raises ValueError on attn_implementation=sdpa;
    retry once with eager instead of failing the cell."""
    try:
        model.load()
        return model
    except ValueError as exc:
        if "scaled_dot_product_attention" not in str(exc) or getattr(model.runtime, "attn_impl", None) == "eager":
            raise
        print(f"[model] {resolved.spec_key}: no SDPA dispatch in the model code ({str(exc)[:80]}...); "
              "retrying with attn_impl=eager")
        args.attn_impl = "eager"
        model, _gen, _spec = load_model(resolved, args, task)
        model.load()
        return model


def build_judge(args):
    """``(judge, coder_provider, coder_model, coder_extra_args)`` — the same three
    providers as the m1_m4 examples; the CLI default is agy, our compose files
    pin claude / claude-opus-5 / high."""
    if args.judge_provider == "agy":
        from evalvitals.agent_runtime.judges import AgyModel

        judge = AgyModel(model=args.judge_model, timeout_sec=300)
        coder = ("antigravity", args.judge_model, ())
    elif args.judge_provider == "claude":
        from evalvitals.eval_agent import ClaudeModel

        judge = ClaudeModel(model=args.judge_model or "sonnet", effort=args.judge_effort, timeout_sec=300)
        coder = ("claude_code", args.judge_model, (("--effort", args.judge_effort) if args.judge_effort else ()))
    else:
        from evalvitals.agent_runtime.judges import CodexModel

        name = args.judge_model or "gpt-5.6-terra"
        judge = CodexModel(model=name, timeout_sec=600)
        coder = ("codex", name, ())
    probe = judge.generate("Reply with exactly OK")
    if not probe.strip():
        raise RuntimeError(f"{args.judge_provider} judge returned an empty availability probe")
    print(f"judge: {args.judge_provider} model={args.judge_model or 'default'} effort={args.judge_effort}")
    return (judge, *coder)


def ensure_manifest(task: T.Task, args) -> Path:
    manifest = T.manifest_path(Path(args.data_dir), task)
    if manifest.is_file():
        return manifest
    if args.no_download:
        raise SystemExit(f"{manifest} is missing and --no-download was given")
    seed = args.seed if args.seed is not None else task.default_seed
    n = args.download_limit if args.download_limit is not None else task.default_limit
    print(f"[data] freezing {task.name}: limit={n} seed={seed} -> {manifest.parent}")
    summary = task.download(manifest.parent, limit=n, seed=seed)
    print("[data] " + json.dumps(summary))
    return manifest


def run_dir_for(args, task: T.Task) -> Path:
    leaf = f"{task.name}.{args.run_tag}" if args.run_tag else task.name
    return Path(args.run_dir) / args.model / leaf


def run(args, task: T.Task, resolved: Resolved) -> int:
    manifest = ensure_manifest(task, args)
    if args.download_only:
        return 0
    limit = args.limit if args.limit is not None else task.default_limit
    candidates, rows = T.build_cases(task, manifest, limit)
    run_dir = run_dir_for(args, task)
    run_dir.mkdir(parents=True, exist_ok=True)
    label = model_label(resolved)
    print(f"[model] {label}; backend={resolved.backend}; benchmark={task.title}; n={len(candidates)}")

    judge = coder_provider = coder_model = coder_extra = None
    if not args.baseline_only:
        judge, coder_provider, coder_model, coder_extra = build_judge(args)

    model, gen_kwargs, spec = load_model(resolved, args, task)
    protocol = task.protocol(label)
    if hasattr(model, "load"):
        # load the weights OUTSIDE the timed discovery pass (hf_local is lazy)
        t0 = time.monotonic()
        model = load_weights(model, resolved, args, task)
        print(f"[model] weights loaded in {time.monotonic() - t0:.0f}s "
              f"(attn_impl={getattr(model.runtime, 'attn_impl', None)}); generation={gen_kwargs}")

    from evalvitals.eval_agent import CaseDiscoveryAgent

    started = time.monotonic()
    discovery = CaseDiscoveryAgent(
        scorer=T.label_case, generation_kwargs=gen_kwargs, include_unknown=False,
        concurrency=getattr(args, "concurrency", 1),
    ).discover(model, candidates, protocol=protocol)
    cases = discovery.cases
    elapsed = time.monotonic() - started
    accuracy = discovery.n_pass / max(1, len(cases))
    baseline = {
        "model": args.model, "spec": spec.key, "hf_repo": spec.hf_repo, "backend": resolved.backend,
        "dataset": task.name, "task": task.kind, "n": len(cases),
        "n_pass": discovery.n_pass, "n_fail": discovery.n_fail, "n_unknown": discovery.n_unknown,
        "accuracy": round(accuracy, 4), "seconds": round(elapsed, 1),
        "generation_kwargs": gen_kwargs, "enable_thinking": bool(args.enable_thinking),
        "manifest": str(manifest), "limit": limit,
        "cases": [{
            "id": c.id, "expected": c.expected, "observed": str(c.observed), "label": c.label.value,
        } for c in cases],
    }
    (run_dir / "baseline.json").write_text(json.dumps(baseline, indent=2, ensure_ascii=False) + "\n",
                                           encoding="utf-8")
    print(f"Baseline: PASS={discovery.n_pass}, FAIL={discovery.n_fail}, UNKNOWN={discovery.n_unknown}, "
          f"accuracy={accuracy:.3f} ({elapsed:.0f}s, {elapsed / max(1, len(cases)):.1f}s/case)")
    if task.modality == "llm" and not 0.15 <= accuracy <= 0.85:
        print("  NOTE accuracy is outside the usable band [0.15, 0.85]: M2 has little to contrast "
              "and the paired tests downstream are short of power (results valid, weak)")
    if task.short_answer:
        # A terse answer that ended well below the cap is a normal EOS stop, not
        # the free-form truncation heuristic.
        for case in cases:
            case.metadata["finish_reason"] = "stop"
    if args.baseline_only:
        print(f"Done (baseline only). {run_dir / 'baseline.json'}")
        return 0
    if not discovery.has_m5_groups:
        raise SystemExit("M5 needs both PASS and FAIL cases; adjust --limit / --seed")

    from evalvitals.eval_agent import (
        CliAgentConfig, DiagnosisAgent, FixAgent, HypothesisTester, ProbeAgent, RunContext,
        StatsAnalysisAgent, StrategyProbe, SurgeryAgent, VLDiagnoseLoop,
    )
    from evalvitals.eval_agent.stages.experiment_writer import ExperimentWriterConfig

    ctx = RunContext(run_dir / "logs", verbose=True, config={
        "benchmark": task.title, "dataset": task.name, "modality": task.modality,
        "model": args.model, "spec": spec.key, "hf_repo": spec.hf_repo, "backend": resolved.backend,
        "n_cases": len(cases), "manifest": str(manifest),
        "manifest_seed": rows[0].get("sample_seed") if rows else None,
        "confirm_split": 0.5, "fix_tier": args.fix_tier, "allow_codegen": args.allow_codegen,
        "auto_escalate": args.auto_escalate, "m1_selection": args.m1_selection,
        "generation_kwargs": gen_kwargs,
        "enable_thinking": bool(args.enable_thinking), "judge_provider": args.judge_provider,
        "judge_model": args.judge_model,
    })
    coder_cfg = CliAgentConfig(provider=coder_provider, model=coder_model, timeout_sec=900,
                               extra_args=coder_extra)
    pinned = list(task.pinned_m1)
    if args.m1_selection == "pinned":
        probe_agent = ProbeAgent(
            probe=StrategyProbe(priority_override={k: pinned for k in ("vlm", "agent", "llm")}),
            judge=None, max_analyzers=len(pinned),
            max_cases_per_analyzer=args.analyzer_max_cases,
        )
    else:
        probe_agent = ProbeAgent(judge=judge, allow_codegen=False,
                                 max_cases_per_analyzer=args.analyzer_max_cases)
    m2_codegen = args.m2_codegen if args.m2_codegen is not None else task.modality == "llm"
    stats_agent = StatsAnalysisAgent(
        judge=judge, figure_dir=str(ctx.figures_dir), max_signal_tools=16,
        allow_codegen=m2_codegen, **({"codegen_config": coder_cfg} if m2_codegen else {}),
    )
    fix_kwargs: dict[str, Any] = {}
    sampled = bool(gen_kwargs.get("do_sample"))
    # Noise model for a sampled baseline: per-case PASS RATE over k samples (the
    # frozen Stage 0 output + k-1 fresh), as llm_benchmark does; greedy = 1.
    fix_kwargs["baseline_repeats"] = (args.fix_baseline_repeats if args.fix_baseline_repeats is not None
                                      else (5 if sampled else 1))
    if task.modality == "llm":
        floor = {"max_tokens": gen_kwargs.get("max_new_tokens", task.max_new_tokens)}
        if sampled:
            floor.update(temperature=gen_kwargs["temperature"], top_p=gen_kwargs["top_p"])
        fix_kwargs.update(baseline_generation_kwargs=floor, floor_candidates=("self_consistency_5",))
    fix_agent = FixAgent(
        judge=judge, max_tier=args.fix_tier, score_fn=T.score_case, run_logger=ctx.logger,
        cli_config=(CliAgentConfig(provider=coder_provider, timeout_sec=420, model=coder_model,
                                   extra_args=coder_extra) if args.allow_codegen else None),
        allow_codegen=args.allow_codegen, run_context=ctx,
        max_validation_cases=args.fix_validation_cases, alpha=0.05,
        candidate_allowlist={"coded_pipeline"} if args.code_only else None,
        max_repair_rounds=2,
        **({"max_judge_candidates": 1} if args.code_only else {}),
        exec_timeout_sec=args.fix_exec_timeout,
        concurrency=getattr(args, "concurrency", 1),
        **fix_kwargs,
    )
    explorer = None
    if args.explore:
        from evalvitals.agent_runtime.sandbox import ExperimentSandbox
        from evalvitals.analysis import ExploratoryAnalysisAgent

        explorer = ExploratoryAnalysisAgent(
            cli_config=coder_cfg,
            sandbox=ExperimentSandbox(workdir=run_dir / "explore" / "sandbox", cleanup=False),
            timeout_sec=900, max_attempts=2,
        )
    loop = VLDiagnoseLoop(
        model=model, protocol=protocol, probe_agent=probe_agent, stats_agent=stats_agent,
        diagnosis_agent=DiagnosisAgent(judge=judge),
        hypothesis_tester=HypothesisTester(judge=judge, min_effect=0.05),
        fix_agent=fix_agent, max_cycles=args.max_cycles, run_logger=ctx.logger,
        confirm_split=0.5, confirm_split_seed=20260818,
        surgery_agent=SurgeryAgent(judge=judge, writer_config=ExperimentWriterConfig(cli_agent=coder_cfg)),
        explorer=explorer, explore_dir=run_dir / "explore", verbose=True,
    )
    report = loop.run(cases)
    discovery_rows = [{
        "id": c.id, "prompt": c.inputs.prompt, "expected": c.expected,
        "observed": str(c.observed), "label": c.label.value,
    } for c in cases]
    ctx.write_diagnose_report(report, cases, discovery=discovery_rows)
    print(f"Diagnosis: stopped_by={report.stopped_by}, cycles={report.cycles}, "
          f"verified={len(report.verified_hypotheses)}")
    summary: dict[str, Any] = {
        "model": args.model, "spec": spec.key, "dataset": task.name, "modality": task.modality,
        "backend": resolved.backend, "n_cases": len(cases), "baseline_accuracy": round(accuracy, 4),
        "cycles": report.cycles, "stopped_by": str(report.stopped_by),
        "n_verified": len(report.verified_hypotheses), "fix": None,
    }
    if not args.skip_fix:
        # No M5-verified hypothesis still gets M4 + a fix attempt on the best
        # unverified leads (the candidate validation on CONFIRM is the gate).
        proposal = loop.run_m4(report, cases, allow_unverified=True)
        if proposal is not None:
            tag = "verified" if report.verified_hypotheses else "UNVERIFIED (best lead)"
            print(f"M4 experiment on the {tag} hypothesis: status={proposal.status}")
        else:
            print("M4: no hypothesis to experiment on")
        outcome = loop.run_fix(report, cases, max_tier=args.fix_tier, auto_escalate=args.auto_escalate,
                               allow_unverified=True)
        attempted = []
        for validation in outcome.attempted:
            effect = "n/a" if validation.effect is None else f"{validation.effect:+.3f}"
            print(f"Fix [{validation.candidate.tier.label}] {validation.candidate.name}: "
                  f"fixed={validation.fixed}, repairs={validation.n_fixed}, "
                  f"breaks={validation.n_broken}, effect={effect}")
            attempted.append({"tier": validation.candidate.tier.label, "name": validation.candidate.name,
                              "fixed": bool(validation.fixed), "repairs": validation.n_fixed,
                              "breaks": validation.n_broken, "effect": validation.effect})
        summary["fix"] = {"recommendation": str(getattr(outcome, "recommendation", "") or ""),
                          "attempted": attempted}
    ctx.finalize()
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Done. Full artifact guide: {(ctx.root / 'README.txt').resolve()}")
    return 0
