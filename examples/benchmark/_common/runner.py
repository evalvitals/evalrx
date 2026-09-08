"""One execution path for every cell: Stage 0 discovery -> M1..M4 -> M5 -> fix.

Mirrors the validated wiring of ``examples/m1_m5/vlm_benchmark_common.py`` (the
ChartQA/Spatial457 chains) with the per-modality settings of the audio examples
and ``llm_benchmark`` folded in as task attributes: pinned M1 set, scorer,
protocol, generation budget. The model under test is ``hf_local`` by default
(white-box capture + paper-method candidates stay available); ``--backend
endpoint`` swaps in an OpenAI-compatible server for the black-box path and the
``gemini`` family runs through Google's Gen AI API (``--backend gemini``, forced).
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

from . import tasks as T
from .models import Resolved


def model_label(resolved: Resolved) -> str:
    from evalrx.specs import get_spec

    spec = get_spec(resolved.spec_key)
    return f"{resolved.label}, spec {spec.key} = {spec.hf_repo or ('api:' + spec.key)}"


def _pinned_priority_override(pinned: list[str]) -> dict[str, list[str]]:
    """Apply a benchmark task's pinned M1 list to every probe model kind."""
    return {k: list(pinned) for k in ("vlm", "avlm", "alm", "agent", "llm")}


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
    from evalrx.core.capability import Capability
    from evalrx.models.compose import compose
    from evalrx.specs import get_spec

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
        from evalrx.models.backends.openai_compat import openai_runtime

        sampling = {"temperature": gen.get("temperature", 0.0), "max_tokens": max_new}
        # vLLM reads non-OpenAI params from the JSON body: top_k (dropped from the
        # OpenAI schema) and chat_template_kwargs — without the latter the server
        # renders the repo template with ITS defaults, and Nemotron 3 defaults
        # enable_thinking=True, silently breaking the thinking-off policy that
        # hf_local enforces through the spec.
        extra_body: dict = {"chat_template_kwargs": dict(spec.chat_template_kwargs)}
        if gen.get("do_sample"):
            sampling["top_p"] = gen["top_p"]
            extra_body["top_k"] = gen["top_k"]
        runtime = openai_runtime(base_url=args.base_url, api_key=args.api_key,
                                 extra_body=extra_body, **sampling)
        # the endpoint's generate_fn carries the sampling itself; per-call kwargs are not forwarded
        return compose(spec, "api", runtime, set()), {}, spec
    if resolved.backend == "gemini":
        from evalrx.models.backends.gemini_compat import ThinkingPolicy, gemini_runtime

        sampling = {"temperature": gen.get("temperature", 0.0), "max_output_tokens": max_new}
        if gen.get("do_sample"):
            sampling["top_p"] = gen["top_p"]
            sampling["top_k"] = gen["top_k"]
        # Thinking OFF policy, Gemini edition: the runtime sends each model's
        # floor (minimal / low on 3.7-flash / budget 0 on 2.5) unless
        # --thinking-level / --thinking-budget name a setting or
        # --enable-thinking leaves the API default. No logprobs: Gemini returns
        # none for the 3.x models, so the backend claims GENERATE only.
        policy = ThinkingPolicy(level=args.thinking_level, budget=args.thinking_budget,
                                floor=not args.enable_thinking)
        runtime = gemini_runtime(api_key=args.api_key, timeout=args.request_timeout,
                                 retries=args.request_retries, thinking=policy,
                                 with_logprobs=False, **sampling)
        return compose(spec, "api", runtime, set()), {}, spec
    from evalrx.models.backends.base import RuntimeConfig

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
    providers as the m1_m5 examples; the benchmark CLI and compose files pin
    Codex / gpt-5.6-terra / medium."""
    if args.judge_provider == "agy":
        from evalrx.agent_runtime.judges import AgyModel

        judge = AgyModel(model=args.judge_model, timeout_sec=300)
        coder = ("antigravity", args.judge_model, ())
    elif args.judge_provider == "claude":
        from evalrx.eval_agent import ClaudeModel

        judge = ClaudeModel(model=args.judge_model or "sonnet", effort=args.judge_effort, timeout_sec=300)
        coder = ("claude_code", args.judge_model, (("--effort", args.judge_effort) if args.judge_effort else ()))
    else:
        from evalrx.agent_runtime.judges import CodexModel

        name = args.judge_model or "gpt-5.6-terra"
        judge = CodexModel(model=name, effort=args.judge_effort, timeout_sec=600)
        coder_extra = (
            ("-c", f'model_reasoning_effort="{args.judge_effort}"')
            if args.judge_effort else ()
        )
        coder = ("codex", name, coder_extra)
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


#: The API backends expose no internals: every L3a repair in the catalog runs a
#: white-box executor (contrastive decoding over corrupted inputs, attention-
#: guided crops) and L3b hooks the forward pass, so ``supports_tier`` is False
#: there and the ladder would only ever "recommend L3b" it cannot reach.
API_FIX_CEILING = "L2"


def effective_fix_tier(backend: str, requested: str) -> str:
    """``--fix-tier`` as the run can honour it: unchanged on hf_local, clamped to
    :data:`API_FIX_CEILING` on the endpoint / gemini backends."""
    from evalrx.eval_agent.stages.fix_tiers import parse_tier

    if backend == "hf_local" or parse_tier(requested) <= parse_tier(API_FIX_CEILING):
        return requested
    return API_FIX_CEILING


def _api_model_version(model) -> str | None:
    """The served model version behind an API model, when the runtime records it
    (gemini: ``response.model_version``; a stable id is re-pointed silently)."""
    fn = getattr(getattr(model, "runtime", None), "generate_fn", None)
    state = getattr(fn, "state", None) or {}
    version = state.get("model_version")
    return str(version) if version else None


def run_fix_isolated(loop, run_dir: Path, ctx, report, cases, **fix_kwargs):
    """``loop.run_fix`` with every file the run has written so far held in
    memory and off disk (``evalrx.eval_agent.label_quarantine``).

    By the time the fix stage starts, ``baseline.json``, ``logs/report/
    discovery_cases.json``, the ``case_record`` events, the M1 signal tables
    (``gold_yes`` is the gold answer on a yes/no task) and the M5 workspace all
    carry per-case labels for EVERY case, CONFIRM included -- and the coder
    CLI (``Bash Edit Write Read``) and the pipeline sandbox both run from a
    workspace two directory levels below them. The prompt-level withholding
    stays as it was; this closes the file-system channel beside it. Restored
    byte-for-byte afterwards, with ``fix_quarantine.json`` naming what was hidden.

    Not covered: the dataset manifest on the ``data/`` bind mount (``gold``
    column) -- a root process in the same container can always read it.
    """
    from evalrx.eval_agent.label_quarantine import quarantine_run_dir

    # V1 appends to run_log.jsonl (merge old + new afterwards). V2 rewrites
    # each M<n>/log.json and run.json atomically, and the rewritten document
    # already contains everything from before the fix — declare those as
    # rewrite logs, or the quarantine mistakes every one of them for a
    # conflict and leaves a non-JSON `log.json.pre_fix` beside each.
    managed = getattr(getattr(ctx, "logger", None), "managed_json_paths", None)
    quarantine_kwargs = ({"rewrite_logs": list(managed)} if managed
                         else {"append_logs": [ctx.log_path]})
    with quarantine_run_dir(run_dir, **quarantine_kwargs) as q:
        print(f"[fix] label quarantine: {len(q.hidden)} run-dir file(s) held in memory "
              "for the fix stage (restored afterwards; see fix_quarantine.json)")
        return loop.run_fix(report, cases, **fix_kwargs)


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

    # Winner-as-new-baseline (recursive rounds with an L2 spec winner): the
    # deployed pipeline becomes the handle everything downstream measures —
    # Stage-0, the M stages, and the fix stage's baseline arm — while fix
    # candidates run on the raw model and REPLACE the pipeline.
    raw_model, deployed_spec = model, None
    if getattr(args, "baseline_spec", ""):
        from evalrx.eval_agent.stages.fix_tools import SpecPipelineModel

        payload = json.loads(Path(args.baseline_spec).read_text(encoding="utf-8"))
        if isinstance(payload, dict) and isinstance(payload.get("payload"), dict):
            payload = payload["payload"]  # a fix attempt's result.json
        model = SpecPipelineModel(model, payload)
        deployed_spec = model.spec.to_dict()
        print(f"[baseline-spec] baseline = deployed pipeline {deployed_spec.get('name')!r} "
              f"(n_samples={model.spec.n_samples}, strategy={model.spec.strategy}); "
              "fix candidates run on the raw model and replace it")

    from evalrx.eval_agent import CaseDiscoveryAgent, RunContext

    tvt = args.split_mode == "tvt"
    ctx = None
    if not args.baseline_only:
        # Create the V2 sink before Stage-0 so each discovery request is durable
        # as it happens, including its real latency and any transport error.
        ctx = RunContext(
            run_dir / "logs", verbose=True,
            logger_version=os.environ.get("EVALRX_RUN_LOGGER_VERSION", "v2"),
            config={
                "benchmark": task.title, "dataset": task.name, "modality": task.modality,
                "model": args.model, "spec": spec.key, "hf_repo": spec.hf_repo,
                "backend": resolved.backend, "n_cases": len(candidates),
                "manifest": str(manifest),
                "manifest_seed": rows[0].get("sample_seed") if rows else None,
                "split_mode": args.split_mode,
                "confirm_split": (1 / 3 if tvt else 0.5), "test_split": (1 / 3 if tvt else 0.0),
                "fix_tier": args.fix_tier,
                "allow_codegen": args.allow_codegen, "auto_escalate": args.auto_escalate,
                "m1_selection": args.m1_selection, "generation_kwargs": gen_kwargs,
                "enable_thinking": bool(args.enable_thinking),
                "judge_provider": args.judge_provider, "judge_model": args.judge_model,
            },
        )

    discovery_model = model
    if ctx is not None and ctx.is_v2:
        from evalrx.eval_agent.model_instrumentation import InstrumentedModel

        candidate_list = list(candidates)
        discovery_model = InstrumentedModel(
            model, ctx.logger, cycle=-1, analyzer="case_discovery",
            case_prompts={c.inputs.prompt: c.id for c in candidate_list},
            batch_case_ids=[c.id for c in candidate_list],
        )
        candidates = candidate_list

    started = time.monotonic()
    discovery = CaseDiscoveryAgent(
        scorer=T.label_case, generation_kwargs=gen_kwargs, include_unknown=False,
        concurrency=getattr(args, "concurrency", 1),
    ).discover(discovery_model, candidates, protocol=protocol)
    cases = discovery.cases
    if ctx is not None:
        ctx.config["n_cases"] = len(cases)
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
    version = _api_model_version(model)
    if version:
        baseline["model_version"] = version
        print(f"[model] served version: {version}")
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
    if not discovery.has_m4_groups:
        raise SystemExit("M4 needs both PASS and FAIL cases; adjust --limit / --seed")

    from evalrx.eval_agent import (
        CliAgentConfig,
        DiagnosisAgent,
        FixAgent,
        HypothesisTester,
        ProbeAgent,
        StatsAnalysisAgent,
        StrategyProbe,
        SurgeryAgent,
        VLDiagnoseLoop,
    )
    from evalrx.eval_agent.stages.experiment_writer import ExperimentWriterConfig
    from evalrx.eval_agent.stages.repair_catalog import method_names

    assert ctx is not None
    coder_cfg = CliAgentConfig(provider=coder_provider, model=coder_model, timeout_sec=900,
                               extra_args=coder_extra)
    pinned = list(task.pinned_m1)
    if args.m1_selection == "pinned":
        probe_agent = ProbeAgent(
            # StrategyProbe dispatches by the model's detected kind.  Keep the
            # benchmark's pinned list authoritative for every supported kind,
            # including the audio-language kinds used by ALM/AVLM examples.
            probe=StrategyProbe(priority_override=_pinned_priority_override(pinned)),
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
    prewritten_code = (
        Path(args.fix_code_file).read_text(encoding="utf-8")
        if args.fix_code_file else ""
    )
    if deployed_spec is not None:
        fix_kwargs.update(candidate_model=raw_model, deployed_spec=deployed_spec)
    fix_agent = FixAgent(
        judge=judge, max_tier=args.fix_tier, score_fn=T.score_case, run_logger=ctx.logger,
        cli_config=(CliAgentConfig(provider=coder_provider, timeout_sec=420, model=coder_model,
                                   extra_args=coder_extra) if args.allow_codegen else None),
        allow_codegen=args.allow_codegen, run_context=ctx,
        max_validation_cases=args.fix_validation_cases, alpha=0.05,
        allow_adapted_paper_methods=args.allow_adapted_paper_methods,
        candidate_allowlist=(
            {args.fix_candidate} if args.fix_candidate
            else ({"coded_pipeline"} if args.code_only
                  else (method_names() if args.registered_repairs_only else None))
        ),
        max_repair_rounds=max(1, args.fix_repair_rounds),
        **({"max_judge_candidates": 1} if args.code_only else {}),
        exec_timeout_sec=args.fix_exec_timeout,
        prewritten_code=prewritten_code,
        concurrency=getattr(args, "concurrency", 1),
        **fix_kwargs,
    )
    explorer = None
    if args.explore:
        from evalrx.agent_runtime.sandbox import ExperimentSandbox
        from evalrx.analysis import ExploratoryAnalysisAgent

        explorer = ExploratoryAnalysisAgent(
            cli_config=coder_cfg,
            sandbox=ExperimentSandbox(
                workdir=(ctx.explore_dir / "sandbox" if ctx.is_v2
                         else run_dir / "explore" / "sandbox"),
                cleanup=False,
            ),
            timeout_sec=900, max_attempts=3,
        )
    loop = VLDiagnoseLoop(
        model=model, protocol=protocol, probe_agent=probe_agent, stats_agent=stats_agent,
        diagnosis_agent=DiagnosisAgent(judge=judge),
        hypothesis_tester=HypothesisTester(judge=judge, min_effect=0.05),
        fix_agent=fix_agent, max_cycles=args.max_cycles, run_logger=ctx.logger,
        confirm_split=(1 / 3 if tvt else 0.5), confirm_split_seed=20260818,
        test_split=(1 / 3 if tvt else 0.0),
        # tvt (default): deterministic 1:1:1 train/val/test. M1-M3 mine on
        # TRAIN; M4 verifies every proposed hypothesis ONCE on VAL (holdout
        # re-probe with the last cycle's pinned analyzers); the fix ladder is
        # searched and its winner selected on VAL; the frozen winner is scored
        # exactly once on TEST — verification and fix development never share
        # cases with the final significance gate.
        # legacy: the pre-2026-09 50/50 explore/confirm design, where M4
        # screened on EXPLORE and CONFIRM was reserved for the frozen repair.
        m4_holdout=tvt,
        surgery_agent=SurgeryAgent(judge=judge, writer_config=ExperimentWriterConfig(cli_agent=coder_cfg)),
        explorer=explorer,
        explore_dir=ctx.explore_dir if ctx.is_v2 else run_dir / "explore",
        verbose=True,
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
    fix_tier = effective_fix_tier(resolved.backend, args.fix_tier)
    summary["fix_tier"] = fix_tier
    if fix_tier != args.fix_tier:
        print(f"[fix] --fix-tier {args.fix_tier} clamped to {fix_tier}: the {resolved.backend} backend "
              "exposes no model internals (L3a/L3b need the white-box executors)")
    if not args.skip_fix:
        # No M4-verified hypothesis still gets M5 + a fix attempt on the best
        # unverified leads (the candidate validation on CONFIRM is the gate).
        # An explicitly pre-registered fix is already the experiment the
        # caller asked to validate.  Running an unrelated M5 surgery first is
        # pure latency and can contend for the same GPU; it cannot influence
        # the frozen candidate or its EXPLORE/CONFIRM verdict.
        if args.skip_m5:
            print("M5: skipped by --skip-m5 (tiered fix search remains enabled)")
        elif args.fix_candidate or args.code_only or args.registered_repairs_only:
            requested = (
                args.fix_candidate or ("coded_pipeline" if args.code_only else "registered methods")
            )
            print(f"M5: skipped for pre-registered fix scope {requested!r}")
        else:
            proposal = loop.run_m5(report, cases, allow_unverified=True)
            if proposal is not None:
                tag = "verified" if report.verified_hypotheses else "UNVERIFIED (best lead)"
                print(f"M5 experiment on the {tag} hypothesis: status={proposal.status}")
            else:
                print("M5: no hypothesis to experiment on")
        outcome = run_fix_isolated(loop, run_dir, ctx, report, cases, max_tier=fix_tier,
                                   auto_escalate=args.auto_escalate, allow_unverified=True)
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
    if ctx.is_v2:
        print(f"Done. Tidy M1-M5 logs: {ctx.root.resolve()}")
    else:
        print(f"Done. Full artifact guide: {(ctx.root / 'README.txt').resolve()}")
    return 0
