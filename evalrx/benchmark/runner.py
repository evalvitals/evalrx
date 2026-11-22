"""One execution path for every cell: Stage 0 discovery -> M1..M4 -> M5 -> fix.

Mirrors the validated wiring of ``examples/m1_m5/vlm_benchmark_common.py`` (the
ChartQA/Spatial457 chains) with the per-modality settings of the audio examples
and ``llm_benchmark`` folded in as task attributes: pinned M1 set, scorer,
protocol, generation budget. The model under test runs on ``endpoint`` by
default (an OpenAI-compatible server — vLLM in practice — at ``--base-url``;
fix ladder clamped to L2 there); ``--backend hf_local`` swaps in in-process
transformers for the white-box path (capture + L3a/L3b paper-method
candidates) and the ``gemini`` family runs through Google's Gen AI API
(``--backend gemini``, forced).
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

from . import tasks as T
from .models import Resolved

#: Seed of the label-stratified EXPLORE/CONFIRM split, shared by the diagnosis
#: loop and the triage controller so both see the same EXPLORE cases.
CONFIRM_SPLIT_SEED = 20260818


def model_label(resolved: Resolved) -> str:
    from evalrx.specs import get_spec

    spec = get_spec(resolved.spec_key)
    return f"{resolved.label}, spec {spec.key} = {spec.hf_repo or ('api:' + spec.key)}"


def _print_fix_progress(event):
    # No scores, outputs, labels, or case IDs: safe even while CONFIRM is sealed.
    print(f"[fix] {event['phase']} candidate={event.get('candidate', '-')} "
          f"completed={event.get('completed', 0)}/{event.get('total', '?')}", flush=True)


def _benchmark_agentic_registry(loop, state):
    """Keep repair/CONFIRM access outside the adaptive controller."""
    from evalrx.eval_agent.loop.judge.tools import ToolRegistry, build_default_registry

    full = build_default_registry(loop, state)
    registry = ToolRegistry()
    for name in sorted(full.tool_names() - {"run_fix", "run_surgery"}):
        registry.register(full.get(name))
    return registry


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
        from evalrx.models.backends.api.openai import openai_runtime

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
        runtime.concurrency = max(1, int(args.concurrency))
        # the endpoint's generate_fn carries the sampling itself; per-call kwargs are not forwarded
        return compose(spec, "api", runtime, set()), {}, spec
    if resolved.backend == "gemini":
        from evalrx.models.backends.api.gemini import ThinkingPolicy, gemini_runtime

        sampling = {"temperature": gen.get("temperature", 0.0), "max_output_tokens": max_new}
        if gen.get("do_sample"):
            sampling["top_p"] = gen["top_p"]
            sampling["top_k"] = gen["top_k"]
        # Thinking OFF policy, Gemini edition: the runtime sends each model's
        # floor (minimal / low on 3.7-flash / budget 0 on 2.5) unless
        # --thinking-level / --thinking-budget name a setting or
        # --enable-thinking leaves the API default. Probe actual probability
        # availability rather than inferring it from a model name or SDK field.
        policy = ThinkingPolicy(level=args.thinking_level, budget=args.thinking_budget,
                                floor=not args.enable_thinking)
        runtime = gemini_runtime(api_key=args.api_key, timeout=args.request_timeout,
                                 retries=args.request_retries, thinking=policy,
                                 with_logprobs="auto",
                                 logprobs_model=spec.hf_repo or spec.key,
                                 top_logprobs=5, **sampling)
        runtime.concurrency = max(1, int(args.concurrency))
        return compose(spec, "api", runtime, set()), {}, spec
    from evalrx.models.backends.base import RuntimeConfig

    # The decoding Stage 0 labels with is the model's own: a probe that calls
    # generate() with no arguments then answers under the same settings.
    sampling = {name: value for name, value in gen.items() if name != "max_new_tokens"}
    if resolved.backend == "jax_local":
        # The JAX twin: same spec (template kwargs, modalities, caveats), the
        # gemma library underneath; text, image and audio inputs. apply_chat_template
        # as on hf_local. A --model-path is a local Orbax mirror of the spec's
        # gs:// checkpoint (17 GB for E2B; see docs/design_jax_backend.md section 4).
        engine_kwargs: dict[str, Any] = {"checkpoint": str(args.model_path)} if getattr(args, "model_path", None) else {}
        # text cells skip the vision / audio towers (~2 GB float32 and their load
        # time); the vlm / alm cells need them, and the adapter reports the
        # modalities it can serve accordingly
        engine_kwargs["text_only"] = task.modality == "llm"
        runtime = RuntimeConfig(device=args.device or "auto", dtype=args.dtype, max_new_tokens=max_new,
                                apply_chat_template=True, engine_kwargs=engine_kwargs, sampling=sampling)
        return compose(spec, "jax_local", runtime, set()), gen, spec

    device = args.device or (resolved.size.default_device if resolved.size is not None else "cuda")
    attn_choice = args.attn_impl or (resolved.size.attn_impl if resolved.size is not None else "sdpa")
    attn = None if attn_choice == "auto" else attn_choice
    # apply_chat_template: a TEXT spec's prompt is one user turn through the chat
    # template (with the spec's enable_thinking=False). Without it hf_local
    # tokenises the raw prompt — completion mode — and Qwen3.5 opens its own
    # <think> block and runs to the cap (the llm smokes of 2026-08-21).
    runtime = RuntimeConfig(device=device, dtype=args.dtype, attn_impl=attn, max_new_tokens=max_new,
                            apply_chat_template=True, sampling=sampling)
    want = {Capability.ATTENTION} if attn == "eager" else set()
    model = compose(spec, "hf_local", runtime, want)
    return model, gen, spec


#: M1 analyzers that RE-SAMPLE the model and are only informative at
#: temperature > 0 (``coverage_verification_gap`` documents the requirement;
#: ``self_consistency`` is trivially 1.0 under greedy decoding). ProbeAgent
#: instantiates the pinned set with constructor defaults — empty gen_kwargs —
#: and hf_local's generate() then runs greedy (the HF generation_config has no
#: do_sample), so on every in-process run the coverage probe reported
#: degenerate_sampling=true and M3 spent a hypothesis on the harness
#: (qwen3.5-2b/chartqa, 2026-09-07). The API backends carry their sampling in
#: the runtime and ignore per-call kwargs, so the overrides are hf_local-only.
_SAMPLED_M1 = ("self_consistency", "coverage_verification_gap")


def _sampling_kwargs(resolved: Resolved, args) -> "dict | None":
    """Per-call sampling settings for re-sampling analyzers; ``None`` where the
    backend fixes sampling in its runtime and ignores per-call kwargs.

    The gemini runtime merges per-call kwargs over its own sampling
    (``_Runtime.config``), so a temperature reaches the request there; the
    endpoint's generate_fn drops them.
    """
    if resolved.backend in ("hf_local", "jax_local"):
        return {"temperature": 1.0, "top_p": args.top_p, "top_k": args.top_k}
    if resolved.backend == "gemini":
        return {"temperature": 1.0}
    return None


def _usage_cost(model, before: "dict | None") -> dict:
    """Provider usage since *before* and its listed-price USD (``{}`` for a model with no counters)."""
    from evalrx.eval_agent.triage.metering import usage_counters, usage_delta
    from evalrx.models.backends.api.prices import usd

    usage = usage_delta(before, usage_counters(model))
    if usage is None:
        return {}
    model_id = getattr(getattr(model, "spec", None), "hf_repo", None) or getattr(getattr(model, "spec", None), "key", None)
    return {"api_usage": usage, "usd": usd(model_id, usage)}


def _sampling_analyzer_overrides(resolved: Resolved, args) -> dict:
    gen = _sampling_kwargs(resolved, args)
    if gen is None:
        return {}
    from evalrx.analyzers.uncertainty.coverage_gap import CoverageVerificationGap
    from evalrx.analyzers.uncertainty.self_consistency import SelfConsistencyAnalyzer

    return {
        "self_consistency": SelfConsistencyAnalyzer(gen_kwargs=dict(gen)),
        "coverage_verification_gap": CoverageVerificationGap(gen_kwargs=dict(gen)),
    }


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
    providers as the m1_m5 examples; the benchmark CLI defaults to
    Claude / sonnet / medium."""
    if args.judge_provider == "gemini":
        adaptive = getattr(args, "diagnosis_mode", "agentic") == "agentic"
        probe_codegen = adaptive and args.allow_codegen
        stats_codegen = args.m2_codegen if args.m2_codegen is not None else adaptive
        if (args.allow_codegen or args.explore or stats_codegen or probe_codegen
                or not args.skip_m5) and not getattr(args, "coder_provider", None):
            raise ValueError("The Gemini API judge requires --no-allow-codegen --no-explore "
                             "--no-m2-codegen --skip-surgery; coding-agent stages need a CLI judge.")
        from evalrx.models import compose
        from evalrx.models.backends.api.gemini import gemini_runtime

        name = args.judge_model or "gemini-2.5-flash"
        judge = compose(name, "api", gemini_runtime(temperature=0.0, max_output_tokens=8192))
        coder = ("llm", name, ())
    elif args.judge_provider == "agy":
        from evalrx.agent_runtime.judges import AgyModel

        judge = AgyModel(model=args.judge_model, effort=args.judge_effort, timeout_sec=300)
        coder = ("antigravity", args.judge_model,
                 (("--effort", args.judge_effort) if args.judge_effort else ()))
    elif args.judge_provider == "claude":
        from evalrx.eval_agent import ClaudeModel

        name = args.judge_model or ClaudeModel.DEFAULT_MODEL

        judge = ClaudeModel(model=name, effort=args.judge_effort, timeout_sec=300)
        coder = ("claude_code", name, (("--effort", args.judge_effort) if args.judge_effort else ()))
    else:
        from evalrx.agent_runtime.judges import CodexModel

        name = args.judge_model or "gpt-5.6-terra"
        judge = CodexModel(model=name, effort=args.judge_effort, timeout_sec=600)
        coder_extra = (
            ("-c", f'model_reasoning_effort="{args.judge_effort}"')
            if args.judge_effort else ()
        )
        coder = ("codex", name, coder_extra)
    if getattr(args, "coder_provider", None):
        coder = (args.coder_provider, args.coder_model or "gemini-2.5-flash", ())
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
    print(f"[data] freezing {task.name}: limit={n} seed={seed}"
          + (f" val_limit={task.val_limit}" if task.val_limit else "") + f" -> {manifest.parent}")
    # A task with a validation slice freezes BOTH manifests in one call — the
    # val draw is derived from (and disjoint with) the main draw, so they must
    # be frozen together or not at all.
    kwargs = {"val_limit": task.val_limit} if task.val_limit else {}
    summary = task.download(manifest.parent, limit=n, seed=seed, **kwargs)
    print("[data] " + json.dumps(summary))
    return manifest


def ensure_val_manifest(task: T.Task, args) -> Path:
    """The held-out validation manifest ``--held-out`` runs on. Never (re)built
    here: a fresh ``ensure_manifest`` download writes it alongside the main
    manifest; a data dir frozen before the val feature must be re-frozen
    deliberately, not silently overwritten."""
    if not task.val_limit:
        raise SystemExit(f"--held-out: task {task.name!r} defines no validation slice "
                         "(val_limit=0); only non-census datasets carry one")
    val_manifest = T.val_manifest_path(Path(args.data_dir), task)
    if not val_manifest.is_file():
        raise SystemExit(
            f"--held-out: {val_manifest} is missing. This data dir was frozen before "
            f"the validation slice existed — delete {val_manifest.parent} to re-freeze "
            "both manifests (the main draw is deterministic and unchanged), or run "
            "without --held-out.")
    return val_manifest


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
    :data:`API_FIX_CEILING` on the endpoint / gemini backends — and, for now, on
    jax_local too: it reads internals but the L3a executors and L3b hooks are
    still hf_local methods (phases 2-3 of docs/design_jax_backend.md)."""
    from evalrx.eval_agent.stages.fix.tiers import parse_tier

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

    # The logger rewrites each M<n>/log.json and run.json atomically, and the
    # rewritten document already contains everything from before the fix —
    # declare those as rewrite logs, or the quarantine mistakes every one of
    # them for a conflict and leaves a non-JSON `log.json.pre_fix` beside each.
    managed = ctx.logger.managed_json_paths
    with quarantine_run_dir(run_dir, rewrite_logs=list(managed)) as q:
        print(f"[fix] label quarantine: {len(q.hidden)} run-dir file(s) held in memory "
              "for the fix stage (restored afterwards; see fix_quarantine.json)")
        return loop.run_fix(report, cases, **fix_kwargs)


def _load_target(resolved: Resolved, args, task: T.Task, run_dir: Path):
    """``(model, generation_kwargs, spec)`` for the model under test, weights in memory.

    The weights load here, OUTSIDE the timed discovery pass (hf_local is lazy).
    """
    model, gen_kwargs, spec = load_model(resolved, args, task)
    if resolved.backend == "gemini":
        from evalrx.models.backends.api.gemini import runtime_state

        capabilities: dict[str, Any] = {"capabilities": sorted(c.value for c in model.capabilities),
                        "runtime": runtime_state(model)}
        (run_dir / "api_capabilities.json").write_text(
            json.dumps(capabilities, indent=2) + "\n", encoding="utf-8")
        print(f"[model] logprobs: {capabilities['runtime'].get('logprobs_probe')}")
    if hasattr(model, "load"):
        t0 = time.monotonic()
        model = load_weights(model, resolved, args, task)
        print(f"[model] weights loaded in {time.monotonic() - t0:.0f}s "
              f"(attn_impl={getattr(model.runtime, 'attn_impl', None)}); generation={gen_kwargs}")
        if resolved.backend == "jax_local":
            import jax

            print(f"[device] {jax.default_backend()}: "
                  + ", ".join(f"{d} ({d.device_kind})" for d in jax.devices()))
    return model, gen_kwargs, spec


def _deploy_baseline_spec(model, spec_path: str):
    """``(the baseline to measure, its spec as a dict)`` for ``--baseline-spec``.

    Winner-as-new-baseline (recursive rounds with an L2 spec winner): the
    deployed pipeline becomes the handle everything downstream measures —
    Stage-0, the M stages, and the fix stage's baseline arm — while fix
    candidates run on the raw model and REPLACE the pipeline. Without a spec
    the model itself is the baseline.
    """
    if not spec_path:
        return model, None
    from evalrx.eval_agent.stages.fix.tools import SpecPipelineModel

    payload = json.loads(Path(spec_path).read_text(encoding="utf-8"))
    if isinstance(payload, dict) and isinstance(payload.get("payload"), dict):
        payload = payload["payload"]  # a fix attempt's result.json
    deployed = SpecPipelineModel(model, payload)
    deployed_spec = deployed.spec.to_dict()
    print(f"[baseline-spec] baseline = deployed pipeline {deployed_spec.get('name')!r} "
          f"(n_samples={deployed.spec.n_samples}, strategy={deployed.spec.strategy}); "
          "fix candidates run on the raw model and replace it")
    return deployed, deployed_spec


def _run_config(args, task: T.Task, resolved: Resolved, spec, gen_kwargs: dict, manifest: Path,
                rows: list, n_cases: int) -> dict[str, Any]:
    """How this cell was asked to run, as the run log records it."""
    return {
        "benchmark": task.title, "dataset": task.name, "modality": task.modality,
        "model": args.model, "spec": spec.key, "hf_repo": spec.hf_repo,
        "backend": resolved.backend, "n_cases": n_cases,
        "manifest": str(manifest),
        "manifest_seed": rows[0].get("sample_seed") if rows else None,
        "split_mode": ("held_out_val" if args.held_out else "explore_confirm"),
        "confirm_split": float(args.confirm_split), "test_split": 0.0,
        "val_limit": int(getattr(args, "val_limit", 0) or 0),
        "held_out": bool(args.held_out),
        "fix_tier": args.fix_tier,
        "allow_codegen": args.allow_codegen, "auto_escalate": args.auto_escalate,
        "m1_selection": args.m1_selection, "generation_kwargs": gen_kwargs,
        "diagnosis_mode": getattr(args, "diagnosis_mode", "agentic"),
        "max_actions": getattr(args, "max_actions", 12),
        "diagnosis_retries": 1,
        "enable_thinking": bool(args.enable_thinking),
        "judge_provider": args.judge_provider, "judge_model": args.judge_model,
    }


def label_samples(task: T.Task, args) -> int:
    """Answers Stage 0 draws per case. A sampling model can answer one case
    both ways, so a single answer labels it by chance; the majority of five
    is the default there, and one answer when decoding is greedy."""
    if args.label_samples is not None:
        return max(1, int(args.label_samples))
    return 5 if generation_settings(task, args)["do_sample"] else 1


def _discover(model, candidates, protocol, gen_kwargs: dict, args, task: T.Task):
    """Stage 0: sample each candidate's answers and label it. ``(discovery, seconds)``."""
    from evalrx.eval_agent import CaseDiscoveryAgent

    started = time.monotonic()
    discovery = CaseDiscoveryAgent(
        scorer=T.label_case, generation_kwargs=gen_kwargs, include_unknown=False,
        n_samples=label_samples(task, args),
    ).discover(model, candidates, protocol=protocol)
    return discovery, time.monotonic() - started


def _case_labels(cases) -> list[dict[str, Any]]:
    """One row per case; ``votes`` when the label is a majority of several answers."""
    rows = []
    for c in cases:
        row = {"id": c.id, "expected": c.expected, "observed": str(c.observed), "label": c.label.value}
        if "discovery_votes" in c.metadata:
            row["votes"] = c.metadata["discovery_votes"]
        rows.append(row)
    return rows


def _split_votes(cases) -> int:
    """Cases whose sampled answers did not all score the same."""
    votes = [c.metadata.get("discovery_votes", {}) for c in cases]
    return sum(1 for v in votes if v.get("pass") and v.get("fail"))


def _mark_short_answers_stopped(task: T.Task, cases) -> None:
    """A terse answer that ended well below the cap is a normal EOS stop, not
    the free-form truncation heuristic."""
    if task.short_answer:
        for case in cases:
            case.metadata["finish_reason"] = "stop"


def _label_validation_set(model, val_manifest: Path, protocol, gen_kwargs: dict, args, task: T.Task,
                          spec, run_dir: Path):
    """The ``--held-out`` validation cases, labeled by their own Stage-0 pass.

    A SEPARATE frozen sample. It takes the M4-holdout + fix-development roles;
    the main batch keeps its full explore/confirm split and CONFIRM stays the
    single frozen-winner gate.
    """
    val_candidates, _ = T.build_cases(task, val_manifest, int(getattr(args, "val_limit", 0) or 0))
    val_discovery, val_elapsed = _discover(model, val_candidates, protocol, gen_kwargs, args, task)
    val_cases = val_discovery.cases
    val_accuracy = val_discovery.n_pass / max(1, len(val_cases))
    (run_dir / "baseline_val.json").write_text(json.dumps({
        "model": args.model, "spec": spec.key, "dataset": task.name, "role": "validation",
        "n": len(val_cases), "n_pass": val_discovery.n_pass, "n_fail": val_discovery.n_fail,
        "n_unknown": val_discovery.n_unknown, "accuracy": round(val_accuracy, 4),
        "seconds": round(val_elapsed, 1), "manifest": str(val_manifest),
        "label_samples": label_samples(task, args), "n_split": _split_votes(val_cases),
        "cases": _case_labels(val_cases),
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Validation baseline: PASS={val_discovery.n_pass}, FAIL={val_discovery.n_fail}, "
          f"accuracy={val_accuracy:.3f} ({val_elapsed:.0f}s, n={len(val_cases)})")
    _mark_short_answers_stopped(task, val_cases)
    if not val_discovery.has_m4_groups:
        print("  NOTE the validation set lacks a PASS or FAIL group; the holdout "
              "M4 verification there will have nothing to contrast")
    return val_cases


def _is_agentic(args) -> bool:
    return getattr(args, "diagnosis_mode", "agentic") == "agentic"


def _probe_agent(args, task: T.Task, resolved: Resolved, model, judge, coder_cfg, config: dict[str, Any]):
    """The M1 agent; records its worker count and selection mode in ``config``."""
    from evalrx.eval_agent import ProbeAgent, StrategyProbe

    pinned = list(task.pinned_m1)
    if resolved.backend == "gemini":
        from evalrx.core.capability import Capability

        if Capability.LOGPROBS in model.capabilities and "logprob_entropy" not in pinned:
            pinned.append("logprob_entropy")
    overrides = _sampling_analyzer_overrides(resolved, args)
    agentic = _is_agentic(args)
    m1_codegen = agentic and args.allow_codegen
    config["m1_codegen"] = bool(m1_codegen)
    config["m1_selection"] = "judge" if agentic or m1_codegen else args.m1_selection
    if args.m1_selection == "pinned" and not agentic and not m1_codegen:
        return ProbeAgent(
            # StrategyProbe dispatches by the model's detected kind.  Keep the
            # benchmark's pinned list authoritative for every supported kind,
            # including the audio-language kinds used by ALM/AVLM examples.
            probe=StrategyProbe(priority_override=_pinned_priority_override(pinned)),
            judge=None, max_analyzers=len(pinned),
            max_cases_per_analyzer=args.analyzer_max_cases,
            analyzer_overrides=overrides,
        )
    return ProbeAgent(judge=judge, allow_codegen=m1_codegen,
                      codegen_config=coder_cfg,
                      max_cases_per_analyzer=args.analyzer_max_cases,
                      analyzer_overrides=overrides)


def _stats_agent(args, task: T.Task, judge, coder_cfg, ctx):
    """The M2 agent; records whether it may write code in ``ctx.config``."""
    from evalrx.eval_agent import StatsAnalysisAgent

    m2_codegen = args.m2_codegen if args.m2_codegen is not None else (_is_agentic(args) or task.modality == "llm")
    ctx.config["m2_codegen"] = bool(m2_codegen)
    stats_kwargs: dict[str, Any] = {"codegen_config": coder_cfg} if m2_codegen else {}
    return StatsAnalysisAgent(
        judge=judge, figure_dir=str(ctx.figures_dir), max_signal_tools=16,
        allow_codegen=m2_codegen, **stats_kwargs,
    )


def _fix_agent(args, task: T.Task, judge, coder_cfg, gen_kwargs: dict, ctx, raw_model, deployed_spec):
    """The fix agent, scoped by the ``--fix-*`` / ``--code-only`` flags."""
    from evalrx.eval_agent import FixAgent
    from evalrx.eval_agent.stages.fix.repair_catalog import method_names

    fix_kwargs: dict[str, Any] = {}
    sampled = bool(gen_kwargs.get("do_sample"))
    # Noise model for a sampled baseline: per-case PASS RATE over k fresh
    # samples; greedy = 1, the Stage 0 answer itself.
    fix_kwargs["baseline_repeats"] = (args.fix_baseline_repeats if args.fix_baseline_repeats is not None
                                      else (5 if sampled else 1))
    if task.modality == "llm":
        floor = {"max_tokens": gen_kwargs.get("max_new_tokens", task.max_new_tokens)}
        if sampled:
            floor.update(temperature=gen_kwargs["temperature"], top_p=gen_kwargs["top_p"])
        fix_kwargs.update(baseline_generation_kwargs=floor)
    prewritten_code = (
        Path(args.fix_code_file).read_text(encoding="utf-8")
        if args.fix_code_file else ""
    )
    if deployed_spec is not None:
        fix_kwargs.update(candidate_model=raw_model, deployed_spec=deployed_spec)
    if args.code_only:
        fix_kwargs["max_judge_candidates"] = 1
    return FixAgent(
        judge=judge, max_tier=args.fix_tier, score_fn=T.score_case, run_logger=ctx.logger,
        cli_config=replace(coder_cfg, timeout_sec=420) if args.allow_codegen else None,
        allow_codegen=args.allow_codegen, run_context=ctx,
        max_validation_cases=args.fix_validation_cases, alpha=0.05,
        allow_adapted_paper_methods=args.allow_adapted_paper_methods,
        candidate_allowlist=(
            {args.fix_candidate} if args.fix_candidate
            else ({"coded_pipeline"} if args.code_only
                  else (method_names() if args.registered_repairs_only else None))
        ),
        max_repair_rounds=max(1, args.fix_repair_rounds),
        exec_timeout_sec=args.fix_exec_timeout,
        prewritten_code=prewritten_code,
        floor_candidates=() if _is_agentic(args) else ("self_consistency_5",),
        progress_callback=_print_fix_progress,
        **fix_kwargs,
    )


def _explorer(coder_cfg, ctx):
    from evalrx.agent_runtime.sandbox import ExperimentSandbox
    from evalrx.analysis import ExploratoryAnalysisAgent

    return ExploratoryAnalysisAgent(
        cli_config=coder_cfg,
        sandbox=ExperimentSandbox(
            workdir=ctx.explore_dir / "sandbox",
            cleanup=False,
        ),
        timeout_sec=900, max_attempts=3,
    )


def _run_m5(loop, report, cases, args) -> None:
    """M5 on the best hypothesis, unless the flags make it pointless.

    No M4-verified hypothesis still gets M5 + a fix attempt on the best
    unverified leads (the candidate validation on CONFIRM is the gate).
    An explicitly pre-registered fix is already the experiment the
    caller asked to validate.  Running an unrelated M5 surgery first is
    pure latency and can contend for the same GPU; it cannot influence
    the frozen candidate or its EXPLORE/CONFIRM verdict.
    """
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


def _run_fix(loop, run_dir: Path, ctx, report, cases, args, fix_tier: str) -> dict[str, Any]:
    """Run the fix ladder; the ``fix`` entry of summary.json."""
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
    return {"recommendation": str(getattr(outcome, "recommendation", "") or ""),
            "attempted": attempted}


def _price_fn(spec):
    from evalrx.models.backends.api.prices import usd

    model_id = spec.hf_repo or spec.key
    return lambda usage: usd(model_id, usage)


def _triage_explainer(args, spec):
    """The chain's judge as the scored triage's explainer (``--triage-explanations judge``).

    Triage runs no coding stage, so the judge is built without the coder checks
    the Gemini judge otherwise needs. The judge must not be the model under test."""
    from evalrx.eval_agent.triage.explanations import Explainer
    from evalrx.models.backends.api.prices import usd

    if args.triage_scheduler != "scored":
        raise SystemExit("--triage-explanations judge needs --triage-scheduler scored")
    judge_args = argparse.Namespace(**{**vars(args), "allow_codegen": False, "explore": False, "m2_codegen": False,
                                       "skip_m5": True, "coder_provider": None})
    judge, _, name, _ = build_judge(judge_args)   # the coder slot carries the judge's resolved model name
    name = name or args.judge_model
    if name and name in {spec.key, spec.hf_repo}:
        raise SystemExit(f"the triage judge ({name}) is the model under test; pick another --judge-model")
    identity = {"provider": args.judge_provider, "model": name, "effort": args.judge_effort}
    return Explainer(judge, identity=identity, price_fn=(lambda u: usd(name, u)) if name else None)


def _validation_source(args, task: T.Task, model, gen_kwargs: dict, protocol, run_dir: Path, ctx):
    """``--triage-validation-source``: cases used only for DIAGNOSTIC_VALIDATION, labelled by
    their own Stage 0 pass (``<run>/baseline_validation_source.json``). The controller and the
    judge never see them; the partition leaves out those on media DISCOVERY contains."""
    manifest = Path(args.triage_validation_source)
    if not manifest.is_file():
        raise SystemExit(f"--triage-validation-source: {manifest} is missing")
    candidates, _ = T.build_cases(task, manifest)
    found, elapsed = _discover(model, candidates, protocol, gen_kwargs, args, task)
    source = found.cases
    _mark_short_answers_stopped(task, source)
    (run_dir / "baseline_validation_source.json").write_text(json.dumps({
        "model": args.model, "dataset": task.name, "role": "validation source", "manifest": str(manifest),
        "n": len(source), "n_pass": found.n_pass, "n_fail": found.n_fail, "n_unknown": found.n_unknown,
        "accuracy": round(found.n_pass / max(1, len(source)), 4), "seconds": round(elapsed, 1),
        "label_samples": label_samples(task, args), "n_split": _split_votes(source),
        "cases": _case_labels(source),
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Validation source: PASS={found.n_pass}, FAIL={found.n_fail}, n={len(source)} "
          f"({elapsed:.0f}s) from {manifest}")
    if ctx is not None:
        ctx.config.update(triage_validation_source=str(manifest), n_validation_source_cases=len(source))
    return source, manifest


def _run_triage(args, task: T.Task, resolved: Resolved, spec, model, gen_kwargs: dict,
                cases, ctx, run_dir: Path, baseline: dict, source=None) -> int:
    """``--m1-policy triage_active``: the triage battery, rule-chosen expansions and
    follow-ups on EXPLORE, then stop.

    Same Stage 0 and the same EXPLORE/CONFIRM split as the diagnosis loop; the
    controller sees EXPLORE only and runs no M2-M5; it calls the judge only with
    ``--triage-explanations judge`` (scored scheduler: explanations for D, from
    route summaries, never a case). Outputs go to ``<run>/probing/``.

    The data roles are fixed first (``<run>/partition_bundle.json``): DISCOVERY is
    EXPLORE, DIAGNOSTIC_VALIDATION the CONFIRM cases on media EXPLORE does not
    contain, plus the cases of a validation-only *source* (``(cases, manifest)``
    from :func:`_validation_source`) under the same rule. With ``--triage-validate``
    the issues the report supports are frozen and validated once on
    DIAGNOSTIC_VALIDATION (``<run>/validation/``).
    """
    from evalrx.eval_agent.triage import explore_confirm_split, run_triage
    from evalrx.eval_agent.triage.scoring import Weights
    from evalrx.eval_agent.triage.validation import freeze_issues, partition_bundle, validate

    from .run import triage_weights

    explore, confirm = explore_confirm_split(cases, float(args.confirm_split), CONFIRM_SPLIT_SEED,
                                             by=args.confirm_split_by)
    bundle, validation = partition_bundle(explore, confirm, seed=CONFIRM_SPLIT_SEED,
                                          confirm_split=float(args.confirm_split), manifest=baseline.get("manifest"),
                                          source=source[0] if source else None,
                                          source_manifest=source[1] if source else None,
                                          split_by=args.confirm_split_by)
    if args.triage_validate and not len(validation):
        raise SystemExit("--triage-validate needs CONFIRM cases on media EXPLORE does not contain; "
                         "set --confirm-split above 0 or give a --triage-validation-source")
    bundle_row = bundle.to_dict()
    (run_dir / "partition_bundle.json").write_text(json.dumps(bundle_row, indent=2) + "\n", encoding="utf-8")
    runtime = getattr(model, "runtime", None)
    identity = {
        "spec": spec.key, "weights": spec.hf_repo, "backend": resolved.backend,
        "dtype": getattr(runtime, "dtype", args.dtype), "attn_impl": getattr(runtime, "attn_impl", None),
        "generation": gen_kwargs, "enable_thinking": bool(args.enable_thinking),
    }
    if not gen_kwargs:
        # API backends carry the sampling (and the output cap) in the runtime, not per call
        identity["runtime_generation"] = generation_settings(task, args)
    explainer = _triage_explainer(args, spec) if args.triage_explanations == "judge" else None
    if explainer is None:
        ctx.config.update(m1_policy="triage_active", judge_provider=None, judge_model=None)  # none was built
    else:
        ctx.config.update(m1_policy="triage_active", judge_provider=explainer.identity["provider"],
                          judge_model=explainer.identity["model"], triage_explanations="judge")
    extras: dict = {"modality_ablation": {"score_fn": T.score_case},
                    "prompt_contrast": {"score_fn": T.score_case}}
    sampling = _sampling_kwargs(resolved, args)
    if sampling is not None:
        # API backends ignore per-call kwargs (their runtime samples); without
        # settings here the U sampling check stays untested, never trivially clear
        extras["self_consistency"] = {"gen_kwargs": sampling}
    # what discovery and validation run with alike
    shared = dict(seed=args.triage_seed,
                  stratum="gold" if task.kind in ("yes_no", "multiple_choice_letter") else "none",
                  host_assumed_finish=bool(task.short_answer), answer_contract=bool(task.output_contract),
                  n_needs_counterfactual=task.kind in ("yes_no", "multiple_choice_letter"),
                  task_kind=task.kind, answer_fn=T.answer_key, agree_fn=T.same_answer, analyzer_kwargs=extras, price_fn=_price_fn(spec),
                  per_call_generation=sampling is not None, chunk=int(args.triage_chunk))
    report = run_triage(
        model, explore, out_dir=run_dir / "probing", model_identity=identity,
        n_fail=args.triage_n_fail, max_fail=args.triage_max_fail, max_actions=args.triage_max_actions,
        budget_sec=float(args.triage_budget_sec),
        stage0_cost={"wall_sec": baseline["seconds"], "calls": {"generate": baseline["n"]},
                     "n_cases": baseline["n"],
                     **{k: baseline[k] for k in ("api_usage", "usd") if k in baseline}},
        scheduler=args.triage_scheduler, budget_usd=float(args.triage_budget_usd), explainer=explainer,
        weights=Weights(**triage_weights(args.triage_weights)), explore_share=float(args.triage_explore_share),
        explore_from=args.triage_explore_from,
        partition={"confirm_split": float(args.confirm_split), "split_by": args.confirm_split_by,
                   "seed": CONFIRM_SPLIT_SEED,
                   "n_explore": len(explore), "n_confirm": len(confirm) if confirm is not None else 0,
                   "n_validation": len(validation), "n_validation_source": len(source[0]) if source else 0,
                   "bundle_sha1": bundle_row["sha1"],
                   "confirm_visible_to_controller": False},
        run_logger=ctx.logger, **shared,
    )
    from evalrx.eval_agent.triage.controller import report_name

    name = report_name(args.triage_scheduler, explainer is not None)
    checked = None
    if args.triage_validate:
        checked = validate(model, explore, validation, {name: freeze_issues(report)}, {name: report},
                           out_dir=run_dir / "validation", model_identity=identity, bundle=bundle,
                           budget_sec=float(args.triage_validation_budget_sec), run_logger=ctx.logger, **shared)
    ctx.finalize()
    summary = {
        "model": args.model, "spec": spec.key, "dataset": task.name, "modality": task.modality,
        "backend": resolved.backend, "n_cases": len(cases), "baseline_accuracy": baseline["accuracy"],
        "m1_policy": "triage_active", "triage": report["summary"],
        "triage_steps": [{k: s.get(k) for k in ("kind", "key", "analyzer", "status", "reused", "n_cases")}
                         for s in report["executions"]],
        "triage_stop": report["stop"], "triage_scheduler": report["scheduler"],
        "code": report["code"],
        "triage_explanations": args.triage_explanations,
        **({"validation": {"frozen_sha1": checked["frozen_sha1"], **checked["arms"][name]}} if checked else {}),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8")
    print(f"Done (triage). {run_dir / 'probing' / (name + '.md')}"
          + (f"; validation: {run_dir / 'validation' / 'validation_report.md'}" if checked else ""))
    return 0


def run(args, task: T.Task, resolved: Resolved) -> int:
    """Run one cell: Stage 0 baseline, the diagnosis loop, then M5 and the fix ladder.

    Returns the process exit code (2 when the diagnosis itself failed).
    """
    manifest = ensure_manifest(task, args)
    # Fail fast: the held-out validation manifest must exist before any model
    # or judge is loaded.
    val_manifest = ensure_val_manifest(task, args) if args.held_out else None
    if args.download_only:
        return 0
    limit = args.limit if args.limit is not None else task.default_limit
    candidates, rows = T.build_cases(task, manifest, limit)
    run_dir = run_dir_for(args, task)
    run_dir.mkdir(parents=True, exist_ok=True)
    label = model_label(resolved)
    print(f"[model] {label}; backend={resolved.backend}; benchmark={task.title}; n={len(candidates)}")

    triage = getattr(args, "m1_policy", "judge") == "triage_active"
    judge = None
    coder_provider: Any = None
    coder_model: Any = None
    coder_extra: Any = None
    if not args.baseline_only and not triage:  # the triage builds its own judge, and only for explanations
        judge, coder_provider, coder_model, coder_extra = build_judge(args)

    model, gen_kwargs, spec = _load_target(resolved, args, task, run_dir)
    protocol = task.protocol(label)
    raw_model = model
    model, deployed_spec = _deploy_baseline_spec(model, getattr(args, "baseline_spec", ""))

    ctx = None
    discovery_model = model
    if not args.baseline_only:
        from evalrx.eval_agent import RunContext
        from evalrx.eval_agent.runlog.instrumentation import InstrumentedModel

        # Create the run log before Stage-0 so each discovery request is durable
        # as it happens, including its real latency and any transport error.
        ctx = RunContext(
            # narrate: aligned live M1-M5 lines ("M1  probe ···· cycle 0 · ..."),
            # same style as examples/m1_m5/qwen_loop_claude, instead of verbose's
            # raw "[M1] probe cycle=0" one-liners.
            run_dir / "logs", narrate=True,
            config=_run_config(args, task, resolved, spec, gen_kwargs, manifest, rows, len(candidates)),
        )
        candidates = list(candidates)
        discovery_model = InstrumentedModel(
            model, ctx.logger, cycle=-1, analyzer="case_discovery",
            case_prompts={c.inputs.prompt: c.id for c in candidates},
            batch_case_ids=[c.id for c in candidates],
        )

    from evalrx.eval_agent.triage.metering import usage_counters

    usage_before = usage_counters(model)
    discovery, elapsed = _discover(discovery_model, candidates, protocol, gen_kwargs, args, task)
    cases = discovery.cases
    if ctx is not None:
        ctx.config["n_cases"] = len(cases)
    accuracy = discovery.n_pass / max(1, len(cases))
    baseline = {
        "model": args.model, "spec": spec.key, "hf_repo": spec.hf_repo, "backend": resolved.backend,
        "dataset": task.name, "task": task.kind, "n": len(cases),
        "n_pass": discovery.n_pass, "n_fail": discovery.n_fail, "n_unknown": discovery.n_unknown,
        "accuracy": round(accuracy, 4), "seconds": round(elapsed, 1),
        "generation_kwargs": gen_kwargs, "enable_thinking": bool(args.enable_thinking),
        "manifest": str(manifest), "limit": limit,
        "label_samples": label_samples(task, args), "n_split": _split_votes(cases),
        # what the provider counted for the discovery pass (API models only)
        **_usage_cost(model, usage_before),
        "cases": _case_labels(cases),
    }
    version = _api_model_version(model)
    if version:
        baseline["model_version"] = version
        print(f"[model] served version: {version}")
    (run_dir / "baseline.json").write_text(json.dumps(baseline, indent=2, ensure_ascii=False) + "\n",
                                           encoding="utf-8")
    print(f"Baseline: PASS={discovery.n_pass}, FAIL={discovery.n_fail}, UNKNOWN={discovery.n_unknown}, "
          f"accuracy={accuracy:.3f} ({elapsed:.0f}s, {elapsed / max(1, len(cases)):.1f}s/case)")
    if baseline["label_samples"] > 1:
        print(f"  labels are the majority of {baseline['label_samples']} sampled answers; "
              f"{baseline['n_split']} of {len(cases)} cases answered both ways")
    if task.modality == "llm" and not 0.15 <= accuracy <= 0.85:
        print("  NOTE accuracy is outside the usable band [0.15, 0.85]: M2 has little to contrast "
              "and the paired tests downstream are short of power (results valid, weak)")
    _mark_short_answers_stopped(task, cases)
    if ctx is None:  # --baseline-only
        print(f"Done (baseline only). {run_dir / 'baseline.json'}")
        return 0
    if not discovery.has_m4_groups:
        raise SystemExit("M4 needs both PASS and FAIL cases; adjust --limit / --seed")
    if triage:
        source = None
        if getattr(args, "triage_validation_source", None):
            source = _validation_source(args, task, model, gen_kwargs, protocol, run_dir, ctx)
        return _run_triage(args, task, resolved, spec, model, gen_kwargs, cases, ctx, run_dir, baseline,
                           source=source)

    val_cases = None
    if val_manifest is not None:
        val_cases = _label_validation_set(model, val_manifest, protocol, gen_kwargs, args, task, spec, run_dir)
        ctx.config["val_manifest"] = str(val_manifest)
        ctx.config["n_val_cases"] = len(val_cases)

    from evalrx.eval_agent import (
        CliAgentConfig,
        DiagnosisAgent,
        HypothesisTester,
        SurgeryAgent,
        VLDiagnoseLoop,
    )
    from evalrx.eval_agent.stages.experiment_writer import ExperimentWriterConfig

    coder_cfg = CliAgentConfig(provider=coder_provider, model=coder_model, timeout_sec=900,
                               extra_args=coder_extra)
    policy = None
    if _is_agentic(args):
        from evalrx.eval_agent.loop.judge import JudgePolicy

        policy = JudgePolicy(judge, max_actions=args.max_actions,
                             registry_factory=_benchmark_agentic_registry)
    loop = VLDiagnoseLoop(
        policy=policy,
        diagnosis_retries=1,
        model=model, protocol=protocol,
        probe_agent=_probe_agent(args, task, resolved, model, judge, coder_cfg, ctx.config),
        stats_agent=_stats_agent(args, task, judge, coder_cfg, ctx),
        diagnosis_agent=DiagnosisAgent(judge=judge),
        hypothesis_tester=HypothesisTester(judge=judge, min_effect=0.05),
        fix_agent=_fix_agent(args, task, judge, coder_cfg, gen_kwargs, ctx, raw_model, deployed_spec),
        max_cycles=args.max_cycles, run_logger=ctx.logger,
        confirm_split=float(args.confirm_split), confirm_split_seed=CONFIRM_SPLIT_SEED,
        # Default: the 50/50 explore/confirm design — M1-M4 work on EXPLORE
        # (M4 in-cycle), the fix ladder searches on EXPLORE, and CONFIRM is
        # reserved for the single frozen-winner validation.
        # --held-out: the SEPARATE frozen validation manifest (a disjoint
        # sample, labeled by its own Stage-0 pass) takes the val role — M4
        # verifies every proposed hypothesis ONCE there (holdout re-probe,
        # pinned analyzers) and the fix ladder searches + selects there — while
        # the main batch keeps its full explore/confirm split and the frozen
        # winner is still scored exactly once on CONFIRM.
        m4_holdout=bool(args.held_out),
        val_data=val_cases,
        surgery_agent=SurgeryAgent(judge=judge, writer_config=ExperimentWriterConfig(cli_agent=coder_cfg)),
        explorer=_explorer(coder_cfg, ctx) if args.explore else None,
        explore_dir=ctx.explore_dir,
        # RunContext(narrate=True) above owns the console; verbose would pipe
        # the stages' logger.info() stream over the aligned narration.
        verbose=False,
    )
    mode = "VLDiagnoseLoop + JudgePolicy" if policy is not None else "VLDiagnoseLoop"
    print(f"\n{'=' * 64}\n{mode}  model={args.model}  max_cycles={args.max_cycles}\n{'=' * 64}")
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
        "diagnosis_status": "error" if report.stopped_by == "diagnosis_error" else "completed",
    }
    fix_tier = effective_fix_tier(resolved.backend, args.fix_tier)
    summary["fix_tier"] = fix_tier
    if fix_tier != args.fix_tier:
        print(f"[fix] --fix-tier {args.fix_tier} clamped to {fix_tier}: the {resolved.backend} backend "
              "exposes no model internals (L3a/L3b need the white-box executors)")
    if not args.skip_fix:
        _run_m5(loop, report, cases, args)
        summary["fix"] = _run_fix(loop, run_dir, ctx, report, cases, args, fix_tier)
    ctx.finalize()
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Done. Tidy M1-M5 logs: {ctx.root.resolve()}")
    return 2 if report.stopped_by == "diagnosis_error" else 0
