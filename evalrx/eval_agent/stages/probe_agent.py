"""M1 — ProbeAgent: select suitable analyzers and execute them.

Combines tool selection (which analyzers to run for this model kind) with
execution (running each analyzer directly or inside a Docker container).

Direct mode (default, static selection)::

    agent = ProbeAgent(max_analyzers=4)
    results = agent.probe(model, data)   # {analyzer_name: Result}

LLM-guided mode — pass a judge model and a protocol::

    agent = ProbeAgent(judge=model, max_analyzers=4)
    results, schema = agent.probe_with_schema(model, data, protocol=protocol)
    print(schema.rationale)   # why these analyzers were chosen

When both *judge* and *protocol* are present the judge LLM reads the protocol
description and the catalog of available analyzers, then selects the most
relevant ones.  If the LLM call fails the agent falls back to
:class:`~evalrx.eval_agent.probe.StrategyProbe` automatically.

Docker mode::

    agent = ProbeAgent(use_docker=True, docker_image="evalrx:latest")
    results = agent.probe(model, data)

Docker mode is only engaged for **black-box-compatible** analyzers (those
whose requirements can be satisfied by a GENERATE-only API model).  Analyzers
that need white-box internals (ATTENTION, HIDDEN_STATES, …) always run
directly — they need the locally-loaded model and cannot be meaningfully
containerised without shipping the weights.

The Docker container receives a JSON payload via stdin::

    {"analyzer": "self_consistency", "params": {}, "cases": [...],
     "model_env": "GEMINI_API_KEY"}   # which env var carries the API key

and writes the analyzer findings as JSON to stdout.
"""

from __future__ import annotations

import inspect
import json
import logging
import re
import subprocess
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING, Any

from evalrx.core.capability import Capability
from evalrx.core.experiment import Experiment, ExperimentRunner
from evalrx.core.registry import registry
from evalrx.eval_agent.prompts.probe_agent import _SELECTION_PROMPT_TMPL
from evalrx.eval_agent.stages.probe import ModelKind, StrategyProbe, get_analyzer_catalog

if TYPE_CHECKING:
    from evalrx.core.analyzer import Analyzer
    from evalrx.core.case import CaseBatch
    from evalrx.core.model import Model
    from evalrx.core.result import Result
    from evalrx.eval_agent.stages.protocol import ExperimentProtocol, ProbingSchema

logger = logging.getLogger(__name__)

# Capabilities that a containerised (API-based) model can satisfy.
_BLACKBOX_CAPS = frozenset({Capability.GENERATE, Capability.LOGPROBS, Capability.TOOL_CALLS})

def _is_blackbox_compatible(analyzer_cls: type) -> bool:
    """True when the analyzer's requires are all satisfiable by a black-box API model."""
    return analyzer_cls.requires <= _BLACKBOX_CAPS


class ProbeAgent:
    """M1: select analyzers and execute them (directly or via Docker).

    Args:
        probe:             Static selector — used when *judge* is absent or the
                           LLM call fails.  Defaults to ``StrategyProbe()``.
        judge:             Any :class:`~evalrx.core.model.Model` with
                           ``Capability.GENERATE``.  When set **and** a
                           protocol is supplied to :meth:`probe`, the judge LLM
                           reads the protocol description and available analyzer
                           catalog to select analyzers dynamically.
        runner:            Executes direct (non-Docker) runs.  Defaults to a
                           fresh ``ExperimentRunner()``.
        max_analyzers:     Cap on analyzers per cycle.
        analyzer_overrides: Pre-instantiated analyzers for those requiring
                           mandatory constructor arguments.
        use_docker:        When ``True``, black-box-compatible analyzers are
                           run inside a Docker container instead of in-process.
        docker_image:      Docker image to use when *use_docker* is ``True``.
        docker_timeout:    Seconds before a Docker run is killed.
        model_env_var:     Name of the env var inside Docker that carries the
                           API key for the containerised model.
    """

    def __init__(
        self,
        probe: StrategyProbe | None = None,
        judge: "Model | None" = None,
        runner: ExperimentRunner | None = None,
        max_analyzers: int | None = None,
        analyzer_overrides: dict[str, "Analyzer"] | None = None,
        use_docker: bool = False,
        docker_image: str = "evalrx:latest",
        docker_timeout: int = 300,
        model_env_var: str = "GEMINI_API_KEY",
        allow_codegen: bool = False,
        codegen_config: "Any | None" = None,
        probe_generator: "Any | None" = None,
        whitebox_generator: "Any | None" = None,
        case_examples: tuple[int, int] = (4, 2),
        run_logger: "Any | None" = None,
        max_cases_per_analyzer: int = 0,
    ) -> None:
        self.selector = probe or StrategyProbe()
        self.judge = judge
        self.runner = runner or ExperimentRunner()
        self.max_analyzers = max_analyzers
        self._overrides: dict[str, "Analyzer"] = analyzer_overrides or {}
        self.use_docker = use_docker
        self.docker_image = docker_image
        self.docker_timeout = docker_timeout
        self.model_env_var = model_env_var
        # Set by probe() / probe_with_schema() so callers can inspect why
        # each analyzer was chosen without changing the return type of probe().
        self.last_schema: "ProbingSchema | None" = None
        # Tier (b): generate a bespoke black-box probe in a sandbox when no
        # catalog analyzer fits.  Disabled unless allow_codegen=True + a backend.
        self.allow_codegen = allow_codegen
        self._codegen_config = codegen_config
        self._probe_generator = probe_generator
        self._whitebox_generator = whitebox_generator
        # Tier (c): generated probes are cached and re-run on later cycles.
        # Each entry is (generator, probe) so cached probes re-run on the
        # generator (black-box or white-box) that created them.
        self._generated_probes: list[Any] = []
        # Optional "need_custom" hint extracted from the LLM selection response.
        self._last_need_custom: str | None = None
        # Analyzer runtime failures, accumulated ACROSS cycles (the loop reuses
        # one agent instance).  Fed back into the next selection prompt so the
        # judge stops expecting evidence from broken tools, and into the
        # tier-(b) trigger so a bespoke probe replaces the missing evidence.
        self._failed_analyzers: dict[str, str] = {}
        # How many FAIL / PASS example cases to fold into the LLM selection
        # prompt.  More examples = more specialised selection, less generalisable
        # (the generalization–specialization trade-off).  ``(0, 0)`` disables it.
        self._case_examples = case_examples
        # Optional RunLogger forwarded to the probe generators so their
        # code-writing attempts surface as "tool_codegen" events.  The loop sets this
        # when it is left unset, so wiring it here is optional for callers.
        self.run_logger = run_logger
        # Last LLM analyzer-selection judge I/O, surfaced for RunLogger.  Empty
        # when selection used the static fallback (no judge / LLM call failed).
        self.last_selection_prompt: str = ""
        self.last_selection_raw: str = ""
        #: Ceiling on how many cases ANY single analyzer receives.  0 = no limit.
        #:
        #: Capping via each analyzer's own ``max_cases`` constructor argument
        #: only reaches the analyzers that HAVE one, and the ones that dominate
        #: M1 wall-clock mostly do not: ``first_error_judge``, ``rise``,
        #: ``self_consistency`` and ``verbalized_confidence`` expose no such knob
        #: and ``trajectory_rubric`` defaults it to None, so all five iterate the
        #: whole batch one generate() at a time.  Measured on qwen3.5-2b /
        #: minervamath (136 explore cases, 20 generating analyzers): the caller
        #: capped 7 of them, observed concurrency decayed 5.8 -> 1.0 over three
        #: hours and then sat at exactly 1.0 while the uncapped ones finished
        #: alone, at ~1/10 of the endpoint's throughput.
        #:
        #: This is the backstop for that: it bounds every analyzer, including the
        #: ones with no knob to turn.  It costs statistical power, not
        #: correctness -- the same measurement, fewer of them.
        self.max_cases_per_analyzer = int(max_cases_per_analyzer or 0)
        #: Names actually truncated by the cap, for the caller to report.
        self.capped_analyzers: dict[str, tuple[int, int]] = {}

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def probe(
        self,
        model: "Model",
        data: "CaseBatch",
        *,
        protocol: "ExperimentProtocol | None" = None,
        prior_hypotheses: list[Any] | None = None,
        hint_failure_modes: list[str] | None = None,
        analyzers: list[str] | None = None,
    ) -> dict[str, "Result"]:
        """Select analyzers and run each one, returning a ``{name: Result}`` dict.

        Analyzer selection strategy (in priority order):

        1. **LLM-guided** — when a *judge* model and a *protocol* are both
           present, the judge reads the protocol description plus an
           auto-generated catalog of available analyzers and picks the most
           relevant ones.  Any *prior_hypotheses* from M3 are included as
           additional context for cycle 2+ focused probing.
        2. **Static fallback** — when no judge or no protocol, uses
           :class:`~evalrx.eval_agent.probe.StrategyProbe` with the
           optional *hint_failure_modes* list for priority-boosting (used by
           :class:`~evalrx.eval_agent.legacy.AutoDiagnoseLoop`).

        Analyzers are executed in parallel using a ``ThreadPoolExecutor`` so
        that independent analyzers do not wait on each other.

        Sets :attr:`last_schema` with the selection rationale so callers can
        inspect which analyzers ran and why without changing the return type.

        Args:
            model:               The model to probe.
            data:                Cases to run analyzers on.
            protocol:            Experiment protocol — enables LLM-guided
                                 selection when a judge is also configured.
            prior_hypotheses:    Hypotheses from M3 in prior cycles.  Passed to
                                 the judge as context for focused follow-up.
            hint_failure_modes:  Failure-mode tags for the static fallback path
                                 (e.g. from :class:`~evalrx.eval_agent.legacy.AutoDiagnoseLoop`).
            analyzers:           Exact analyzer names to run, bypassing selection
                                 entirely (still filtered for applicability).
                                 Used by the loop's held-out M4 confirmation to
                                 re-run the same set on the confirm split.
        """
        rationale: str
        self._last_need_custom = None
        if analyzers is not None:
            # Pinned re-run: the caller already knows exactly which analyzers to
            # execute (M4's held-out confirmation re-runs the last explore
            # cycle's set so the designated signals exist on the confirm split
            # too — a fresh judge selection could pick a different set and turn
            # a real signal into a spurious "designated evidence not measured").
            names = list(dict.fromkeys(analyzers))
            rationale = "Pinned by caller (held-out confirmation re-run)."
        elif self.judge is not None and protocol is not None:
            names, rationale = self._llm_select(protocol, model, prior_hypotheses, data)
        else:
            names = self.selector.select(
                model,
                max_analyzers=self.max_analyzers,
                hint_failure_modes=hint_failure_modes or None,
                data=data,
            )
            rationale = _static_rationale(names, hint_failure_modes)
        names, filtered = self._filter_applicable_names(names, model, data)
        if filtered:
            rationale += f" Filtered inapplicable analyzer(s): {', '.join(filtered)}."

        # Build (name, analyzer) pairs up front — _make_analyzer is cheap and
        # not thread-safe to call concurrently.
        tasks: list[tuple[str, "Analyzer"]] = []
        for name in names:
            analyzer = self._make_analyzer(name)
            if analyzer is not None:
                tasks.append((name, analyzer))

        results: dict[str, "Result"] = {}

        def _run_one(name: str, analyzer: "Analyzer") -> tuple[str, "Result | None"]:
            cls = type(analyzer)
            subset = self._cap_cases(name, analyzer, data)
            if self.use_docker and _is_blackbox_compatible(cls):
                return name, self._run_in_docker(name, analyzer, subset)
            return name, self._run_direct(analyzer, model, subset)

        # White-box analyzers do GPU forward passes on the SHARED local model;
        # running them in threads races on the model (accelerate device_map
        # hooks are not thread-safe → meta-tensor/dtype errors) and stacks
        # transient activations until the GPU OOMs. Only black-box analyzers
        # (GENERATE/LOGPROBS, possibly Dockerised) are safe to parallelise.
        parallel = [(n, a) for n, a in tasks if _is_blackbox_compatible(type(a))]
        serial = [(n, a) for n, a in tasks if not _is_blackbox_compatible(type(a))]

        def _record(name: str, result, exc=None) -> None:
            if exc is not None:
                warnings.warn(f"Analyzer '{name}' raised in probe run: {exc}", stacklevel=2)
            elif result is not None:
                results[name] = result

        if parallel:
            max_workers = min(len(parallel), 8)
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                futures = {pool.submit(_run_one, n, a): n for n, a in parallel}
                for future in as_completed(futures):
                    try:
                        name, result = future.result()
                        _record(name, result)
                    except Exception as exc:
                        _record(futures[future], None, exc)
        for name, analyzer in serial:  # white-box: one at a time on the GPU
            try:
                _, result = _run_one(name, analyzer)
                _record(name, result)
            except Exception as exc:
                _record(name, None, exc)

        selected = [t[0] for t in tasks]
        failed_selected = [n for n in selected if n not in results]
        for n in failed_selected:
            self._failed_analyzers.setdefault(n, "produced no result")
        if failed_selected:
            rationale += (f" Failed at runtime: {', '.join(failed_selected)} — "
                          "their evidence is missing this cycle.")
        # Tier (b)/(c): reuse + generate a bespoke probe when the catalog is
        # thin OR the selected evidence failed to materialize.
        if self.allow_codegen:
            selected += self._maybe_generate(model, data, results,
                                             failed=failed_selected, protocol=protocol)

        self.last_schema = _build_schema(selected, rationale, protocol)
        return results

    def probe_with_schema(
        self,
        model: "Model",
        data: "CaseBatch",
        *,
        protocol: "ExperimentProtocol | None" = None,
        prior_hypotheses: list[Any] | None = None,
        hint_failure_modes: list[str] | None = None,
    ) -> "tuple[dict[str, Result], ProbingSchema]":
        """Like :meth:`probe`, but also returns the :class:`~evalrx.eval_agent.protocol.ProbingSchema`.

        The schema records which analyzers were selected and why — useful for
        M2/M4 to understand the M1 reasoning without re-running selection.

        Returns:
            ``(results_dict, schema)`` — the same dict as :meth:`probe` plus a
            :class:`~evalrx.eval_agent.protocol.ProbingSchema`.
        """
        results = self.probe(
            model, data,
            protocol=protocol,
            prior_hypotheses=prior_hypotheses,
            hint_failure_modes=hint_failure_modes,
        )
        schema = self.last_schema
        if schema is None:
            from evalrx.eval_agent.stages.protocol import ProbingSchema
            schema = ProbingSchema(selected_analyzers=list(results), protocol=protocol)
        return results, schema

    # ------------------------------------------------------------------
    # LLM-guided selection
    # ------------------------------------------------------------------

    def _llm_select(
        self,
        protocol: "ExperimentProtocol",
        model: "Model",
        prior_hypotheses: list[Any] | None,
        data: "CaseBatch | None" = None,
    ) -> tuple[list[str], str]:
        """Ask the judge LLM to choose analyzers from the available catalog.

        The selection prompt is grounded in the *observed* cases (a digest of
        real PASS/FAIL prompts and the model's actual answers) so the judge
        picks analyzers for what actually failed, not just the abstract protocol.

        Returns ``(selected_names, rationale_string)``.  Falls back to static
        selection on any error.
        """
        assert self.judge is not None  # caller guarantees this

        kind = self.selector.detect_kind(model, data)
        catalog = self._offerable(get_analyzer_catalog(model), data, model)
        if not catalog:
            return self._static_fallback(model, data), "no analyzers available"

        max_n = self.max_analyzers if self.max_analyzers is not None else len(catalog)

        analyzer_lines = "\n".join(
            f"  - {name}: {desc}" for name, desc in catalog.items()
        )

        task_domain_section = (
            f"\nTASK DOMAIN: {protocol.task_domain}\n" if protocol.task_domain else ""
        )
        success_criteria_section = (
            f"\nSUCCESS CRITERIA: {protocol.success_criteria}\n"
            if protocol.success_criteria
            else ""
        )
        failure_patterns_section = (
            f"\nADDITIONAL OBSERVATIONS: {protocol.failure_patterns}\n"
            if protocol.failure_patterns
            else ""
        )

        prior_section = ""
        if prior_hypotheses:
            lines = []
            for h in prior_hypotheses:
                stmt = getattr(h, "statement", str(h))
                status = ""
                s = getattr(h, "status", None)
                if s is not None:
                    status = f" [{getattr(s, 'value', str(s))}]"
                lines.append(f"  - {stmt}{status}")
                design = getattr(h, "test_design", "")
                if design:
                    lines.append(f"    → proposed test: {design}")
            prior_section = (
                "\nPRIOR HYPOTHESES FROM THIS INVESTIGATION "
                "(select analyzers that collect the evidence their proposed "
                "tests call for):\n"
                + "\n".join(lines)
                + "\n"
            )

        failed_section = ""
        if self._failed_analyzers:
            failed_lines = "\n".join(
                f"  - {n}: {err}" for n, err in self._failed_analyzers.items())
            failed_section = (
                "\nANALYZERS THAT FAILED AT RUNTIME IN THIS INVESTIGATION "
                "(selecting them again will NOT produce evidence — pick "
                "alternatives or request a custom probe):\n" + failed_lines + "\n")

        n_fail, n_pass = self._case_examples
        cases_section = _summarize_cases(data, max_fail=n_fail, max_pass=n_pass)

        prompt = _SELECTION_PROMPT_TMPL.format(
            description=protocol.description,
            task_domain_section=task_domain_section,
            success_criteria_section=success_criteria_section,
            failure_patterns_section=failure_patterns_section,
            cases_section=cases_section,
            model_kind=kind.value,
            n_available=len(catalog),
            analyzer_list=analyzer_lines,
            failed_section=failed_section,
            prior_hypotheses_section=prior_section,
            max_n=max_n,
        )

        self.last_selection_prompt = prompt
        self.last_selection_raw = ""
        try:
            sig = inspect.signature(self.judge.generate)
            if "temperature" in sig.parameters:
                raw = self.judge.generate(prompt, temperature=0)
            else:
                raw = self.judge.generate(prompt)
        except Exception as exc:
            logger.warning("ProbeAgent LLM selection failed (%s) — using static fallback", exc)
            return self._static_fallback(model, data), "static fallback (LLM call failed)"

        self.last_selection_raw = str(raw)
        return self._parse_llm_response(str(raw), set(catalog), max_n, model, data)

    def _parse_llm_response(
        self,
        raw: str,
        valid_names: set[str],
        max_n: int,
        model: "Model",
        data: "CaseBatch | None" = None,
    ) -> tuple[list[str], str]:
        """Extract analyzer names and rationale from the LLM JSON response.

        Strips ``<think>…</think>`` blocks emitted by reasoning models (e.g.
        Qwen3) before searching for the JSON payload so that greedy matching
        does not swallow reasoning text as part of the JSON.
        """
        # Remove <think>…</think> reasoning blocks before JSON search
        cleaned = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()

        # An empty response (e.g. agy rate-limited/quota-exhausted) is distinct
        # from an unparseable one — surface that so the fallback is diagnosable.
        if not cleaned:
            logger.warning("ProbeAgent: LLM returned an empty response — using static fallback")
            return self._static_fallback(model, data), "static fallback (LLM returned empty response)"

        # Use a non-greedy search that stops at the first closing brace so we
        # don't accidentally span multiple JSON objects.
        match = re.search(r"\{[^{}]*\}", cleaned)
        if match:
            try:
                payload = json.loads(match.group())
                # Normalise to lowercase so the LLM can return "POPE" or "Pope"
                names = [
                    n.lower() for n in payload.get("analyzers", [])
                    if n.lower() in valid_names
                ]
                rationale = str(payload.get("rationale", "LLM-selected"))
                need = payload.get("need_custom")
                if isinstance(need, str) and need.strip():
                    self._last_need_custom = need.strip()
                if names:
                    return names[:max_n], rationale
            except json.JSONDecodeError:
                pass

        # Soft parse: case-insensitive scan for any known analyzer name
        cleaned_lower = cleaned.lower()
        found = [n for n in sorted(valid_names) if n in cleaned_lower]
        if found:
            logger.warning("ProbeAgent: LLM response not valid JSON — extracted names by text scan")
            return found[:max_n], "LLM-selected (text-extracted)"

        logger.warning("ProbeAgent: could not parse LLM response — using static fallback")
        return self._static_fallback(model, data), "static fallback (LLM parse failed)"

    def _static_fallback(self, model: "Model", data: "CaseBatch | None" = None) -> list[str]:
        return self.selector.select(model, max_analyzers=self.max_analyzers, data=data)

    def _filter_applicable_names(
        self,
        names: list[str],
        model: "Model",
        data: "CaseBatch",
    ) -> tuple[list[str], list[str]]:
        """Drop analyzers whose data preconditions are not met.

        LLM selection can choose semantically plausible tools that require a
        specific case shape.  Filtering here prevents invalid runs such as POPE
        on open-ended VQA prompts without yes/no gold labels.
        """
        max_n = self.max_analyzers if self.max_analyzers is not None else len(names)
        selected: list[str] = []
        filtered: list[str] = []
        filtered_seen: set[str] = set()

        def _consider(candidate: str) -> None:
            if candidate in selected:
                return
            # Both selection paths (judge and static) converge here, so this is
            # where "the data is the wrong shape" and "the collaborator was never
            # injected" have to be caught. Previously neither was: an analyzer
            # that could not be built was selected anyway and then dropped at
            # instantiation with a warning, having already cost a slot.
            if not _analyzer_data_preconditions_met(candidate, data, model) \
                    or self._needs_injection(candidate):
                if candidate not in filtered_seen:
                    filtered.append(candidate)
                    filtered_seen.add(candidate)
                return
            selected.append(candidate)

        if max_n <= 0:
            return selected, filtered

        for name in names:
            _consider(name)
            if len(selected) >= max_n:
                return selected, filtered

        # Backfill from the static selector so one bad LLM choice does not leave
        # M1 empty when other compatible analyzers can run.
        for name in self.selector.select(model, max_analyzers=None, data=data):
            _consider(name)
            if len(selected) >= max_n:
                break
        return selected, filtered

    # ------------------------------------------------------------------
    # Tier (b)/(c): probe generation
    # ------------------------------------------------------------------

    def _maybe_generate(
        self,
        model: "Model",
        data: "CaseBatch",
        results: "dict[str, Result]",
        failed: "list[str] | None" = None,
        protocol: "ExperimentProtocol | None" = None,
    ) -> list[str]:
        """Reuse cached generated probes, then generate one if the catalog is thin.

        Generator dispatch: when the requested probe targets the model's
        *internals* (attention/layers/hidden states/…) and the model is
        white-box, the capture-then-compute
        :class:`~evalrx.eval_agent.stages.whitebox_probe_generator.WhiteboxProbeGenerator`
        is used; otherwise the black-box output-probe generator.

        Mutates *results* in place (adds ``generated:<name>`` entries) and returns
        the list of probe names added, for inclusion in the schema.
        """
        added: list[str] = []

        # Tier (c): re-run cached generated probes on the generator that made them.
        for gen, probe in self._generated_probes:
            r = gen.run_cached(probe, model, data)
            if r is not None:
                results[r.analyzer] = r
                added.append(r.analyzer)

        # Tier (b): generate a new probe when the judge asked for one, when the
        # judge's SELECTED analyzers failed at runtime (their evidence is
        # missing either way), or when nothing produced any output.
        need = self._last_need_custom
        should_generate = (
            (bool(need) or bool(failed)) and not self._generated_probes
        ) or not results
        if not should_generate:
            return added

        if need:
            goal = need
        elif failed:
            goal = (
                f"Selected analyzers ({', '.join(failed)}) failed to run; collect "
                "equivalent evidence for the failure mechanism: "
                + (protocol.description if protocol is not None
                   else "see the experiment protocol.")
            )
        else:
            goal = "Probe the model outputs for the failure described in the protocol."
        generator = self._dispatch_generator(goal, model)
        if generator is None:
            return added
        name = f"probe{len(self._generated_probes) + 1}"
        result, probe = generator.generate(goal, model, data, name=name)
        if result is not None:
            results[result.analyzer] = result
            added.append(result.analyzer)
        if probe is not None:
            self._generated_probes.append((generator, probe))
        return added

    # Mechanism wording that calls for internal (white-box) evidence.
    _WHITEBOX_NEED_KEYWORDS = (
        "attention", "layer", "hidden", "logit", "internal", "sink",
        "head", "patch", "token mass", "image token",
    )

    def _dispatch_generator(self, need: str, model: "Model") -> "Any | None":
        """Choose the white-box or black-box generator for *need*."""
        lowered = need.lower()
        wants_internals = any(k in lowered for k in self._WHITEBOX_NEED_KEYWORDS)
        is_whitebox = Capability.ATTENTION in getattr(model, "capabilities", frozenset())
        if wants_internals and is_whitebox:
            gen = self._get_whitebox_generator()
            if gen is not None:
                return gen
        return self._get_generator()

    def _get_generator(self) -> "Any | None":
        """Lazily build the black-box probe generator from the judge / config."""
        if self._probe_generator is not None:
            return self._attach_logger(self._probe_generator)
        if self.judge is None and self._codegen_config is None:
            return None
        from evalrx.eval_agent.stages.probe_generator import ProbeGenerator
        gen = ProbeGenerator(
            judge=self.judge, cli_config=self._codegen_config, run_logger=self.run_logger
        )
        if not gen.available:
            return None
        self._probe_generator = gen
        return gen

    def _get_whitebox_generator(self) -> "Any | None":
        """Lazily build the white-box (capture-then-compute) probe generator."""
        if self._whitebox_generator is not None:
            return self._attach_logger(self._whitebox_generator)
        if self.judge is None and self._codegen_config is None:
            return None
        from evalrx.eval_agent.stages.whitebox_probe_generator import (
            WhiteboxProbeGenerator,
        )
        gen = WhiteboxProbeGenerator(
            judge=self.judge, cli_config=self._codegen_config, run_logger=self.run_logger
        )
        if not gen.available:
            return None
        self._whitebox_generator = gen
        return gen

    def _attach_logger(self, gen: "Any") -> "Any":
        """Ensure a pre-built / cached generator points at the current RunLogger."""
        if self.run_logger is not None and getattr(gen, "run_logger", None) is None:
            try:
                gen.run_logger = self.run_logger
            except Exception:  # noqa: BLE001
                pass
        return gen

    # ------------------------------------------------------------------
    # Execution strategies
    # ------------------------------------------------------------------

    def _cap_cases(self, name: str, analyzer: "Analyzer", data: "CaseBatch"):
        """Bound how many cases *analyzer* receives, keeping the labels balanced.

        Three things this must not do:

        * **Fight the analyzer's own knob.**  If it already caps itself at or
          below the ceiling, it is left alone — otherwise a 12-case analyzer
          configured deliberately would silently be handed 32.
        * **Destroy the contrast.**  M1 exists to compare PASS against FAIL, so a
          head-truncation is not acceptable: on a batch that happens to open with
          a run of one label it would hand the analyzer a single class and the
          result would read as a null rather than as a missing comparison.  Each
          label group is subsampled proportionally.
        * **Vary between runs.**  Even striding, no RNG: two runs on the same
          batch must select the same cases or their numbers are not comparable.
        """
        cap = self.max_cases_per_analyzer
        if cap <= 0 or data is None or len(data) <= cap:
            return data
        own = getattr(analyzer, "max_cases", None)
        if isinstance(own, int) and 0 < own <= cap:
            return data

        from evalrx.core.case import CaseBatch

        groups: dict[Any, list] = {}
        for case in data:
            groups.setdefault(getattr(case, "label", None), []).append(case)

        picked: list = []
        # Largest group first, so integer truncation costs the class that can
        # best afford it rather than wiping out a small one.
        order = sorted(groups.items(), key=lambda kv: -len(kv[1]))
        remaining = cap
        for index, (_, members) in enumerate(order):
            share = remaining if index == len(order) - 1 else round(
                cap * len(members) / len(data))
            share = max(1, min(share, len(members), remaining))
            if len(members) <= share:
                picked.extend(members)
            else:  # even stride across the group, not its head
                step = len(members) / share
                picked.extend(members[int(i * step)] for i in range(share))
            remaining -= share
            if remaining <= 0:
                break

        self.capped_analyzers[name] = (len(data), len(picked))
        return CaseBatch(picked)

    def _run_direct(
        self,
        analyzer: "Analyzer",
        model: "Model",
        data: "CaseBatch",
    ) -> "Result | None":
        # Wrap the model so every generate/forward/logprobs/chat call THIS
        # analyzer makes is durably recorded (see model_instrumentation.py) —
        # including calls an analyzer makes several times per case and then
        # reduces to one derived score, which otherwise leave no trace. A
        # fresh proxy per call keeps call_index scoped to (cycle, analyzer);
        # __repr__ forwards to the wrapped model so Experiment.fingerprint()
        # and Result.model stay unaffected by the wrapping. Only reaches
        # in-process (non-Docker) analyzers — Docker-mode calls happen inside
        # a subprocess and are out of reach of this wrapper.
        if self.run_logger is not None:
            from evalrx.eval_agent.model_instrumentation import InstrumentedModel

            # Best-effort (prompt -> case_id) for THIS analyzer's batch, so a
            # call that reuses the case's own prompt unmodified can be tied
            # back to it; batch_case_ids is the fallback for calls that
            # rewrite the prompt (format_sensitivity, cot_faithfulness, ...).
            batch_case_ids = [str(getattr(c, "id", "")) for c in data]
            case_prompts = {
                str(prompt): case_id
                for c, case_id in zip(data, batch_case_ids)
                if (prompt := getattr(getattr(c, "inputs", None), "prompt", None)) is not None
            }
            model = InstrumentedModel(
                model, self.run_logger,
                cycle=getattr(self.run_logger, "current_cycle", -1),
                analyzer=analyzer.name,
                case_prompts=case_prompts,
                batch_case_ids=batch_case_ids,
            )
        exp = Experiment(model=model, analyzer=analyzer, data=data)
        try:
            return self.runner.run(exp)
        except Exception as exc:
            self._failed_analyzers[analyzer.name] = f"{type(exc).__name__}: {str(exc)[:200]}"
            warnings.warn(
                f"Analyzer '{analyzer.name}' raised during direct run: {exc}",
                stacklevel=3,
            )
            return None

    def _run_in_docker(
        self,
        name: str,
        analyzer: "Analyzer",
        data: "CaseBatch",
    ) -> "Result | None":
        """Run *analyzer* inside a Docker container, returning the Result or None."""
        from evalrx.core.result import Result

        payload = json.dumps({
            "analyzer": name,
            "params": analyzer.get_params(),
            "cases": _serialize_cases(data),
            "model_env": self.model_env_var,
        })
        cmd = [
            "docker", "run", "--rm", "-i",
            "--env", self.model_env_var,
            self.docker_image,
            "python", "-m", "evalrx.agent_runtime._docker_runner",
        ]
        try:
            proc = subprocess.run(
                cmd,
                input=payload.encode(),
                capture_output=True,
                timeout=self.docker_timeout,
            )
        except FileNotFoundError:
            warnings.warn(
                "Docker is not available; falling back to direct execution "
                f"for analyzer '{name}'.",
                stacklevel=3,
            )
            return None
        except subprocess.TimeoutExpired:
            warnings.warn(f"Docker run for '{name}' timed out.", stacklevel=3)
            return None

        if proc.returncode != 0:
            warnings.warn(
                f"Docker run for '{name}' failed (exit {proc.returncode}): "
                f"{proc.stderr.decode()[:300]}",
                stacklevel=3,
            )
            return None

        try:
            raw = json.loads(proc.stdout)
            return Result(
                analyzer=name,
                model=f"docker:{self.docker_image}",
                findings=raw.get("findings", {}),
                metadata={"docker": True, "image": self.docker_image},
                cases=data,
            )
        except Exception as exc:
            warnings.warn(f"Could not parse Docker output for '{name}': {exc}", stacklevel=3)
            return None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _offerable(self, catalog: "dict[str, str]", data: "Any" = None,
                   model: "Model | None" = None) -> "dict[str, str]":
        """Drop analyzers the judge could pick but this run could never use.

        Capability + modality matching answers "does the MODEL provide what this
        needs". Two things it cannot answer, both of which produced real damage
        on a single-turn text run:

        1. **Is the data the right shape?** Every analyzer in ``analyzers/agent/``
           declares no capability (correctly — they read trajectories, sometimes
           with no model at all), so all six matched a plain QA batch.
           ``first_error_judge`` was duly selected, ran, and reported
           ``n_trajectories: 0``, which M2 then listed as a healthy metric.
           The same hole exists one axis over, for modality: the registry match
           is an intersection against what the MODEL declares, so an omni model
           evaluated on an audio benchmark still declares image and ``pope`` /
           ``chair`` were duly offered a batch with no images in it.
           ``requires_modalities`` is that gate.
        2. **Did the caller inject the collaborator?** ``reliability_probe``
           wants ``runs_fn``, ``tool_shap`` a tool runner, ``trajectory_rubric``
           a judge. These were offered every run and skipped at instantiation
           with a warning, spending a selection slot a runnable analyzer could
           have had.

        An override supplies exactly that missing collaborator, so anything in
        ``analyzer_overrides`` survives the second check.
        """
        from evalrx.eval_agent.stages.probe import StrategyProbe, _carries_trajectories

        has_traj = data is not None and _carries_trajectories(data)
        starved = StrategyProbe.slot_starved(set(catalog), data, model)
        keep: dict[str, str] = {}
        for name, desc in catalog.items():
            cls = registry.analyzers.get(name)
            if getattr(cls, "requires_trajectories", False) and not has_traj:
                logger.debug(
                    "not offering analyzer '%s': it reads agent trajectories and "
                    "this batch carries none", name,
                )
                continue
            if name in starved:
                logger.debug(
                    "not offering analyzer '%s': it needs one of %s filled and this "
                    "batch fills none of them", name,
                    sorted(getattr(cls, "requires_modalities", frozenset())),
                )
                continue
            needs = self._needs_injection(name)
            if needs:
                logger.debug(
                    "not offering analyzer '%s': needs %s, which must be injected "
                    "via analyzer_overrides", name, needs,
                )
                continue
            keep[name] = desc
        return keep

    def _needs_injection(self, name: str) -> "list[str]":
        """Constructor arguments the caller must supply; empty when buildable.

        An entry in ``analyzer_overrides`` IS that supply, so an overridden
        analyzer always reads as buildable.
        """
        import inspect

        if name in self._overrides:
            return []
        cls = registry.analyzers.get(name) if registry.analyzers.has(name) else None
        if cls is None:
            return []
        try:
            params = inspect.signature(cls.__init__).parameters
        except (TypeError, ValueError):
            return []
        return [
            n for n, p in params.items()
            if n != "self"
            and p.default is inspect.Parameter.empty
            and p.kind not in (inspect.Parameter.VAR_POSITIONAL,
                               inspect.Parameter.VAR_KEYWORD)
        ]

    def _make_analyzer(self, name: str) -> "Analyzer | None":
        if name in self._overrides:
            return self._overrides[name]
        cls = registry.analyzers.get(name)
        try:
            return cls()
        except TypeError as exc:
            warnings.warn(
                f"Skipping analyzer '{name}': cannot instantiate with default args "
                f"({exc}). Pass an instance via analyzer_overrides.",
                stacklevel=3,
            )
            return None

    def detect_kind(self, model: "Model", data: "CaseBatch | None" = None) -> ModelKind:
        """Delegate to the inner :class:`~evalrx.eval_agent.probe.StrategyProbe`."""
        return self.selector.detect_kind(model, data)


def _clip(text: str, max_chars: int) -> str:
    """Collapse whitespace and truncate *text* for compact prompt inclusion."""
    flat = re.sub(r"\s+", " ", str(text)).strip()
    return flat if len(flat) <= max_chars else flat[:max_chars] + "…"


def _summarize_cases(
    data: "CaseBatch | None",
    *,
    max_fail: int = 4,
    max_pass: int = 2,
    max_chars: int = 160,
) -> str:
    """Digest observed PASS/FAIL cases (prompt + expected + model answer).

    Folded into the M1 selection prompt so the judge selects analyzers grounded
    in concrete failures.  Returns ``""`` when there are no labeled cases or the
    feature is disabled (``max_fail == max_pass == 0``).
    """
    from evalrx.core.case import Label

    if data is None or (max_fail <= 0 and max_pass <= 0):
        return ""

    counts = {"pass": 0, "fail": 0, "unknown": 0}
    fails: list[Any] = []
    passes: list[Any] = []
    for c in data:
        label = getattr(c, "label", None)
        key = getattr(label, "value", None)
        if key in counts:
            counts[key] += 1
        if label == Label.FAIL and len(fails) < max_fail:
            fails.append(c)
        elif label == Label.PASS and len(passes) < max_pass:
            passes.append(c)

    # No concrete PASS/FAIL examples to show (e.g. all UNKNOWN) → no useful digest.
    if not fails and not passes:
        return ""

    lines = [
        f"\nOBSERVED CASES (PASS={counts['pass']} FAIL={counts['fail']} "
        f"UNKNOWN={counts['unknown']}) — select analyzers for what actually failed:"
    ]

    def _emit(case: Any, tag: str) -> None:
        inp = getattr(case, "inputs", None)
        prompt = getattr(inp, "prompt", "") if inp is not None else ""
        observed = (
            getattr(case, "observed", None)
            or getattr(case, "metadata", {}).get("discovery_observed", "")
        )
        lines.append(f"  [{tag}] prompt: {_clip(prompt, max_chars)}")
        if getattr(case, "expected", None) is not None:
            lines.append(f"         expected: {_clip(case.expected, max_chars)}")
        lines.append(f"         model answered: {_clip(observed, max_chars)}")

    for c in fails:
        _emit(c, "FAIL")
    for c in passes:
        _emit(c, "PASS")
    return "\n".join(lines) + "\n"


def _static_rationale(names: list[str], hints: list[str] | None) -> str:
    parts = []
    if hints:
        parts.append(f"failure-mode hints: {', '.join(hints)}")
    return (
        f"Selected {len(names)} analyzer(s) ({', '.join(names)}) "
        + ("guided by " + " + ".join(parts) if parts else "by capability matching")
        + "."
    )


def _build_schema(
    selected: list[str],
    rationale: str,
    protocol: "ExperimentProtocol | None",
) -> "ProbingSchema":
    from evalrx.eval_agent.stages.protocol import ProbingSchema
    return ProbingSchema(
        selected_analyzers=selected,
        rationale=rationale,
        protocol=protocol,
    )


def _serialize_cases(data: "CaseBatch") -> list[dict[str, Any]]:
    """Minimal JSON-serialisable representation of a CaseBatch for Docker."""
    out = []
    for case in data:
        inp = getattr(case, "inputs", None)
        out.append({
            "id": case.id,
            "prompt": str(inp.prompt) if inp else "",
            "label": getattr(case.label, "value", None) if hasattr(case, "label") else None,
            "metadata": getattr(case, "metadata", {}),
        })
    return out


def _analyzer_data_preconditions_met(name: str, data: "CaseBatch",
                                     model: "Model | None" = None) -> bool:
    cls = registry.analyzers.get(name) if registry.analyzers.has(name) else None
    if cls is not None and getattr(cls, "requires_modalities", None):
        # All three selection paths (static, judge, pinned) converge here, so
        # this is where the modality gate has to hold for the pinned one too.
        from evalrx.eval_agent.stages.probe import StrategyProbe

        if name in StrategyProbe.slot_starved({name}, data, model):
            return False
    if cls is not None and getattr(cls, "requires_trajectories", False):
        # The agent lane declares no capability — correctly, since these read
        # trajectories and often take model=None — so capability matching let all
        # of them through to single-turn batches. This is the precondition that
        # actually applies: are there trajectories to read?
        from evalrx.eval_agent.stages.probe import _carries_trajectories

        return _carries_trajectories(data)
    if name == "pope":
        return any(
            str((getattr(case, "metadata", {}) or {}).get("pope_label", "")).lower()
            in {"yes", "no"}
            for case in data
        )
    if name == "chair":
        return any(
            bool((getattr(case, "metadata", {}) or {}).get("gt_objects"))
            for case in data
        )
    return True
