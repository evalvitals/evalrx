"""VLMProbe lens, attention, and causal methods through EvalRX's analyzer API.

The required ``method`` makes this an opt-in white-box analysis: agents cannot
accidentally launch integrations, calibration, or interventions with guessed
targets. Existing EvalRX analyzers retain their names and scientific semantics.

References:
- VLMProbe: https://github.com/gexinh/vlm-probing
- Method papers and portability contracts: docs/vlmprobe.md
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from evalrx.core.analyzer import Analyzer
from evalrx.core.capability import Capability
from evalrx.core.case import CaseBatch
from evalrx.core.registry import register_analyzer

from .bridge import make_metric, native_inputs, resolve_prober
from .catalog import METHOD_SPECS
from .result import VLMProbeResult, json_safe, summarize_probe


@register_analyzer("vlmprobe")
class VLMProbeAnalyzer(Analyzer):
    """One opt-in integration of all 24 VLMProbe public method factories.

    Configure factory arguments in ``method_config``; provide reference inputs,
    targets and other execution arguments in ``run_config``. A run-config
    callable receives each FailureCase, allowing different references/metrics
    per case. Calibration is always explicit and outside ``run``.

    ``requires`` is only EvalRX's coarse white-box gate. Exact architecture,
    gradient, editable-attention and graph contracts are checked by VLMProbe.
    Use ``prober`` plus ``input_builder(model, case)`` for explicit adapters.
    """

    name = "vlmprobe"
    requires = frozenset({Capability.HIDDEN_STATES})
    applies_to_modalities = frozenset({"text", "image"})

    def __init__(
        self,
        method: str,
        *,
        method_config: dict | None = None,
        run_config: Any = None,
        metric: Any = None,
        prober: Any = None,
        input_builder: Any = None,
        configured_method: Any = None,
        calibration_dir: str | None = None,
        output_dir: str | None = None,
        max_cases: int = 1,
        operation: str = "run",
    ) -> None:
        super().__init__(
            method=method, method_config=method_config, run_config=run_config,
            metric=metric, prober=prober, input_builder=input_builder,
            configured_method=configured_method, calibration_dir=calibration_dir,
            output_dir=output_dir, max_cases=max_cases, operation=operation,
        )
        self._validate()

    def _validate(self):
        if self.method not in METHOD_SPECS:
            raise ValueError(f"Unknown VLMProbe method {self.method!r}; choose {sorted(METHOD_SPECS)}")
        if type(self.max_cases) is not int or self.max_cases < 0:
            raise ValueError("max_cases must be a nonnegative integer; 0 analyzes every case")
        if self.operation not in {"run", "sweep", "generate", "evaluate"}:
            raise ValueError("operation must be run, sweep, generate, or evaluate")
        allowed = {
            "sweep": {"causal.path"},
            "generate": {"causal.steer", "causal.vsv"},
            "evaluate": {"causal.eap_ig"},
        }
        if self.operation != "run" and self.method not in allowed[self.operation]:
            raise ValueError(f"operation={self.operation!r} is not supported for {self.method}")
        if self.method_config is not None and not isinstance(self.method_config, Mapping):
            raise TypeError("method_config must be a mapping of VLMProbe factory arguments")
        if self.run_config is not None and not callable(self.run_config) and not isinstance(self.run_config, Mapping):
            raise TypeError("run_config must be a mapping or callable(case) -> mapping")
        spec = METHOD_SPECS[self.method]
        if self.calibration_dir is not None and not spec.calibrated:
            raise ValueError("calibration_dir is only applicable to trained lens methods")
        if self.configured_method is not None:
            if not spec.calibrated:
                raise ValueError("configured_method is only supported for explicitly fitted lens methods")
            if self.calibration_dir is not None:
                raise ValueError("Provide configured_method or calibration_dir, not both")
            if getattr(self.configured_method, "name", None) != spec.factory:
                raise ValueError("configured_method must match the declared lens method")
            if getattr(self.configured_method, "probe", None) is None:
                raise ValueError("configured_method must be a model-bound VLMProbe lens")

    def _check_capabilities(self, model):
        explicit = self.prober or getattr(self.configured_method, "probe", None)
        if explicit is None:
            super()._check_capabilities(model)
        # Explicit adapters may expose more than EvalRX's coarse capability
        # vocabulary (e.g. a ViT classification model). Their contracts are
        # validated by the VLMProbe factory, not inferred from detached traces.

    def _prober(self, model):
        explicit = self.prober or getattr(self.configured_method, "probe", None)
        return resolve_prober(model, explicit)

    def bind(self, model):
        """Configure/load a bound method; callers can explicitly fit/save it.

        For trained lenses: bind, fit on independent calibration inputs, then
        reuse the method via ``configured_method=...`` or save/load its artifact.
        VSV directions instead depend on each case's positive/negative contexts.
        """
        self._validate()
        probe = self._prober(model)
        if self.configured_method is not None:
            if getattr(self.configured_method, "probe", probe) is not probe:
                raise ValueError("configured_method and prober must share the same model-bound Prober")
            return self.configured_method
        spec = METHOD_SPECS[self.method]
        if self.method == "causal.vsv":
            raise ValueError("VSV directions require positive/negative contexts; supply them in run_config")
        if spec.calibrated and self.calibration_dir is None:
            # bind is a training on-ramp; run checks this before binding.
            return getattr(getattr(probe, spec.family), spec.factory)(**dict(self.method_config or {}))
        method = getattr(getattr(probe, spec.family), spec.factory)(**dict(self.method_config or {}))
        if self.calibration_dir is not None:
            if not spec.calibrated:
                raise ValueError("calibration_dir is only applicable to trained lens methods")
            method.load(self.calibration_dir)
        return method

    def _run(self, model, cases):
        self._validate()
        spec = METHOD_SPECS[self.method]
        if spec.calibrated and self.calibration_dir is None and self.configured_method is None:
            raise ValueError(
                f"{self.method} requires calibration_dir=... or configured_method=... fitted on "
                "independent calibration inputs. Evaluation cases are never used to fit a lens."
            )
        selected = CaseBatch(cases.stratified_head(self.max_cases))
        if not len(selected):
            raise ValueError("VLMProbe analysis requires at least one case")
        probe = self._prober(model)
        # Do not resolve a second adapter in bind: configured hooks must share
        # one model-bound Prober. Make a local method without mutating parameters.
        method = None
        if self.method != "causal.vsv":
            method = self.configured_method
            if method is not None and getattr(method, "probe", probe) is not probe:
                raise ValueError("configured_method and prober must share the same Prober")
            if method is None:
                method = getattr(getattr(probe, spec.family), spec.factory)(**dict(self.method_config or {}))
                if self.calibration_dir is not None:
                    if not spec.calibrated:
                        raise ValueError("calibration_dir is only applicable to trained lens methods")
                    method.load(self.calibration_dir)
        results, summaries, evaluations = [], [], {}
        for index, case in enumerate(selected):
            inputs = self.input_builder(model, case) if self.input_builder else native_inputs(model, case.inputs)
            options = self.run_config(case) if callable(self.run_config) else self.run_config
            if options is not None and not isinstance(options, Mapping):
                raise TypeError("run_config callback must return a mapping")
            options = dict(options or {})
            for key in ("source", "donor", "target_inputs", "positive", "negative"):
                if key in options:
                    options[key] = native_inputs(model, options[key])
            metric = make_metric(options.pop("metric", self.metric))
            if spec.needs_metric and self.operation != "generate":
                if metric is None:
                    raise ValueError(f"{self.method} requires an explicit metric or per-case run_config['metric']")
                options["metric"] = metric
            if self.method in {"causal.patch", "causal.attribute"}:
                self._pair(options, "source")
            elif self.method == "causal.path":
                self._pair(options, "donor")
            elif self.method == "causal.eap_ig":
                key = "donor" if (self.method_config or {}).get("graph", "activation") == "transformer" else "source"
                self._pair(options, key)
            if self.method == "lens.patchscope" and "target_inputs" not in options:
                raise ValueError("lens.patchscope requires target_inputs in run_config")
            if self.method == "causal.vsv":
                if "positive" not in options or "negative" not in options:
                    raise ValueError("causal.vsv requires positive and negative contexts in run_config")
                config = dict(self.method_config or {})
                if "strength" not in config:
                    raise ValueError("causal.vsv requires an explicit model-specific strength in method_config")
                strength, tokens = config.pop("strength"), config.pop("tokens", "prediction")
                visual = probe.causal.vsv(options.pop("positive"), options.pop("negative"), **config)
                method = visual.configure(strength=strength, tokens=tokens)
                options["direction"] = visual.directions
            if self.method == "attention.dtd_lrp":
                if "target" not in options:
                    raise ValueError("attention.dtd_lrp requires an explicit classification target in run_config")
                values = inputs.kwargs if hasattr(inputs, "kwargs") else inputs
                if "pixel_values" not in values:
                    raise ValueError("DTD/LRP requires pixel_values and an eligible native ViT classifier")
                native = method.run(values["pixel_values"], **options)
            elif self.operation == "evaluate":
                if (self.method_config or {}).get("graph") != "transformer":
                    raise ValueError("EAP faithfulness evaluation requires graph='transformer' (audited GPT-2)")
                if "budgets" not in options:
                    raise ValueError("EAP faithfulness evaluation requires retained-edge budgets in run_config")
                evaluation_options = {
                    key: options.pop(key) for key in ("budgets", "rankings", "strategy", "epsilon")
                    if key in options
                }
                native = method.run(inputs, **options)
                evaluations[index] = method.evaluate(
                    inputs, attribution=native, **evaluation_options, **options
                )
            else:
                native = getattr(method, self.operation)(inputs, **options)
            summaries.append(summarize_probe(native, case.id))
            if self.method == "causal.vsv":
                native.metadata["visual_direction"] = json_safe(visual.metadata)
            results.append(native)
        result = VLMProbeResult(
            analyzer=self.name, model=repr(model), cases=selected,
            findings={"method": self.method, "n_cases_analyzed": len(results), "per_case": summaries},
            artifacts={"probe_results": results, "evaluations": evaluations, "model_support": probe.describe()},
            metadata={
                "integration": "vlmprobe", "model_id": probe.adapter.model_id,
                "operation": self.operation, "method_config": json_safe(self.method_config or {}),
                "parameters": json_safe(self.get_params()),
                "calibration_dir": str(self.calibration_dir) if self.calibration_dir else None,
                "automatic_calibration": False,
            },
        )
        if self.output_dir is not None:
            result.save_probes(self.output_dir)
        return result

    @staticmethod
    def _pair(options, key):
        alternative = "donor" if key == "source" else "source"
        if key in options and alternative in options:
            raise ValueError("Provide only one reference input via source or donor, not both")
        if key not in options and alternative in options:
            options[key] = options.pop(alternative)
        if key not in options:
            raise ValueError(f"Paired probing requires {key}=reference inputs in run_config")
