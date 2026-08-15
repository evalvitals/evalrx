"""EvalVitals — failure case analysis for LLMs and VLMs.

A model is built once from a **spec** (identity, in :mod:`evalvitals.specs`) and a
**backend** (runtime: ``hf_local`` / ``api`` / ``vllm_offline``); the backend
determines the capability set.  Analyzers are sklearn-style estimators matched to
models by capability.

Equivalent ways to run an analysis:

0. Bring your own model — wrap an already-loaded HF model + tokenizer::

    import evalvitals
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from evalvitals.analyzers.lens.logit_lens import LogitLensAnalyzer

    m = AutoModelForCausalLM.from_pretrained("my-org/my-llama")
    tok = AutoTokenizer.from_pretrained("my-org/my-llama")
    model = evalvitals.wrap(m, tok)                         # captum-style on-ramp
    result = LogitLensAnalyzer().run(model, "The capital of France is")

1. Curated checkpoints — build a model from the registry by key::

    import evalvitals
    from evalvitals.analyzers.attention.summary import AttentionAnalyzer

    model = evalvitals.load("qwen2.5-7b-instruct")          # spec key
    result = AttentionAnalyzer(top_k=5).run(model, "The capital of France is")
    print(result.summary())

2. Config-driven — declare model + analysis in YAML::

    from evalvitals import load_config, run

    config = load_config("configs/qwen_attention.yaml")
    result = run(config, "The capital of France is")

3. Hybrid convenience shim (auto-derived from capabilities)::

    result = model.call_attention("The capital of France is")

4. Explicit engine — pick the backend yourself::

    from evalvitals.models import compose
    model = compose("qwen2.5-7b-instruct", "hf_local", want={evalvitals.Capability.ATTENTION})
"""

import logging as _logging

# Importing these populates the registry (models + analyzers self-register).
import evalvitals.analyzers as _analyzers  # noqa: E402,F401
from evalvitals.config import AnalysisConfig, load_config
from evalvitals.core import (
    Analyzer,
    Capability,
    CaseBatch,
    FailureCase,
    Model,
    Result,
    registry,
)
from evalvitals.core.tool import Tool, ToolCall
from evalvitals.logging_utils import disable_console_logging, enable_console_logging
from evalvitals.models import Agent, RuntimeConfig, compose, load, load_model, wrap
from evalvitals.specs import get_spec, list_specs

# Library hygiene: silent by default (no "No handlers could be found" noise,
# no raw unformatted last-resort stderr output for a stray .warning() call)
# until a caller opts in via evalvitals.enable_console_logging(). See
# logging_utils.py's module docstring for the full rationale.
_logging.getLogger("evalvitals").addHandler(_logging.NullHandler())

__version__ = "0.1.1"
__all__ = [
    "load",
    "load_config",
    "load_model",
    "wrap",
    "compose",
    "RuntimeConfig",
    "Agent",
    "Tool",
    "ToolCall",
    "get_spec",
    "list_specs",
    "run",
    "explore",
    "run_codebase",
    "AnalysisConfig",
    "Model",
    "Analyzer",
    "Result",
    "FailureCase",
    "CaseBatch",
    "Capability",
    "registry",
    "enable_console_logging",
    "disable_console_logging",
]


def __getattr__(name: str):
    # Lazy: evalvitals.analysis pulls in the CLI-agent runtime, which most
    # `import evalvitals` callers (running an analyzer directly) don't need.
    if name == "explore":
        from evalvitals.analysis import explore

        return explore
    if name == "run_codebase":
        from evalvitals.analysis import run_codebase

        return run_codebase
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def run(config: AnalysisConfig, data, **kwargs):
    """Run the analysis declared by *config* on *data*.

    Equivalent to::

        analyzer = registry.analyzers.get(config.analysis)(**config.analysis_kwargs)
        analyzer.run(load_model(config.model), data)

    Args:
        config: Loaded :class:`AnalysisConfig` (see :func:`load_config`).
        data:   ``str | FailureCase | CaseBatch | iterable`` to analyse.
        **kwargs: Override or extend ``config.analysis_kwargs`` at call time.

    Returns:
        A :class:`~evalvitals.core.result.Result` subclass.
    """
    # Back-compat: tolerate a leading "call_" in the config (old style).
    name = config.analysis
    if name.startswith("call_"):
        name = name[len("call_"):]

    analyzer_cls = registry.analyzers.get(name)
    analyzer = analyzer_cls(**{**config.analysis_kwargs, **kwargs})
    model = load_model(config.model)
    return analyzer.run(model, data)
