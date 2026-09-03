"""EvalRX core — the sklearn-like substrate.

Public contracts every other module builds on:

  Capability   the vocabulary connecting models and analyzers
  Model        analyzable model:  generate() + forward(capture)->Trace
  Analyzer     sklearn-style estimator:  Analyzer(**params).run(model, data)->Result
  FailureCase  the central data unit;  CaseBatch is a collection of them
  Result       uniform, agent-readable output (findings + artifacts)
  registry     discovery: list/match models and analyzers by capability
  Pipeline     compose analyzers
  Experiment   declarative spec the agent layer targets
"""

from evalrx.core.analyzer import Analyzer
from evalrx.core.capability import Capability, CapabilityError
from evalrx.core.case import (
    CaseBatch,
    FailureCase,
    Inputs,
    Label,
    Provenance,
    Source,
    Step,
    StepRole,
    Trajectory,
    as_casebatch,
)
from evalrx.core.experiment import Experiment, ExperimentRunner
from evalrx.core.model import CaptureSpec, Model, TokenLogprob, Trace
from evalrx.core.pipeline import Pipeline
from evalrx.core.registry import (
    Registry,
    register_analyzer,
    register_model,
    registry,
)
from evalrx.core.result import Result
from evalrx.core.spec import (
    AttnSemantics,
    AudioSpec,
    ModelSpec,
    ModulePaths,
    VisionSpec,
)
from evalrx.core.tokentype import TokenTypeMap, build_token_type_map
from evalrx.core.tool import ChatTurn, Tool, ToolCall

__all__ = [
    "Capability",
    "CapabilityError",
    "Model",
    "Trace",
    "TokenLogprob",
    "CaptureSpec",
    "Analyzer",
    "FailureCase",
    "CaseBatch",
    "Inputs",
    "Label",
    "Provenance",
    "Source",
    "Step",
    "StepRole",
    "Trajectory",
    "as_casebatch",
    "ModelSpec",
    "VisionSpec",
    "AudioSpec",
    "ModulePaths",
    "AttnSemantics",
    "Tool",
    "ToolCall",
    "ChatTurn",
    "TokenTypeMap",
    "build_token_type_map",
    "Result",
    "registry",
    "Registry",
    "register_model",
    "register_analyzer",
    "Pipeline",
    "Experiment",
    "ExperimentRunner",
]
