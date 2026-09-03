"""Config loading and dataclasses for evalrx."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ModelConfig:
    """Declarative model selection — resolved to a spec + backend at load time.

    ``name`` is a :mod:`evalrx.specs` key (e.g. ``"qwen2.5-7b-instruct"``) or a
    legacy alias (e.g. ``"qwen"``); resolution happens in
    :func:`evalrx.models.load_model` via ``compose(spec, backend)``.  No
    checkpoint is baked here — identity lives in the spec.

    Attributes:
        name:       Spec key or legacy alias.
        checkpoint: Optional override of the spec's ``hf_repo``.
        device:     Runtime device (``"auto"``, ``"cuda"``, ``"cpu"``, ``"cuda:N"``).
        dtype:      Runtime dtype (``"bfloat16"``, ``"float16"``, ``"float32"``).
        backend:    ``"hf_local"`` (internals) / ``"api"`` / ``"vllm_offline"``.
        want:       Capability names the backend must provide (e.g. ``["attention"]``).
    """

    name: str
    checkpoint: str | None = None
    device: str = "auto"
    dtype: str = "bfloat16"
    backend: str = "hf_local"
    want: list[str] = field(default_factory=list)


@dataclass
class AnalysisConfig:
    model: ModelConfig
    analysis: str
    analysis_kwargs: dict[str, Any] = field(default_factory=dict)


def load_config(path: str | Path) -> AnalysisConfig:
    """Load an AnalysisConfig from a YAML file.

    Supports both short form::

        model: qwen2.5-7b-instruct      # spec key (or a legacy alias like "qwen")
        analysis: attention

    and full form::

        model:
          name: qwen2.5-7b-instruct
          backend: hf_local             # hf_local | api | vllm_offline
          device: auto
          dtype: bfloat16
          want: [attention]             # capabilities the backend must provide
        analysis: attention             # registered analyzer name
        analysis_kwargs:
          layer: -1
          head: mean
          top_k: 10
    """
    with open(path) as f:
        raw = yaml.safe_load(f)

    model_raw = raw["model"]
    model_cfg = ModelConfig(name=model_raw) if isinstance(model_raw, str) else ModelConfig(**model_raw)

    return AnalysisConfig(
        model=model_cfg,
        analysis=raw["analysis"],
        analysis_kwargs=raw.get("analysis_kwargs", {}),
    )
