"""Docker worker — runs inside the analyzer container.

Invoked as ``python -m evalrx.agent_runtime._docker_runner``.

Reads a JSON payload from stdin::

    {
        "analyzer": "self_consistency",
        "params":   {"n": 3},
        "cases":    [{"id": "...", "prompt": "...", "label": null, "metadata": {}}],
        "model_env": "GEMINI_API_KEY"
    }

Writes the analyzer findings as JSON to stdout::

    {"findings": {...}}

The containerised model is constructed from the ``model_env`` key: the value
names the environment variable that holds the API key.  Currently supports
Gemini (``GEMINI_API_KEY``) and OpenAI-compatible (``OPENAI_API_KEY``) keys.
An unrecognised or missing key produces a ``{"error": "..."}`` response.
"""

from __future__ import annotations

import json
import os
import sys


def _build_model(model_env: str):
    key = os.getenv(model_env, "")
    if not key:
        raise ValueError(
            f"Environment variable '{model_env}' is not set inside the container."
        )
    if "GEMINI" in model_env.upper():
        from evalrx.models.blackbox.gemini import GeminiModel

        return GeminiModel(api_key=key)
    raise ValueError(
        f"No model factory for env var '{model_env}'. "
        "Add a case to evalrx/agent_runtime/_docker_runner.py."
    )


def _build_cases(raw_cases: list[dict]):
    from evalrx.core.case import CaseBatch, FailureCase, Inputs, Label

    cases = []
    for rec in raw_cases:
        label = None
        raw_label = rec.get("label")
        if raw_label is not None:
            try:
                label = Label(raw_label)
            except ValueError:
                pass
        cases.append(
            FailureCase(
                inputs=Inputs(prompt=rec.get("prompt", "")),
                label=label,
                metadata=rec.get("metadata", {}),
            )
        )
    return CaseBatch(cases)


def main() -> None:
    payload = json.loads(sys.stdin.read())
    try:
        model = _build_model(payload.get("model_env", "GEMINI_API_KEY"))
        cases = _build_cases(payload.get("cases", []))
        analyzer_name = payload["analyzer"]
        params = payload.get("params", {})

        from evalrx.core.registry import registry

        cls = registry.analyzers.get(analyzer_name)
        analyzer = cls(**params)
        result = analyzer.run(model, cases)
        print(json.dumps({"findings": result.findings}, default=str))
    except Exception as exc:
        print(json.dumps({"error": str(exc), "findings": {}}))
        sys.exit(1)


if __name__ == "__main__":
    main()
