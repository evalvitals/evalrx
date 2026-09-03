"""Natural-language scaffold for the EvalRX diagnosis pipeline.

Generates a self-contained Docker experiment from a plain-text description
of a model failure.  The resulting directory mirrors the ``examples/`` layout
and can be launched immediately::

    python -m evalrx.eval_agent.nl_runner \\
        --description "My VLM confuses left and right in spatial questions" \\
        --model qwen2.5-vl-7b-instruct \\
        --out ./my_experiment

    cd my_experiment && docker compose up

Two generation modes
--------------------
CLI agent  (``--provider claude_code / gemini_cli / …``)
    A coding agent writes a bespoke ``run.py`` tailored to the description.
    Requires the corresponding binary and API key.

Template   (default, no extra binary needed)
    A parametrised template is filled with the description and model key.
    The produced ``run.py`` is a complete, working script that users can
    customise further.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from evalrx.eval_agent.prompts.nl_runner import _AGENT_PROMPT_TEMPLATE

# ---------------------------------------------------------------------------
# File templates
# ---------------------------------------------------------------------------

_RUN_PY_TEMPLATE = '''\
"""EvalRX diagnosis experiment — generated from description.

Description
-----------
{description}

Model : {model_key}

Usage (via Docker — preferred)
    docker compose up

Usage (direct)
    python run.py --smoke-test      # quick wiring check, no GPU/API key
    python run.py                   # full run

Edit the CASES list below to add your own failure examples.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_OUTPUTS_DIR = Path(__file__).parent / "outputs"

# ---------------------------------------------------------------------------
# Failure cases — EDIT THESE to match your scenario
# ---------------------------------------------------------------------------
# Each FailureCase has:
#   id       : unique string key
#   inputs   : Inputs(prompt=..., image=...) — image is optional
#   expected : what the correct answer looks like (string or rubric dict)
# ---------------------------------------------------------------------------

def _build_cases():
    from evalrx.core.case import CaseBatch, FailureCase, Inputs

    return CaseBatch([
        # Replace these with real failure cases from your evaluation.
        FailureCase(
            id="example_0",
            inputs=Inputs(prompt="[TODO: replace with a prompt that exposes the failure]"),
            expected="[expected answer]",
        ),
        FailureCase(
            id="example_1",
            inputs=Inputs(prompt="[TODO: replace with a passing control prompt]"),
            expected="[expected answer]",
        ),
    ])


# ---------------------------------------------------------------------------
# Smoke-test stand-in (no model / GPU required)
# ---------------------------------------------------------------------------

class _SmokeModel:
    def __init__(self):
        from evalrx.core.capability import Capability
        self.capabilities = frozenset({{Capability.GENERATE}})
        self.modalities = frozenset({{"text"}})

    def generate(self, inputs, **kwargs) -> str:
        return "smoke-test answer"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="{model_key}")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--max-cycles", type=int, default=2)
    parser.add_argument("--max-analyzers", type=int, default=2)
    parser.add_argument("--smoke-test", action="store_true",
                        help="Run with a tiny stand-in model (no GPU/API key needed)")
    parser.add_argument("--run-dir", default=str(_OUTPUTS_DIR))
    args = parser.parse_args()

    import evalrx  # noqa: F401 — registers all analyzers

    from evalrx.eval_agent import AgyModel, RunContext, VLDiagnoseLoop
    from evalrx.eval_agent.stages.diagnosis import DiagnosisAgent
    from evalrx.eval_agent.stages.probe_agent import ProbeAgent
    from evalrx.eval_agent.stages.protocol import ExperimentProtocol
    from evalrx.analysis.stats_agent import StatsAnalysisAgent

    protocol = ExperimentProtocol(
        description=(
            "{description}"
        ),
        task_domain="{task_domain}",
        success_criteria="The model answer matches the expected output.",
    )

    if args.smoke_test:
        model = _SmokeModel()
        print("[smoke-test] using stand-in model — skipping real inference")
    else:
        from evalrx import compose
        from evalrx.core.capability import Capability
        model = compose(args.model, "hf_local",
                        want={{Capability.GENERATE, Capability.ATTENTION}},
                        device=args.device, dtype=args.dtype)

    cases = _build_cases()

    try:
        judge = AgyModel()
    except RuntimeError:
        judge = None
        print("[warn] agy binary not found — M3/M5 will be skipped")

    ctx = RunContext(args.run_dir, verbose=True)
    print(f"Logging to: {{ctx.root}}")

    loop = VLDiagnoseLoop(
        model=model,
        probe_agent=ProbeAgent(max_analyzers=args.max_analyzers),
        stats_agent=StatsAnalysisAgent(judge=judge, figure_dir=str(ctx.figures_dir)),
        diagnosis_agent=DiagnosisAgent(judge=judge),
        max_cycles=args.max_cycles,
        protocol=protocol,
        run_logger=ctx.logger,
    )
    report = loop.run(cases)
    ctx.write_diagnose_report(report, cases)
    ctx.finalize()

    print(f"\\n[VLDiagnoseLoop] cycles={{report.cycles}}  resolved={{report.resolved}}")
    for h in report.final_hypotheses:
        print(f"  hypothesis : {{h.statement}}")
        print(f"  status     : {{h.status}}")
    print(f"\\nFull guide -> {{ctx.root / 'README.txt'}}")


if __name__ == "__main__":
    main()
'''

_DOCKERFILE_TEMPLATE = """\
FROM python:3.11-slim

WORKDIR /app

# Install torch with CUDA 12.4 wheels first so [local] won't reinstall it
RUN pip install --no-cache-dir \\
    torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Install evalrx from local source
COPY . /evalrx
RUN pip install --no-cache-dir "/evalrx[local,gemini,data,viz]"

WORKDIR /app/example
COPY {rel_dir}/ .

ENV PYTHONUNBUFFERED=1

CMD ["python", "run.py"]
"""

_COMPOSE_TEMPLATE = """\
services:
  evalrx_experiment:
    build:
      context: {build_context}
      dockerfile: {rel_dir}/Dockerfile
    image: evalrx-{slug}
    environment:
      - CUDA_DEVICE_ORDER=PCI_BUS_ID
      - CUDA_VISIBLE_DEVICES=${{CUDA_VISIBLE_DEVICES:-0}}
      - PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
      - HF_HOME=/hf_cache
      - HOME=/home/user
    volumes:
      - ${{HF_HOME:-~/.cache/huggingface}}:/hf_cache
      - ./outputs:/app/example/outputs
      - ${{AGY_PATH:-~/.local/bin/agy}}:/usr/local/bin/agy:ro
      - ~/.gemini:/home/user/.gemini
      - ~/.cache/antigravity:/home/user/.cache/antigravity
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    command: >
      python run.py
        --model {model_key}
        --device cuda
        --dtype bfloat16
        --max-cycles 2
        --max-analyzers 2
"""

_GITIGNORE = "outputs/\n__pycache__/\n"

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def scaffold_from_description(
    description: str,
    model_key: str = "qwen2.5-vl-7b-instruct",
    output_dir: str | Path = "./my_evalrx_experiment",
    provider: str = "",
    cli_binary: str = "",
    cli_model: str = "",
    cli_timeout: int = 180,
    task_domain: str = "",
) -> Path:
    """Scaffold a Docker experiment from a natural-language failure description.

    Args:
        description:  Plain-text description of the failure to diagnose.
        model_key:    EvalRX model key (from ``evalrx.list_specs()``).
        output_dir:   Where to write the generated scaffold.
        provider:     CLI agent provider (``"claude_code"``, ``"gemini_cli"``,
                      ``"codex"``, …).  Leave empty to use the template path.
        cli_binary:   Explicit path to the CLI binary (auto-detected when empty).
        cli_model:    Model flag forwarded to the CLI binary (e.g. ``"sonnet"``).
        cli_timeout:  Seconds allowed for the CLI agent to write ``run.py``.
        task_domain:  Short label for the experiment (e.g. ``"spatial reasoning"``).

    Returns:
        ``Path`` to the generated scaffold directory.

    Raises:
        RuntimeError: When ``provider`` is set but the binary is not found.
    """
    out = Path(output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    run_py_path = out / "run.py"

    if provider:
        _write_run_py_via_agent(
            run_py_path,
            description=description,
            model_key=model_key,
            provider=provider,
            binary=cli_binary,
            model=cli_model,
            timeout=cli_timeout,
        )
    else:
        _write_run_py_template(
            run_py_path,
            description=description,
            model_key=model_key,
            task_domain=task_domain or _infer_domain(description),
        )

    _write_docker_files(out, model_key=model_key)
    return out


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _infer_domain(description: str) -> str:
    """Return a short domain label heuristically extracted from the description."""
    desc_lower = description.lower()
    for keyword in ("spatial", "temporal", "count", "ocr", "grounding",
                    "hallucination", "reasoning", "chart", "navigation"):
        if keyword in desc_lower:
            return keyword + " understanding"
    return "visual understanding"


def _write_run_py_template(
    path: Path,
    description: str,
    model_key: str,
    task_domain: str,
) -> None:
    escaped_desc = description.replace('"', '\\"')
    content = _RUN_PY_TEMPLATE.format(
        description=escaped_desc,
        model_key=model_key,
        task_domain=task_domain,
    )
    path.write_text(content, encoding="utf-8")
    print(f"[nl_runner] wrote {path}")
    print("[nl_runner] edit CASES in run.py to add your failure examples")


def _write_run_py_via_agent(
    path: Path,
    *,
    description: str,
    model_key: str,
    provider: str,
    binary: str,
    model: str,
    timeout: int,
) -> None:
    from evalrx.agent_runtime.cli_types import CliAgentConfig
    from evalrx.agent_runtime.codegen import CodegenRunner

    prompt = _AGENT_PROMPT_TEMPLATE.format(
        description=description,
        model_key=model_key,
    )
    cfg = CliAgentConfig(
        provider=provider,
        binary_path=binary,
        model=model,
        timeout_sec=timeout,
    )
    result = CodegenRunner(cfg).write_code(
        prompt,
        workdir=path.parent,
        preferred_filenames=("run.py",),
    )

    if not result.ok:
        raise RuntimeError(
            f"CLI agent ({provider}) failed to generate run.py: {result.error}"
        )

    generated = result.code
    if generated is None:
        raise RuntimeError(
            f"CLI agent ({provider}) produced no .py files in {path.parent}"
        )

    path.write_text(generated, encoding="utf-8")
    print(f"[nl_runner] {provider} wrote {path} ({len(result.files)} file(s))")


def _write_docker_files(out: Path, *, model_key: str) -> None:
    slug = model_key.replace(".", "-").replace("/", "-")

    # Detect whether out is inside the repo (for a relative build context)
    try:
        rel = out.relative_to(Path.cwd())
        rel_dir = str(rel).replace("\\", "/")
        build_context = str(Path.cwd()).replace("\\", "/")
    except ValueError:
        rel_dir = out.name
        build_context = str(out.parent).replace("\\", "/")

    dockerfile = _DOCKERFILE_TEMPLATE.format(rel_dir=rel_dir)
    compose = _COMPOSE_TEMPLATE.format(
        build_context=build_context,
        rel_dir=rel_dir,
        slug=slug,
        model_key=model_key,
    )

    (out / "Dockerfile").write_text(dockerfile, encoding="utf-8")
    (out / "docker-compose.yml").write_text(compose, encoding="utf-8")
    (out / ".gitignore").write_text(_GITIGNORE, encoding="utf-8")
    print(f"[nl_runner] scaffold ready in {out}")
    print(f"[nl_runner]   cd {out} && docker compose up")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Scaffold an EvalRX Docker experiment from a natural-language "
            "description of a model failure."
        )
    )
    p.add_argument(
        "--description", "-d", required=True,
        help="Plain-text description of the failure to diagnose.",
    )
    p.add_argument(
        "--model", "-m", default="qwen2.5-vl-7b-instruct",
        metavar="MODEL_KEY",
        help="EvalRX model key (see evalrx.list_specs()).",
    )
    p.add_argument(
        "--out", "-o", default="./my_evalrx_experiment",
        metavar="DIR",
        help="Output directory for the generated scaffold.",
    )
    p.add_argument(
        "--provider", default="",
        metavar="PROVIDER",
        help=(
            "CLI agent provider for run.py generation "
            "(claude_code / gemini_cli / codex / opencode / kimi_cli). "
            "Omit to use the built-in template."
        ),
    )
    p.add_argument("--cli-model", default="", help="Model flag forwarded to the CLI binary.")
    p.add_argument("--cli-binary", default="", help="Explicit path to the CLI binary.")
    p.add_argument("--cli-timeout", type=int, default=180, help="CLI agent timeout (seconds).")
    p.add_argument("--task-domain", default="", help='Short domain label, e.g. "spatial reasoning".')
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    out = scaffold_from_description(
        description=args.description,
        model_key=args.model,
        output_dir=args.out,
        provider=args.provider,
        cli_binary=args.cli_binary,
        cli_model=args.cli_model,
        cli_timeout=args.cli_timeout,
        task_domain=args.task_domain,
    )
    print(f"\nScaffold created at: {out}")
    print(f"  cd {out} && docker compose up")


if __name__ == "__main__":
    main()
