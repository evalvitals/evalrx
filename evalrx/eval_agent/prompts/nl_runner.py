"""Prompt templates for natural-language diagnosis scaffolding."""

_AGENT_PROMPT_TEMPLATE = """\
Generate a complete `run.py` for an EvalRX diagnosis experiment.

DESCRIPTION OF FAILURE TO DIAGNOSE
===================================
{description}

TARGET MODEL
============
{model_key}

REQUIREMENTS
============
1. Import from `evalrx` and `evalrx.eval_agent`.
2. Create an `ExperimentProtocol` whose `description` field captures the
   failure described above (verbatim or paraphrased).
3. Build a `CaseBatch` with 6-10 `FailureCase` objects that probe the
   described failure.  Include at least two "easy" control cases that the
   model should PASS and several "hard" cases that expose the failure.
   Use `Inputs(prompt=..., image=...)` — image is optional for text-only failures.
4. Load the model with `compose("{model_key}", "hf_local", ...)`.
5. Run `VLDiagnoseLoop` with `ProbeAgent`, `StatsAnalysisAgent`,
   `DiagnosisAgent`, and `AgyModel` as judge (with a try/except fallback
   when the agy binary is absent).
6. Accept CLI flags: --model, --device, --dtype, --max-cycles,
   --max-analyzers, --smoke-test, --run-dir.
7. Write outputs via `RunContext(args.run_dir)` (from `evalrx.eval_agent`),
   defaulting `--run-dir` to `Path(__file__).parent / "outputs"`. This
   defaults to the V2 logger (`run.json` + `M1..M5/log.json`, no flat
   `run_log.jsonl`, no `README.txt`/`manifest.json`) — don't assume those
   V1-only files exist when printing a completion summary.
8. Include a `--smoke-test` path with a `_SmokeModel` stand-in so the
   script can be verified without a GPU.

Write ONLY `run.py` — nothing else.  No markdown, no explanation.
"""
