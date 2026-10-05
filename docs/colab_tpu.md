# Colab TPU notebook

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/evalvitals/evalrx/blob/ruinan/examples/colab/chartqa_repair.ipynb)

The [notebook](https://github.com/evalvitals/evalrx/blob/ruinan/examples/colab/chartqa_repair.ipynb)
contains the complete workflow: install EvalRX, check the TPU, run `evalrx run`,
and inspect the resulting diagnosis and prompt-repair evaluation.
Run all cells and save the notebook to retain console output, tables, figures,
agent-generated code and stage records.

Select a TPU runtime and add `GEMINI_API_KEY` to Colab Secrets. The notebook
installs Gemini CLI, verifies it can create and execute code, and runs
`evalrx run --judge-provider gemini --coder-provider gemini_cli`.
Gemma 4 E2B-it runs locally with `jax_local`; Gemini CLI performs analysis and
code generation inside the Colab runtime. The small ChartQA sample
is a workflow demonstration, not a full benchmark or a promised accuracy gain.

See [CLI reference](cli.md#evalrx-run) for flags and
[JAX backend design](design_jax_backend.md) for backend capabilities.
