# EvalRX on a Colab TPU

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/evalvitals/evalrx/blob/ruinan/examples/colab/chartqa_repair.ipynb)

Open [chartqa_repair.ipynb](chartqa_repair.ipynb), select a TPU runtime, and run the cells in order.
The notebook uses `evalrx run` to evaluate and diagnose Gemma 4 E2B-it on ChartQA,
then search and confirm L1 prompt repairs. Add `GEMINI_API_KEY` to Colab Secrets
and enable notebook access. The notebook installs and launches Gemini CLI for
analysis and code generation; the target model runs on the TPU. No interactive
agent login is required.

Installation, commands, terminal output, result tables, figures and full stage
records are saved in notebook cell outputs when the run completes. No companion script or saved
prediction file is required. Runtime scratch files are created automatically.
