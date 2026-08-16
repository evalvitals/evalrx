FROM python:3.11-slim

WORKDIR /app

# CUDA torch first so the [local] extra doesn't pull a CPU-only wheel
RUN pip install --no-cache-dir \
    torch torchvision --index-url https://download.pytorch.org/whl/cu124

COPY . /evalvitals
RUN pip install --no-cache-dir "/evalvitals[local,data]" openai

WORKDIR /app/example
COPY examples/m4/vlm_paper_benchmark/ .

ENV PYTHONUNBUFFERED=1

# run_hf_autofix.py drives the white-box (HFLocalModel, no served endpoint
# needed) arc -- the one this directory's own experiments_s_tier_2026-08.md
# documents in depth. run_autofix.py (black-box, needs an external
# OpenAI-compatible endpoint via AUTOFIX_BASE_URL) and run_hf_vicrop.py
# (LLaVA-only) are the other two entry points; override `command:` in
# docker-compose.yml to use them instead.
CMD ["sh", "-c", "python download_benchmarks.py --paper pope_random --per-paper 20 && python run_hf_autofix.py pope_random --model qwen3-vl-2b-instruct --limit 20 --diagnosis-cases 4 --selection-cases 8"]
