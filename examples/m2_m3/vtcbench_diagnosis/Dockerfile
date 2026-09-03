FROM python:3.11-slim

WORKDIR /app

# CUDA torch first so the [local] extra doesn't pull a CPU-only wheel
RUN pip install --no-cache-dir \
    torch torchvision --index-url https://download.pytorch.org/whl/cu124

COPY . /evalrx
RUN pip install --no-cache-dir "/evalrx[local]"

WORKDIR /app/example
COPY examples/m2_m3/vtcbench_diagnosis/ .

ENV PYTHONUNBUFFERED=1

CMD ["sh", "-c", "python run_m1.py --task counting --detect-device cuda:0 --base-url ${VLLM_BASE_URL:-http://host.docker.internal:8901/v1} && bash run_explore.sh"]
