# The report renderer is built from source: web_dist/ is gitignored, so a
# fresh clone has none and the explore step's report.html needs it.
FROM node:22-slim AS web
WORKDIR /src/evalrx/reporting/web
COPY evalrx/reporting/web/package.json evalrx/reporting/web/package-lock.json ./
RUN npm ci
COPY evalrx/reporting/web/ ./
RUN npm run build

FROM python:3.11-slim

WORKDIR /app

# CUDA torch first so the [local] extra doesn't pull a CPU-only wheel
RUN pip install --no-cache-dir \
    torch torchvision --index-url https://download.pytorch.org/whl/cu124

COPY . /evalrx
COPY --from=web /src/evalrx/reporting/web_dist /evalrx/evalrx/reporting/web_dist
RUN pip install --no-cache-dir "/evalrx[local]"

WORKDIR /app/example
COPY examples/m2_m3/vtcbench_diagnosis/ .

ENV PYTHONUNBUFFERED=1

CMD ["sh", "-c", "python run_m1.py --task counting --detect-device cuda:0 --base-url ${VLLM_BASE_URL:-http://host.docker.internal:8901/v1} && bash run_explore.sh"]
