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

RUN pip install --no-cache-dir \
    torch torchvision --index-url https://download.pytorch.org/whl/cu124

COPY . /evalrx
COPY --from=web /src/evalrx/reporting/web_dist /evalrx/evalrx/reporting/web_dist
RUN pip install --no-cache-dir "/evalrx[local,data]" PyYAML

WORKDIR /app/example
COPY examples/m1_m5/deco_hallu/ .

ENV PYTHONUNBUFFERED=1

CMD ["python", "run.py"]
