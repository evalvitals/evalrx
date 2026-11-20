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

COPY . /evalrx
COPY --from=web /src/evalrx/reporting/web_dist /evalrx/evalrx/reporting/web_dist
RUN pip install --no-cache-dir "/evalrx[viz]"

WORKDIR /app/example
COPY examples/m2_m3/synthetic_yield_explore/ .

ENV PYTHONUNBUFFERED=1

CMD ["bash", "run.sh"]
