FROM python:3.11-slim

WORKDIR /app

COPY . /evalrx
RUN pip install --no-cache-dir "/evalrx" requests

WORKDIR /app/example
COPY examples/dataset_selection/llm_band_probe/ .

ENV PYTHONUNBUFFERED=1

CMD ["python", "band_locate.py", "--n", "60", "--concurrency", "16", "--out", "outputs/band_results.json"]
