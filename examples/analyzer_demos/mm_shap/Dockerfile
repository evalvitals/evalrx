FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

COPY . /evalrx
RUN pip install --no-cache-dir "/evalrx[api]" Pillow

WORKDIR /app/example
COPY examples/analyzer_demos/mm_shap/ .

ENV PYTHONUNBUFFERED=1

CMD ["python", "run.py"]
