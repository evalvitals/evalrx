FROM python:3.11-slim

WORKDIR /app

COPY . /evalrx
RUN pip install --no-cache-dir "/evalrx[api]"

WORKDIR /app/example
COPY examples/analyzer_demos/logprob_entropy/ .

ENV PYTHONUNBUFFERED=1

CMD ["python", "run.py"]
