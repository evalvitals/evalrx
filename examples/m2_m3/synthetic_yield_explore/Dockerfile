FROM python:3.11-slim

WORKDIR /app

COPY . /evalrx
RUN pip install --no-cache-dir "/evalrx[dashboard,viz]"

WORKDIR /app/example
COPY examples/m2_m3/synthetic_yield_explore/ .

ENV PYTHONUNBUFFERED=1

CMD ["bash", "run.sh"]
