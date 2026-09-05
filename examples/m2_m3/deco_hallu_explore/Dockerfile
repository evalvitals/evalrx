FROM python:3.11-slim

WORKDIR /app

COPY . /evalrx
RUN pip install --no-cache-dir "/evalrx[dashboard,viz]"

WORKDIR /app/example
COPY examples/m2_m3/deco_hallu_explore/ .
COPY examples/m1_m5/deco_hallu/data/cases/ ./source_cases/

ENV PYTHONUNBUFFERED=1

CMD ["bash", "run.sh"]
