FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    torch torchvision --index-url https://download.pytorch.org/whl/cu124

COPY . /evalrx
RUN pip install --no-cache-dir "/evalrx[local,data]" PyYAML

WORKDIR /app/example
COPY examples/m1_m5/deco_chair/ .

ENV PYTHONUNBUFFERED=1

CMD ["python", "run.py"]
