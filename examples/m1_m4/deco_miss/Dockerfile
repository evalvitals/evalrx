FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    torch torchvision --index-url https://download.pytorch.org/whl/cu124

COPY . /evalvitals
RUN pip install --no-cache-dir "/evalvitals[local,data]" PyYAML

WORKDIR /app/example
COPY examples/m1_m4/deco_miss/ .

ENV PYTHONUNBUFFERED=1

CMD ["python", "run.py"]
