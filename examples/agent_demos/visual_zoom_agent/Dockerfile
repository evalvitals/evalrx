FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    torch torchvision --index-url https://download.pytorch.org/whl/cu124

COPY . /evalvitals
RUN pip install --no-cache-dir "/evalvitals[local]"

WORKDIR /app/example
COPY examples/agent_demos/visual_zoom_agent/ .

ENV PYTHONUNBUFFERED=1

CMD ["python", "run.py", "--device", "cuda"]
