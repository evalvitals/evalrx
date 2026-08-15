FROM python:3.11-slim

WORKDIR /app

COPY . /evalvitals
RUN pip install --no-cache-dir "/evalvitals[api,gemini]"

WORKDIR /app/example
COPY examples/preregistered_ab_demo/eval_agent/ .

ENV PYTHONUNBUFFERED=1

CMD ["python", "run.py"]
