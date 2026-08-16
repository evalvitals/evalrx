FROM python:3.11-slim

WORKDIR /app

COPY . /evalvitals
RUN pip install --no-cache-dir "/evalvitals[dashboard,data]"

WORKDIR /app/example
COPY examples/paper_diagnosis_benchmark/ .

ENV PYTHONUNBUFFERED=1

CMD ["sh", "-c", "python download_papers.py && python build_records.py && python run_benchmark.py --backend claude_code"]
