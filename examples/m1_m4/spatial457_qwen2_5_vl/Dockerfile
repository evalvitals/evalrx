FROM python:3.11-slim

WORKDIR /app
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu124
COPY . /evalvitals
RUN pip install --no-cache-dir "/evalvitals[local,data]"

WORKDIR /app/example
COPY examples/m1_m4/vlm_benchmark_common.py /app/vlm_benchmark_common.py
COPY examples/m1_m4/spatial457_qwen2_5_vl/ .
ENV PYTHONUNBUFFERED=1 PYTHONPATH=/app
CMD ["python", "run.py"]

