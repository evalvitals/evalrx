FROM python:3.11-slim

WORKDIR /app
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu124
COPY . /evalrx
# viz: matplotlib for M2 figures + explore chart rendering; pandas for the
# explorer's generated analysis.py (both were silently skipped without it).
RUN pip install --no-cache-dir "/evalrx[local,data,viz]" pandas

WORKDIR /app/example
COPY examples/m1_m4/vlm_benchmark_common.py /app/vlm_benchmark_common.py
COPY examples/m1_m4/chartqa_qwen2_5_vl/ .
ENV PYTHONUNBUFFERED=1 PYTHONPATH=/app
CMD ["python", "run.py"]

