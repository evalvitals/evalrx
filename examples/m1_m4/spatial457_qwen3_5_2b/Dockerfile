FROM python:3.11-slim

WORKDIR /app
# Qwen3.5 (architecture `qwen3_5`) needs transformers >= 5.15, which is not what
# the package's [local] extra pins (< 5). The cu129 torch wheel is the one the
# cluster's vLLM venv pairs with transformers 5.15 and it runs on a 12.4 driver.
RUN pip install --no-cache-dir "torch==2.13.0" --index-url https://download.pytorch.org/whl/cu129
COPY . /evalvitals
# viz: matplotlib for M2 figures + explore chart rendering; pandas for the
# explorer's generated analysis.py. transformers/accelerate installed explicitly
# (see above) instead of via [local].
RUN pip install --no-cache-dir "/evalvitals[data,viz]" pandas "transformers==5.15.0" "accelerate>=0.26.0"

WORKDIR /app/example
COPY examples/m1_m4/vlm_benchmark_common.py /app/vlm_benchmark_common.py
COPY examples/m1_m4/spatial457_qwen3_5_2b/ .
ENV PYTHONUNBUFFERED=1 PYTHONPATH=/app
CMD ["python", "run.py"]
