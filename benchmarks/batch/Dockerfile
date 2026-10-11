FROM python:3.12-slim
# rasterio wheels link against libexpat, which the slim image lacks.
RUN apt-get update && apt-get install -y --no-install-recommends libexpat1 \
 && rm -rf /var/lib/apt/lists/*
COPY constraints.txt disp_xr-*.whl /tmp/
RUN pip install --no-cache-dir -c /tmp/constraints.txt /tmp/disp_xr-*.whl \
 && rm /tmp/disp_xr-*.whl
COPY bench_downsample.py run.py /app/
ENTRYPOINT ["python", "-u", "/app/run.py"]
