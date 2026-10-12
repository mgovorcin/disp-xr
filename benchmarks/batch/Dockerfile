FROM python:3.12-slim
# rasterio wheels link against libexpat, which the slim image lacks.
RUN apt-get update && apt-get install -y --no-install-recommends libexpat1 \
 && rm -rf /var/lib/apt/lists/*
COPY constraints.txt disp_xr-*.whl /tmp/
# aws and cloud extras: S3 access, Icechunk, obstore (storage_s3.py).
RUN for w in /tmp/disp_xr-*.whl; do \
      pip install --no-cache-dir -c /tmp/constraints.txt "$w[aws,cloud]"; done \
 && rm /tmp/disp_xr-*.whl
COPY *.py /app/
ENTRYPOINT ["python", "-u", "/app/run.py"]
