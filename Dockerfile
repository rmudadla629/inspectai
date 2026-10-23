FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TORCH_HOME=/app/.cache/torch
WORKDIR /app
COPY requirements.txt ./
# The x86 CPU index avoids pulling CUDA libraries. Linux ARM wheels are CPU-only.
RUN if [ "$(uname -m)" = "x86_64" ]; then \
      pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cpu; \
    else pip install torch==2.9.1 torchvision==0.24.1; fi \
    && pip install -r requirements.txt && pip check
RUN groupadd --gid 10001 app && useradd --uid 10001 --gid app --no-create-home app \
    && mkdir -p artifacts .cache/torch && chown -R app:app /app
COPY inspectai/ ./inspectai/
USER app
EXPOSE 8100
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8100/health', timeout=4)"
CMD ["waitress-serve", "--host=0.0.0.0", "--port=8100", "--max-request-body-size=8388608", "inspectai.web:app"]
