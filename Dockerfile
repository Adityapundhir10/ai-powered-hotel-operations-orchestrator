FROM python:3.11-slim

ARG INSTALL_FULL_AI=false
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl tesseract-ocr && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements-full.txt ./
RUN if [ "$INSTALL_FULL_AI" = "true" ]; then \
      pip install -r requirements-full.txt; \
    else \
      pip install -r requirements.txt; \
    fi

COPY . .
RUN mkdir -p models artifacts/benchmarks artifacts/evaluations

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
