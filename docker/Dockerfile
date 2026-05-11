FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    tzdata \
    git \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY requirements_full.txt .
RUN pip install --no-cache-dir -r requirements_full.txt

COPY . .

RUN mkdir -p /app/artifacts/{reports,runs,logs,chromadb,tool_logs,memory}

EXPOSE 8000
EXPOSE 9090

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default: run the API server
CMD ["python", "-m", "src.api.server"]