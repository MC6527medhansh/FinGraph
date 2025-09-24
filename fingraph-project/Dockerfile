# Multi-stage build to reduce final image size
FROM python:3.11-slim as base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    PYTHONPATH=/app

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl && \
    rm -rf /var/lib/apt/lists/* && \
    apt-get clean

# Copy requirements
COPY dashboard-requirements.txt worker-requirements.txt ./

# Install Python dependencies and clean up in same layer to reduce size
RUN python -m pip install --upgrade pip && \
    python -m pip install -r dashboard-requirements.txt && \
    python -m pip install -r worker-requirements.txt && \
    pip cache purge && \
    find /usr/local/lib/python3.11 -name "*.pyc" -delete && \
    find /usr/local/lib/python3.11 -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

# Copy application code
COPY . /app

# Remove unnecessary files to reduce image size
RUN rm -rf /app/.git* /app/README.md /app/render.yaml && \
    find /app -name "*.pyc" -delete && \
    find /app -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

EXPOSE 8000
ENV PORT=8000

# Default command (dashboard) - worker will override via start command
CMD ["sh","-lc","python -m streamlit run src/dashboard/app.py --server.port ${PORT:-8080} --server.address 0.0.0.0"]