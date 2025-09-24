# Slim, reliable, and includes pip
FROM python:3.11-slim

# Avoid interactive prompts, speed up pip, and quiet Streamlit telemetry
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    PYTHONPATH=/app

WORKDIR /app

# System deps needed by pandas/pyarrow on slim
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements files
COPY requirements.txt /app/requirements.txt
COPY dashboard-requirements.txt /app/dashboard-requirements.txt  
COPY worker-requirements.txt /app/worker-requirements.txt

# Install all dependencies
RUN python -m pip install --upgrade pip && \
    python -m pip install -r /app/requirements.txt && \
    python -m pip install -r /app/worker-requirements.txt && \
    python -m pip install -r /app/dashboard-requirements.txt

# Copy app code
COPY . /app

# Default port for dashboard
EXPOSE 8000
ENV PORT=8000

# Default CMD (dashboard services can override, worker uses start command)
CMD ["sh","-lc","python -m streamlit run src/dashboard/app.py --server.port ${PORT:-8080} --server.address 0.0.0.0"]