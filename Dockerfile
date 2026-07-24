# KnowledgeForge application image.
#
# NOTE: This image contains ONLY the Streamlit app. The LLM/embedding models run
# in a SEPARATE Ollama server (see docker-compose.yml). The app reaches Ollama
# over the network via the OLLAMA host (default http://localhost:11434, or
# http://ollama:11434 when running under docker compose). Ollama is intentionally
# NOT bundled into this image.

FROM python:3.13-slim

WORKDIR /app

# Build essentials are only needed if a dependency has to compile from source.
# Most wheels are prebuilt for slim, so keep this minimal and clean up after.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first so this layer is cached across code changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application source.
COPY . .

# Streamlit default port.
EXPOSE 8501

# Container health is reported by Streamlit's built-in health endpoint.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8501/_stcore/health').status == 200 else 1)"

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
