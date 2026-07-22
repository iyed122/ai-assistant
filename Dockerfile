FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Step 1: Install CUDA PyTorch BEFORE requirements ───────────────────────────
# Pinned to 2.5.1+cu121 to match the working local venv. sentence-transformers
# 5.2.3 was built against torch 2.5.x and uses APIs absent from older releases.
# Installed BEFORE requirements.docker.txt so pip sees torch as already satisfied
# and never re-resolves it as a transitive dependency.
# No NVIDIA GPU? This wheel still runs (falls back to CPU); start the stack with
# docker-compose.cpu.yml so compose doesn't require a GPU device.
RUN pip install --no-cache-dir \
    torch==2.5.1+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

# ── Step 2: Install project dependencies ───────────────────────────────────────
COPY requirements.docker.txt .
RUN pip install --no-cache-dir -r requirements.docker.txt

# ── Step 3: Copy source code ───────────────────────────────────────────────────
# Explicit directory copies (not COPY . .) so build context stays clean
# and large files excluded by .dockerignore are never even considered.
COPY agent/    agent/
COPY api/      api/
COPY hammer/   hammer/
COPY pipeline/ pipeline/
COPY rag/      rag/
COPY sentries/ sentries/
COPY training/ training/

# ── Runtime config ─────────────────────────────────────────────────────────────
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
