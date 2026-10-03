# syntax=docker/dockerfile:1
FROM python:3.11-slim-bookworm

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_HTTP_TIMEOUT=600 \
    HF_HOME=/app/.cache/huggingface \
    TORCH_HOME=/app/.cache/torch \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ffmpeg: audio + frames from videos. git: some Python packages install from GitHub.
# The lib* packages are needed by OpenCV, Open3D and moderngl.
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    libsm6 \
    libx11-6 \
    libxext6 \
    libxrender1 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN uv venv /app/.venv --python 3.11
ENV VIRTUAL_ENV=/app/.venv \
    PATH="/app/.venv/bin:$PATH"

# Install dependencies first (MoGe is installed from src/MoGe), so code changes do not reinstall everything.
COPY requirements.txt .
COPY src/MoGe ./src/MoGe
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --index-strategy unsafe-best-match -r requirements.txt

COPY . .

# Model downloads (~7 GB) and results live in volumes so they survive container restarts.
VOLUME ["/app/.cache", "/app/output"]

ENTRYPOINT ["python", "src/app.py"]
CMD ["--help"]
