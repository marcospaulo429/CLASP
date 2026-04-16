# CLASP — imagem com CUDA (NVIDIA), Python 3.10, extras VoxPopuli + torchcodec.
# Alvo: Linux x86_64 + GPU NVIDIA (não Apple Silicon).
#
# Build:
#   docker build -t clasp:gpu .
#
# Shell interativo com GPU e volumes para dados/checkpoints no host:
#   docker run --rm -it --gpus all \
#     -v "$(pwd)/data:/app/data" -v "$(pwd)/models:/app/models" \
#     clasp:gpu bash
#
# Verificar CUDA dentro do container:
#   docker run --rm --gpus all clasp:gpu python -c "import torch; print('cuda:', torch.cuda.is_available())"
#
# Requer no host: driver NVIDIA + NVIDIA Container Toolkit (`docker run --gpus`).

# Tag oficial NVIDIA (cudnn incluído; ver outras em hub.docker.com/r/nvidia/cuda/tags)
ARG CUDA_IMAGE_TAG=12.4.1-cudnn-runtime-ubuntu22.04
FROM nvidia/cuda:${CUDA_IMAGE_TAG}

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    UV_LINK_MODE=copy

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    ffmpeg \
    git \
    libgomp1 \
    libsndfile1 \
    python3.10 \
    python3.10-venv \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Camada de dependências: só manifestos primeiro (cache de build)
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
COPY scripts ./scripts

RUN uv venv --python 3.10 /app/.venv \
    && uv sync --frozen --extra voxpopuli \
    && uv pip install torchcodec

ENV PATH="/app/.venv/bin:${PATH}" \
    PYTHONPATH="/app/src"

WORKDIR /app
CMD ["bash"]
