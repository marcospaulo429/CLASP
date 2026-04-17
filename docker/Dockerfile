# syntax=docker/dockerfile:1.4
# Single Dockerfile for all platforms; pick arch via docker compose services (linux/arm64 vs linux/amd64).
ARG BASE_IMAGE=python:3.10-bookworm

FROM ${BASE_IMAGE} AS base

ENV PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        ffmpeg \
        libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

FROM base AS runtime

COPY pyproject.toml uv.lock README.md ./
COPY src ./src
COPY scripts ./scripts

ARG UV_EXTRA_GROUPS=""
RUN --mount=type=cache,target=/root/.cache/uv \
    if [ -n "$UV_EXTRA_GROUPS" ]; then \
      uv sync --frozen --extra "${UV_EXTRA_GROUPS}"; \
    else \
      uv sync --frozen; \
    fi

ENV VIRTUAL_ENV=/app/.venv
ENV PATH="/app/.venv/bin:${PATH}"

CMD ["bash"]
