# syntax=docker/dockerfile:1.7
FROM python:3.12-slim AS runtime-base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Copy only dependency metadata first for better layer caching
COPY pyproject.toml ./

# Install third-party runtime dependencies from pyproject extras.
# This layer is invalidated only when dependency metadata changes.
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --upgrade pip setuptools wheel && \
    python - <<'PY' > /tmp/requirements-runtime.txt
import tomllib
from pathlib import Path

cfg = tomllib.loads(Path("pyproject.toml").read_text())
project = cfg["project"]
deps = list(project.get("dependencies", []))
optional = project.get("optional-dependencies", {})
deps.extend(optional.get("server", []))

# Stable order and dedupe for deterministic cache behavior.
seen = set()
ordered = []
for dep in deps:
    if dep not in seen:
        seen.add(dep)
        ordered.append(dep)
print("\n".join(ordered))
PY
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install -r /tmp/requirements-runtime.txt

# Copy package sources late so app code changes do not bust dependency layers
COPY README.md ./
COPY foretrieval ./foretrieval

# Install local package without re-resolving dependencies
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --no-deps .







# Shared CUDA Python runtime base for GPU stages.
# Keep this stage free of app source copies so app-only changes do not bust
# heavyweight builder caches (e.g. flash-attn wheel stage).
FROM nvidia/cuda:12.2.2-cudnn8-devel-ubuntu22.04 AS cuda-python-runtime-base

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    CUDA_HOME=/usr/local/cuda

RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    ca-certificates \
    gnupg \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-venv \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.12 /usr/local/bin/python
RUN python -m venv /opt/venv
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

WORKDIR /app
COPY pyproject.toml ./

# Single dependency/app installation point for GPU pipeline.
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --upgrade pip setuptools wheel && \
    python - <<'PY' > /tmp/requirements-runtime.txt
import tomllib
from pathlib import Path

cfg = tomllib.loads(Path("pyproject.toml").read_text())
project = cfg["project"]
deps = list(project.get("dependencies", []))
optional = project.get("optional-dependencies", {})
deps.extend(optional.get("server", []))

seen = set()
ordered = []
for dep in deps:
    if dep not in seen:
        seen.add(dep)
        ordered.append(dep)
print("\n".join(ordered))
PY

RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install -r /tmp/requirements-runtime.txt && \
    python -m pip install "bitsandbytes>=0.43"

# App installation layer for CUDA runtime images.
FROM cuda-python-runtime-base AS cuda-app-base

COPY README.md ./
COPY foretrieval ./foretrieval

RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --no-deps .

# Dedicated builder for flash-attn wheel artifacts.
FROM cuda-python-runtime-base AS flashattn-builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
RUN mkdir -p /wheels && \
    python -m pip wheel --no-build-isolation flash-attn -w /wheels

# GPU target: installs flash-attn wheel built in dedicated builder.
FROM cuda-app-base AS gpu

COPY --from=flashattn-builder /wheels /tmp/wheels

RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install /tmp/wheels/flash_attn-*.whl

# Server runtime defaults (override at `docker run -e ...`)
ENV FOR_EMBED_MODEL=vidore/colqwen2-v1.0 \
    FOR_EMBED_DEVICE=cuda \
    FOR_EMBED_VERBOSE=1 \
    FOR_EMBED_FLASH_ATTENTION=auto \
    FOR_SERVER_WORKERS=1 \
    FOR_SERVER_MAX_INFLIGHT=1 \
    FOR_SERVER_SINGLE_MODEL_CACHE=true \
    FOR_EMBED_LOAD_IN_4BIT=false \
    FOR_EMBED_LOAD_IN_8BIT=false \
    FOR_EMBED_BNB_4BIT_QUANT_TYPE=nf4 \
    FOR_EMBED_BNB_4BIT_COMPUTE_DTYPE=float16

EXPOSE 8000

CMD ["sh", "-c", "python -m uvicorn foretrieval.server.server_main:app --host 0.0.0.0 --port 8000 --workers ${FOR_SERVER_WORKERS:-1}"]






# Default target: CPU-friendly runtime without GPU-specific optional deps.
FROM runtime-base AS cpu

# Server runtime defaults (override at `docker run -e ...`)
ENV FOR_EMBED_MODEL=vidore/colqwen2-v1.0 \
    FOR_EMBED_DEVICE=cpu \
    FOR_EMBED_VERBOSE=1 \
    FOR_EMBED_FLASH_ATTENTION=auto \
    FOR_SERVER_WORKERS=1 \
    FOR_SERVER_MAX_INFLIGHT=1 \
    FOR_SERVER_SINGLE_MODEL_CACHE=true \
    FOR_EMBED_LOAD_IN_4BIT=false \
    FOR_EMBED_LOAD_IN_8BIT=false \
    FOR_EMBED_BNB_4BIT_QUANT_TYPE=nf4 \
    FOR_EMBED_BNB_4BIT_COMPUTE_DTYPE=float16

EXPOSE 8000

CMD ["sh", "-c", "python -m uvicorn foretrieval.server.server_main:app --host 0.0.0.0 --port 8000 --workers ${FOR_SERVER_WORKERS:-1}"]
