# syntax=docker/dockerfile:1.7
FROM python:3.12-slim

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
deps.extend(optional.get("quant", []))

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

# Server runtime defaults (override at `docker run -e ...`)
ENV FOR_EMBED_MODEL=vidore/colqwen2-v1.0 \
    FOR_EMBED_DEVICE=cpu \
    FOR_EMBED_VERBOSE=1 \
    FOR_SERVER_WORKERS=1 \
    FOR_SERVER_MAX_INFLIGHT=1 \
    FOR_EMBED_LOAD_IN_4BIT=false \
    FOR_EMBED_LOAD_IN_8BIT=false \
    FOR_EMBED_BNB_4BIT_QUANT_TYPE=nf4 \
    FOR_EMBED_BNB_4BIT_COMPUTE_DTYPE=float16

EXPOSE 8000

CMD ["sh", "-c", "python -m uvicorn foretrieval.server.server_main:app --host 0.0.0.0 --port 8000 --workers ${FOR_SERVER_WORKERS:-1}"]
