#!/usr/bin/env bash
set -euo pipefail

# Image selection (GHCR-friendly defaults)
REGISTRY="${REGISTRY:-ghcr.io}"
OWNER="${OWNER:-random-plm}"
IMAGE_NAME="${IMAGE_NAME:-foretrieval-server}"
IMAGE_TAG="${IMAGE_TAG:-v0.0.1}"
IMAGE="${IMAGE:-${REGISTRY}/${OWNER}/${IMAGE_NAME}:${IMAGE_TAG}}"
GITHUB_PAT="${GITHUB_PAT:-}"

# Container/runtime settings
CONTAINER_NAME="${CONTAINER_NAME:-foretrieval-server}"
HOST_PORT="${HOST_PORT:-8000}"
CONTAINER_PORT="${CONTAINER_PORT:-8000}"
HF_CACHE_DIR="${HF_CACHE_DIR:-/models/foretrieval}"

# GPU pinning
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# Server env vars
FOR_EMBED_MODEL="${FOR_EMBED_MODEL:-vidore/colqwen2-v1.0}"
FOR_EMBED_DEVICE="${FOR_EMBED_DEVICE:-cuda}"
FOR_EMBED_VERBOSE="${FOR_EMBED_VERBOSE:-1}"
FOR_SERVER_WORKERS="${FOR_SERVER_WORKERS:-1}"
FOR_SERVER_MAX_INFLIGHT="${FOR_SERVER_MAX_INFLIGHT:-1}"
FOR_SERVER_SINGLE_MODEL_CACHE="${FOR_SERVER_SINGLE_MODEL_CACHE:-true}"
FOR_EMBED_LOAD_IN_4BIT="${FOR_EMBED_LOAD_IN_4BIT:-false}"
FOR_EMBED_LOAD_IN_8BIT="${FOR_EMBED_LOAD_IN_8BIT:-false}"
FOR_EMBED_BNB_4BIT_QUANT_TYPE="${FOR_EMBED_BNB_4BIT_QUANT_TYPE:-nf4}"
FOR_EMBED_BNB_4BIT_COMPUTE_DTYPE="${FOR_EMBED_BNB_4BIT_COMPUTE_DTYPE:-float16}"
HF_TOKEN="${HF_TOKEN:-}"

echo "Starting container '${CONTAINER_NAME}' from image '${IMAGE}'..."
echo "GPU pinning: CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "HF cache mount: ${HF_CACHE_DIR} -> /root/.cache/huggingface"

if [[ -z "${GITHUB_PAT}" ]]; then
  echo "Error: GITHUB_PAT is not set."
  echo "Set it before running, e.g.:"
  echo "  export GITHUB_PAT=ghp_xxx"
  exit 1
fi

printf '%s' "${GITHUB_PAT}" | docker login ghcr.io -u "${OWNER}" --password-stdin >/dev/null
echo "Authenticated to ghcr.io as ${OWNER}."

mkdir -p "${HF_CACHE_DIR}"

env_args=(
  -e CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}"
  -e FOR_EMBED_MODEL="${FOR_EMBED_MODEL}"
  -e FOR_EMBED_DEVICE="${FOR_EMBED_DEVICE}"
  -e FOR_EMBED_VERBOSE="${FOR_EMBED_VERBOSE}"
  -e FOR_SERVER_WORKERS="${FOR_SERVER_WORKERS}"
  -e FOR_SERVER_MAX_INFLIGHT="${FOR_SERVER_MAX_INFLIGHT}"
  -e FOR_SERVER_SINGLE_MODEL_CACHE="${FOR_SERVER_SINGLE_MODEL_CACHE}"
  -e FOR_EMBED_LOAD_IN_4BIT="${FOR_EMBED_LOAD_IN_4BIT}"
  -e FOR_EMBED_LOAD_IN_8BIT="${FOR_EMBED_LOAD_IN_8BIT}"
  -e FOR_EMBED_BNB_4BIT_QUANT_TYPE="${FOR_EMBED_BNB_4BIT_QUANT_TYPE}"
  -e FOR_EMBED_BNB_4BIT_COMPUTE_DTYPE="${FOR_EMBED_BNB_4BIT_COMPUTE_DTYPE}"
)

if [[ -n "${HF_TOKEN}" ]]; then
  env_args+=(-e HF_TOKEN="${HF_TOKEN}")
else
  echo "HF_TOKEN is empty; not passing it to the container."
fi

docker run --rm -it \
  --name "${CONTAINER_NAME}" \
  --gpus "device=${CUDA_VISIBLE_DEVICES}" \
  -p "${HOST_PORT}:${CONTAINER_PORT}" \
  -v "${HF_CACHE_DIR}:/root/.cache/huggingface" \
  "${env_args[@]}" \
  "${IMAGE}"
