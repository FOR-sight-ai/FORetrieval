import io
from typing import Any

import httpx
from PIL import Image

from app.config import get_settings


async def call_embedding_api_for_query(query: str) -> dict[str, Any]:
    # Request query embeddings from the external embedding service.
    timeout = httpx.Timeout(get_settings().http_timeout_seconds)
    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(
            f"{get_settings().embedding_service_url}/v1/embed/query",
            json={"query": query},
        )
        response.raise_for_status()
        return response.json()


async def call_embedding_api_for_page(image: Image.Image, filename: str) -> dict[str, Any]:
    # Request page embeddings from the external embedding service using a PNG upload.
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    files = {"file": (filename, buffer.getvalue(), "image/png")}

    timeout = httpx.Timeout(get_settings().http_timeout_seconds)
    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(
            f"{get_settings().embedding_service_url}/v1/embed/page",
            files=files,
        )
        response.raise_for_status()
        return response.json()


async def check_embedding_api_health() -> str:
    # Probe the embedding service health endpoint and return a compact status string.
    try:
        timeout = httpx.Timeout(5.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(f"{get_settings().embedding_service_url}/health")
            response.raise_for_status()
    except Exception as exc:
        return f"error: {exc.__class__.__name__}"
    return "ok"
