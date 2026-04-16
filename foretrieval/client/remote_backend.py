from __future__ import annotations

import io
from typing import List, Optional

import torch
from PIL import Image

from .transport import dumps_payload, loads_payload


class RemoteEmbeddingClient:
    def __init__(
        self,
        server_url: str,
        model_name: str,
        token: Optional[str] = None,
        timeout: float = 30.0,
    ):
        try:
            import httpx
        except ImportError as exc:
            raise ImportError(
                "Remote embedding mode requires `httpx`. Install with `pip install httpx`."
            ) from exc

        self._httpx = httpx
        self.server_url = server_url.rstrip("/")
        self.model_name = model_name
        headers = {}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        self.client = httpx.Client(timeout=timeout, headers=headers)

    @staticmethod
    def _image_to_bytes(image: Image.Image) -> bytes:
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        return buf.getvalue()

    def encode_images(self, images: List[Image.Image]) -> torch.Tensor:
        payload = {
            "model": self.model_name,
            "images": [self._image_to_bytes(image.convert("RGB")) for image in images],
        }
        resp = self.client.post(
            f"{self.server_url}/v1/embed/images",
            content=dumps_payload(payload),
            headers={"Content-Type": "application/octet-stream"},
        )
        resp.raise_for_status()
        out = loads_payload(resp.content)
        return out["embeddings"].cpu()

    def encode_queries(self, queries: List[str]) -> torch.Tensor:
        payload = {"model": self.model_name, "queries": queries}
        resp = self.client.post(
            f"{self.server_url}/v1/embed/queries",
            content=dumps_payload(payload),
            headers={"Content-Type": "application/octet-stream"},
        )
        resp.raise_for_status()
        out = loads_payload(resp.content)
        return out["embeddings"].cpu()
