from __future__ import annotations

import io
from concurrent.futures import ThreadPoolExecutor
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
        verify_ssl: bool = True,
        concurrency: int = 1,
        request_batch_size: Optional[int] = None,
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
        self.concurrency = max(1, int(concurrency))
        self.request_batch_size = int(request_batch_size) if request_batch_size else None
        if self.request_batch_size is not None and self.request_batch_size < 1:
            raise ValueError("request_batch_size must be >= 1 when provided.")
        headers = {}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        self.client = httpx.Client(timeout=timeout, headers=headers, verify=verify_ssl)

    @staticmethod
    def _image_to_bytes(image: Image.Image) -> bytes:
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        return buf.getvalue()

    @staticmethod
    def _chunked(items: list, chunk_size: int) -> List[list]:
        return [items[i : i + chunk_size] for i in range(0, len(items), chunk_size)]

    def _post_embeddings(self, endpoint: str, payload: dict) -> torch.Tensor:
        resp = self.client.post(
            f"{self.server_url}{endpoint}",
            content=dumps_payload(payload),
            headers={"Content-Type": "application/octet-stream"},
        )
        resp.raise_for_status()
        out = loads_payload(resp.content)
        return out["embeddings"].cpu()

    def _encode_batched(self, endpoint: str, field_name: str, values: list) -> torch.Tensor:
        if not values:
            return torch.empty((0,))

        batch_size = self.request_batch_size or len(values)
        chunks = self._chunked(values, batch_size)

        #  avoids thread pool overhead if only one element
        if self.concurrency == 1 or len(chunks) == 1:
            tensors = []
            for chunk in chunks:
                payload = {"model": self.model_name, field_name: chunk}
                tensors.append(self._post_embeddings(endpoint, payload))
            return torch.cat(tensors, dim=0) if len(tensors) > 1 else tensors[0]

        with ThreadPoolExecutor(max_workers=self.concurrency) as pool:
            futures = []
            for idx, chunk in enumerate(chunks):
                payload = {"model": self.model_name, field_name: chunk}
                futures.append((idx, pool.submit(self._post_embeddings, endpoint, payload)))

            ordered = [None] * len(chunks)
            for idx, future in futures:
                ordered[idx] = future.result()

        return torch.cat(ordered, dim=0) if len(ordered) > 1 else ordered[0]

    def encode_images(self, images: List[Image.Image]) -> torch.Tensor:
        encoded = [self._image_to_bytes(image.convert("RGB")) for image in images]
        return self._encode_batched("/v1/embed/images", "images", encoded)

    def encode_queries(self, queries: List[str]) -> torch.Tensor:
        return self._encode_batched("/v1/embed/queries", "queries", queries)
