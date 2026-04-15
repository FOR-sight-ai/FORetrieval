"""HTTP client for the remote vLLM embedding server.

Communicates with a vLLM instance serving a ColPali/ColQwen3.5 model via the
/pooling endpoint with task=token_embed.  Returns multi-vector embeddings as
CPU tensors, matching the shape produced by the local colpali-engine pipeline.

OOM handling: the server may return HTTP 500 with an OOM message when a batch
is too large.  The client detects this and retries with progressively halved
batch sizes down to 1.
"""

from __future__ import annotations

import base64
import io
import logging
from typing import List

import requests
import torch
from PIL import Image

from .config import EmbeddingServerConfig

logger = logging.getLogger(__name__)

# Substrings in vLLM error responses that indicate GPU OOM.
_OOM_MARKERS = (
    "CUDA out of memory",
    "out of memory",
    "OOM",
    "RESOURCE_EXHAUSTED",
)

_POOLING_ENDPOINT = "/pooling"
_HEALTH_ENDPOINT = "/health"


class ServerOOMError(RuntimeError):
    """Raised when the server reports a GPU out-of-memory condition."""


class EmbeddingServerClient:
    """HTTP client that talks to a vLLM /pooling endpoint.

    Parameters
    ----------
    config:
        EmbeddingServerConfig with url, model_name, batch_size, etc.
    """

    def __init__(self, config: EmbeddingServerConfig) -> None:
        self.config = config
        self._session = requests.Session()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def health_check(self) -> bool:
        """Return True if the server is reachable and healthy."""
        try:
            resp = self._session.get(
                self.config.url + _HEALTH_ENDPOINT,
                timeout=10,
            )
            return resp.status_code == 200
        except requests.RequestException:
            return False

    def embed_images(self, images: List[Image.Image]) -> List[torch.Tensor]:
        """Embed a list of PIL images via the remote server.

        Sends images in batches to /pooling (task=token_embed).  Automatically
        reduces batch size on OOM until batch_size reaches 1.

        Parameters
        ----------
        images:
            List of PIL Images (document pages).

        Returns
        -------
        List of CPU tensors, one per image, each of shape [n_tokens, embed_dim].
        """
        if not images:
            return []
        return self._embed_images_with_oom_retry(images)

    def embed_query(self, query: str) -> List[torch.Tensor]:
        """Embed a text query string via the remote server.

        Parameters
        ----------
        query:
            Plain text query.

        Returns
        -------
        List containing a single CPU tensor of shape [n_tokens, embed_dim].
        """
        payload = {
            "model": self.config.model_name,
            "input": query,
            "task": "token_embed",
        }
        resp = self._post_pooling(payload)
        data = resp["data"]
        if not data:
            raise ValueError("Server returned empty data for query embedding")
        return [torch.tensor(data[0]["data"], dtype=torch.float32)]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _embed_images_with_oom_retry(
        self, images: List[Image.Image]
    ) -> List[torch.Tensor]:
        """Embed images, halving batch size on OOM until batch_size=1."""
        batch_size = self.config.batch_size
        while batch_size >= 1:
            try:
                return self._embed_images_batched(images, batch_size)
            except ServerOOMError:
                new_size = batch_size // 2
                if new_size < 1:
                    logger.error(
                        "Server OOM even at batch_size=1. "
                        "The model may be too large for the available GPU memory."
                    )
                    raise
                logger.warning(
                    "Server OOM at batch_size=%d — retrying with batch_size=%d",
                    batch_size,
                    new_size,
                )
                batch_size = new_size
        # Unreachable, but satisfies type checkers.
        raise ServerOOMError("OOM at all batch sizes")

    def _embed_images_batched(
        self, images: List[Image.Image], batch_size: int
    ) -> List[torch.Tensor]:
        """Split images into batches, call server, collect tensors."""
        results: List[torch.Tensor] = []
        for start in range(0, len(images), batch_size):
            batch = images[start : start + batch_size]
            tensors = self._embed_image_batch(batch)
            results.extend(tensors)
        return results

    def _embed_image_batch(
        self, images: List[Image.Image]
    ) -> List[torch.Tensor]:
        """Embed a single batch of images (no retry logic here)."""
        input_items = []
        for img in images:
            b64 = _pil_to_base64(img)
            input_items.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{b64}"},
                }
            )

        payload = {
            "model": self.config.model_name,
            "input": input_items,
            "task": "token_embed",
        }
        resp = self._post_pooling(payload)

        tensors = []
        for item in resp["data"]:
            tensors.append(torch.tensor(item["data"], dtype=torch.float32))
        return tensors

    def _post_pooling(self, payload: dict) -> dict:
        """POST to /pooling and return parsed JSON.  Raises ServerOOMError on OOM."""
        try:
            resp = self._session.post(
                self.config.url + _POOLING_ENDPOINT,
                json=payload,
                timeout=self.config.request_timeout,
            )
        except requests.Timeout as exc:
            raise TimeoutError(
                f"Embedding server timed out after {self.config.request_timeout}s"
            ) from exc
        except requests.ConnectionError as exc:
            raise ConnectionError(
                f"Cannot reach embedding server at {self.config.url}"
            ) from exc

        if resp.status_code != 200:
            body = resp.text
            if any(marker in body for marker in _OOM_MARKERS):
                raise ServerOOMError(f"Server OOM: {body[:300]}")
            raise RuntimeError(
                f"Embedding server returned HTTP {resp.status_code}: {body[:500]}"
            )

        return resp.json()


# ------------------------------------------------------------------
# Utility
# ------------------------------------------------------------------

def _pil_to_base64(img: Image.Image) -> str:
    """Encode a PIL image as a base64 PNG string."""
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")
