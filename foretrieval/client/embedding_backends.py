from __future__ import annotations

from typing import List, Protocol, Union

import torch
from PIL import Image

from .remote_backend import RemoteEmbeddingClient


class EmbeddingBackend(Protocol):
    def embed_images(self, images: List[Image.Image]) -> torch.Tensor:
        ...

    def embed_queries(self, queries: List[str]) -> torch.Tensor:
        ...


class LocalEmbeddingBackend:
    def __init__(self, model, processor, device):
        self.model = model
        self.processor = processor
        self.device = device

    def embed_images(self, images: List[Image.Image]) -> torch.Tensor:
        with torch.inference_mode():
            batch = self.processor.process_images(images)
            batch = {
                k: v.to(self.device).to(
                    self.model.dtype
                    if v.dtype in [torch.float16, torch.bfloat16, torch.float32]
                    else v.dtype
                )
                for k, v in batch.items()
            }
            embeddings = self.model(**batch)
        return embeddings.cpu()

    def embed_queries(self, queries: List[str]) -> torch.Tensor:
        with torch.inference_mode():
            batch = self.processor.process_queries(queries)
            batch = {
                k: v.to(self.device).to(
                    self.model.dtype
                    if v.dtype in [torch.float16, torch.bfloat16, torch.float32]
                    else v.dtype
                )
                for k, v in batch.items()
            }
            embeddings = self.model(**batch)
        return embeddings.cpu()


class RemoteEmbeddingBackend:
    def __init__(self, client: RemoteEmbeddingClient):
        self.client = client

    def embed_images(self, images: List[Image.Image]) -> torch.Tensor:
        return self.client.encode_images(images).cpu()

    def embed_queries(self, queries: List[str]) -> torch.Tensor:
        return self.client.encode_queries(queries).cpu()
