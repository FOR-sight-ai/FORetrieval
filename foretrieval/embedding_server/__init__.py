"""Remote embedding server package for FORetrieval.

Provides:
- EmbeddingServerConfig  — Pydantic config model
- EmbeddingServerClient  — HTTP client for the vLLM /pooling endpoint
- EmbeddingServerManager — SSH-based Docker deployment manager
"""

from .client import EmbeddingServerClient
from .config import EmbeddingServerConfig
from .manager import EmbeddingServerManager

__all__ = [
    "EmbeddingServerConfig",
    "EmbeddingServerClient",
    "EmbeddingServerManager",
]
