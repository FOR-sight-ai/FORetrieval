"""Remote vector-DB server package for FORetrieval.

Provides:
- VectorDBServerConfig  — Pydantic config model
- VectorDBServerClient  — HTTP client for the vector-DB API
- VectorDBServerManager — SSH-based Docker deployment manager
"""

from .client import VectorDBServerClient
from .config import VectorDBServerConfig
from .manager import VectorDBServerManager

__all__ = [
    "VectorDBServerConfig",
    "VectorDBServerClient",
    "VectorDBServerManager",
]
