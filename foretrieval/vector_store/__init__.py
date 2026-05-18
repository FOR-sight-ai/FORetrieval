"""foretrieval.vector_store — generic vector-store interface for FORetrieval.

Public API:
    VectorStore         — abstract base class
    StoredPoint         — dataclass for indexed items
    SearchHit           — dataclass for search results
    MultiVectorQuery    — dataclass for search queries
    make_point_id       — deterministic integer point-ID helper
    make_vector_store   — factory function

    LocalVectorStore    — in-memory + .pt/.json.gz file backend
    QdrantVectorStore   — embedded Qdrant (local path) backend
    MilvusVectorStore   — Milvus Lite two-collection backend
"""

from .base import (
    VectorStore,
    StoredPoint,
    SearchHit,
    MultiVectorQuery,
    make_point_id,
)
from .local import LocalVectorStore
from .qdrant import QdrantVectorStore
from .milvus import MilvusVectorStore
from .factory import make_vector_store, BACKEND_REGISTRY

__all__ = [
    "VectorStore",
    "StoredPoint",
    "SearchHit",
    "MultiVectorQuery",
    "make_point_id",
    "make_vector_store",
    "BACKEND_REGISTRY",
    "LocalVectorStore",
    "QdrantVectorStore",
    "MilvusVectorStore",
]
