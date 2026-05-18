"""Factory for VectorStore instances.

Usage:
    from foretrieval.vector_store import make_vector_store

    vs = make_vector_store("qdrant")
    vs.open("my_index", Path(".foretrieval"), create=True, dim=128)

Supported backend names:
    "local"  — LocalVectorStore
    "qdrant" — QdrantVectorStore  (requires foretrieval[qdrant])
    "milvus" — MilvusVectorStore  (requires foretrieval[milvus])

Future backends can be registered via BACKEND_REGISTRY without changing callers.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Type

from .base import VectorStore
from .local import LocalVectorStore
from .qdrant import QdrantVectorStore
from .milvus import MilvusVectorStore

# Registry maps backend name → class.  Add new entries here to register future
# backends (e.g. "remote" → RemoteVectorStore) without touching make_vector_store().
BACKEND_REGISTRY: Dict[str, Type[VectorStore]] = {
    "local": LocalVectorStore,
    "qdrant": QdrantVectorStore,
    "milvus": MilvusVectorStore,
}


def make_vector_store(
    backend: str,
    storage_config: Optional[Dict[str, Any]] = None,
) -> VectorStore:
    """Instantiate a VectorStore for the given backend name.

    Parameters:
        backend:        Backend identifier, e.g. "local", "qdrant", "milvus".
        storage_config: Optional dict of backend-specific keyword arguments
                        passed to the VectorStore constructor.  Currently only
                        MilvusVectorStore accepts ``candidate_limit``.

    Returns:
        An uninitialised VectorStore.  Call .open() before using it.

    Raises:
        ValueError:   Unknown backend name.
        RuntimeError: Required optional dependency not installed.
    """
    key = (backend or "local").strip().lower()
    cls = BACKEND_REGISTRY.get(key)
    if cls is None:
        supported = ", ".join(sorted(BACKEND_REGISTRY))
        raise ValueError(
            f"Unknown storage backend {backend!r}. "
            f"Supported backends: {supported}."
        )

    kwargs = dict(storage_config or {})

    # Pass only kwargs that the constructor accepts
    if cls is LocalVectorStore:
        return LocalVectorStore()

    if cls is QdrantVectorStore:
        return QdrantVectorStore()

    if cls is MilvusVectorStore:
        candidate_limit = kwargs.get("candidate_limit", 64)
        return MilvusVectorStore(candidate_limit=int(candidate_limit))

    # Generic fallback for future registered backends
    return cls(**kwargs)  # type: ignore[call-arg]
