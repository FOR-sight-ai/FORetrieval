"""RemoteVectorStore — VectorStore client that delegates to a remote server.

This is the client-side VectorStore implementation.  It holds a
VectorDBServerClient and implements the VectorStore ABC by forwarding every
call to the corresponding HTTP endpoint on the remote server.

``export_sidecar`` and ``load_sidecar`` are no-ops because the data lives on
the server, not on the client filesystem.

Usage:
    From make_vector_store() factory (preferred):

        vs = make_vector_store(
            "remote",
            {
                "url": "http://gpu-server:18000",
                "backend": "qdrant",        # server-side backend
                "api_key": "secret",        # optional
            }
        )
        vs.open("my_index", Path(".foretrieval"), create=True, dim=128)

    Or directly:

        from foretrieval.vector_db_server import VectorDBServerClient, VectorDBServerConfig
        from foretrieval.vector_store.remote import RemoteVectorStore

        cfg = VectorDBServerConfig(url="http://gpu-server:18000", backend="qdrant")
        client = VectorDBServerClient(cfg)
        vs = RemoteVectorStore(client, backend="qdrant")
        vs.open("my_index", Path("."), create=True, dim=128)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional

import torch

from .base import MultiVectorQuery, SearchHit, StoredPoint, VectorStore


class RemoteVectorStore(VectorStore):
    """VectorStore that delegates to a remote FORetrieval vector-DB server.

    Parameters
    ----------
    client:
        VectorDBServerClient connected to the target server.
    backend:
        Server-side storage backend (``"local"``, ``"qdrant"``, or
        ``"milvus"``).  Used when creating a new collection.
    storage_config:
        Optional backend-specific config forwarded to the server
        (e.g. ``{"candidate_limit": 128}`` for Milvus).
    """

    backend_name: ClassVar[str] = "remote"
    supports_multivector_native: ClassVar[bool] = True
    # (informational — depends on the server-side backend, but we report True
    # since qdrant/local are both exact; milvus is approximate but this
    # attribute is only used for informational logging)

    def __init__(
        self,
        client: Any,  # VectorDBServerClient — avoid circular import at module level
        backend: str = "qdrant",
        storage_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._client = client
        self._backend = backend
        self._storage_config = storage_config

        # Set by open()
        self._index_name: Optional[str] = None
        self._opened: bool = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def open(
        self,
        index_name: str,
        index_root: Path,
        *,
        create: bool,
        dim: Optional[int] = None,
    ) -> None:
        """Open or create the remote collection.

        ``index_root`` is ignored — data lives on the server.
        """
        self._index_name = index_name
        self._client.open_collection(
            index_name,
            self._backend,
            create=create,
            dim=dim,
            storage_config=self._storage_config,
        )
        self._opened = True

    def close(self) -> None:
        """Close the underlying HTTP client."""
        self._client.close()
        self._opened = False

    # ------------------------------------------------------------------
    # Collection management
    # ------------------------------------------------------------------

    def collection_exists(self) -> bool:
        if self._index_name is None:
            return False
        return self._client.collection_exists(self._index_name)

    def create_collection(self, dim: int) -> None:
        if self._index_name is None:
            raise RuntimeError("Call open() before create_collection()")
        self._client.create_collection(
            self._index_name, self._backend, dim, self._storage_config
        )

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def upsert(self, points: List[StoredPoint]) -> None:
        if self._index_name is None:
            raise RuntimeError("Call open() before upsert()")
        self._client.upsert(self._index_name, points)

    def point_exists(self, point_id: int) -> bool:
        if self._index_name is None:
            return False
        return self._client.point_exists(self._index_name, point_id)

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def search(
        self,
        query: MultiVectorQuery,
        k: int,
    ) -> List[SearchHit]:
        if self._index_name is None:
            raise RuntimeError("Call open() before search()")
        return self._client.search(self._index_name, query, k)

    def fetch_vector(self, point_id: int) -> Optional[torch.Tensor]:
        if self._index_name is None:
            return None
        return self._client.fetch_vector(self._index_name, point_id)

    # ------------------------------------------------------------------
    # Persistence helpers — no-ops for remote store
    # ------------------------------------------------------------------

    def export_sidecar(self, index_path: Path) -> None:
        """No-op — data lives on the server."""

    def load_sidecar(self, index_path: Path) -> None:
        """No-op — data lives on the server."""

    # ------------------------------------------------------------------
    # Internal accessors (for ColPaliModel compatibility checks)
    # ------------------------------------------------------------------

    @property
    def is_opened(self) -> bool:
        return self._opened

    @property
    def server_backend(self) -> str:
        return self._backend
