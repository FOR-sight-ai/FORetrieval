"""QdrantVectorStore — embedded Qdrant backend for FORetrieval.

Uses a local QdrantClient(path=...) so no external server is required.
The collection is configured with native multi-vector MAX_SIM scoring, which
gives exact late-interaction ColPali results without any approximation.

Remote Qdrant (URL-based) is intentionally out of scope for now; it will be
handled by a future RemoteVectorStore that wraps an HTTP client to a dedicated
DB server.

Metadata filtering
------------------
Filters are translated from the generic dict representation into Qdrant's
Filter(must=[FieldCondition(key="metadata.<k>", match=MatchValue(v))]) form.
Values may be strings, ints, or lists (via MetadataFilter.regex / range fields).

Storage layout
--------------
Vectors and payloads live inside the embedded Qdrant database at:
    <index_root>/<index_name>/qdrant/

No additional sidecar files are needed for the vector data itself (unlike local).
The embed_id_to_extra, doc_ids_to_file_names, metadata, and index_config sidecars
are still managed by ColPaliModel directly.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional

import torch

from .base import (
    MultiVectorQuery,
    SearchHit,
    StoredPoint,
    VectorStore,
)

logger = logging.getLogger(__name__)

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import (
        Distance,
        FieldCondition,
        Filter,
        MatchValue,
        MultiVectorComparator,
        MultiVectorConfig,
        PointStruct,
        VectorParams,
    )
    _QDRANT_AVAILABLE = True
except ImportError:
    _QDRANT_AVAILABLE = False


def _require_qdrant() -> None:
    if not _QDRANT_AVAILABLE:
        raise RuntimeError(
            "The Qdrant storage backend requires the qdrant-client package.\n"
            "Install it with:  pip install \"foretrieval[qdrant]\"\n"
            "or:               uv add foretrieval --extra qdrant"
        )


class QdrantVectorStore(VectorStore):
    """Embedded Qdrant vector store with native multi-vector MAX_SIM scoring."""

    backend_name: ClassVar[str] = "qdrant"
    supports_multivector_native: ClassVar[bool] = True

    def __init__(self) -> None:
        self._client: Optional["QdrantClient"] = None
        self._collection_name: Optional[str] = None
        self._index_root: Optional[Path] = None

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
        _require_qdrant()
        self._collection_name = index_name
        self._index_root = Path(index_root)
        qdrant_path = self._index_root / index_name / "qdrant"
        qdrant_path.mkdir(parents=True, exist_ok=True)
        self._client = QdrantClient(path=str(qdrant_path))

        if create and dim is not None and not self._client.collection_exists(index_name):
            self.create_collection(dim)

    def close(self) -> None:
        self._client = None

    # ------------------------------------------------------------------
    # Collection management
    # ------------------------------------------------------------------

    def collection_exists(self) -> bool:
        if self._client is None or self._collection_name is None:
            return False
        return self._client.collection_exists(self._collection_name)

    def create_collection(self, dim: int) -> None:
        if self._client is None or self._collection_name is None:
            raise RuntimeError("QdrantVectorStore.open() must be called first.")
        if self._client.collection_exists(self._collection_name):
            return
        self._client.create_collection(
            collection_name=self._collection_name,
            vectors_config=VectorParams(
                size=dim,
                distance=Distance.COSINE,
                multivector_config=MultiVectorConfig(
                    comparator=MultiVectorComparator.MAX_SIM,
                ),
            ),
        )

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def upsert(self, points: List[StoredPoint]) -> None:
        if self._client is None or self._collection_name is None:
            raise RuntimeError("QdrantVectorStore.open() must be called first.")
        qdrant_points = [
            PointStruct(
                id=sp.point_id,
                vector=sp.vector.float().numpy().tolist(),
                payload=sp.payload,
            )
            for sp in points
        ]
        self._client.upsert(
            collection_name=self._collection_name,
            points=qdrant_points,
        )

    def point_exists(self, point_id: int) -> bool:
        if self._client is None or self._collection_name is None:
            return False
        if not self._client.collection_exists(self._collection_name):
            return False
        found = self._client.retrieve(
            collection_name=self._collection_name,
            ids=[point_id],
            with_payload=False,
            with_vectors=False,
        )
        return bool(found)

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def search(
        self,
        query: MultiVectorQuery,
        k: int,
    ) -> List[SearchHit]:
        if self._client is None or self._collection_name is None:
            raise RuntimeError("QdrantVectorStore.open() must be called first.")

        qfilter = self._build_filter(query.filter_metadata)

        response = self._client.query_points(
            collection_name=self._collection_name,
            query=query.vectors.float().numpy().tolist(),
            query_filter=qfilter,
            limit=k,
            with_payload=True,
            with_vectors=False,
        )

        points = response.points if hasattr(response, "points") else response

        return [
            SearchHit(
                point_id=int(p.id),
                score=float(p.score),
                payload=p.payload or {},
            )
            for p in points
        ]

    def fetch_vector(self, point_id: int) -> Optional[torch.Tensor]:
        if self._client is None or self._collection_name is None:
            return None
        retrieved = self._client.retrieve(
            collection_name=self._collection_name,
            ids=[point_id],
            with_payload=False,
            with_vectors=True,
        )
        if not retrieved:
            return None
        return torch.tensor(retrieved[0].vector)

    # ------------------------------------------------------------------
    # Filter helper
    # ------------------------------------------------------------------

    def _build_filter(
        self, filter_metadata: Optional[Dict[str, Any]]
    ) -> Optional["Filter"]:
        if not filter_metadata:
            return None
        must = [
            FieldCondition(
                key=f"metadata.{k}",
                match=MatchValue(value=v),
            )
            for k, v in filter_metadata.items()
        ]
        return Filter(must=must)

    # ------------------------------------------------------------------
    # Client accessor (for testing / ColPaliModel compatibility)
    # ------------------------------------------------------------------

    @property
    def client(self) -> Optional["QdrantClient"]:
        return self._client

    @property
    def collection_name(self) -> Optional[str]:
        return self._collection_name
