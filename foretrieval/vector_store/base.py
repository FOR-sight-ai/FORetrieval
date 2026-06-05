"""Generic vector-store interface for FORetrieval.

This module defines the VectorStore ABC and the shared dataclasses used by all
backend implementations (local, Qdrant, Milvus) and by the future remote-DB-server
client (RemoteVectorStore).

Design for remote-server forward-compatibility
-----------------------------------------------
Every method on VectorStore takes/returns plain Python dataclasses whose fields
are either primitive types, dicts, or torch.Tensors — all serialisable to JSON or
to torch.save().  A future RemoteVectorStore subclass will simply wrap an httpx
client and forward each call to a corresponding HTTP endpoint:

    open()           →  POST /v1/collection/open    (or GET /health + lazy init)
    close()          →  (disconnect client)
    collection_exists() → GET /v1/collection/{name}
    create_collection() → POST /v1/collection
    upsert()         →  POST /v1/upsert              (body: list[StoredPoint])
    point_exists()   →  GET  /v1/point/{point_id}
    search()         →  POST /v1/search              (body: MultiVectorQuery)
    fetch_vector()   →  GET  /v1/vector/{point_id}
    export_sidecar() →  no-op for remote
    load_sidecar()   →  no-op for remote

The factory function make_vector_store() in factory.py will accept "remote" as a
backend name and return a RemoteVectorStore when that module is implemented.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional

import torch


# ---------------------------------------------------------------------------
# Shared point-ID helper (same formula used everywhere, including Qdrant side-
# cars and local embed_id_to_doc_id)
# ---------------------------------------------------------------------------

def make_point_id(doc_id: int, page_id: int, chunk_id: Optional[int] = None) -> int:
    """Compute a deterministic integer point ID from (doc_id, page_id, chunk_id).

    The formula guarantees uniqueness as long as:
      - doc_id  < 10^7
      - page_id < 10^4
      - chunk_id < 10^4
    which comfortably covers any realistic corpus.
    """
    chunk_val = 0 if chunk_id is None else int(chunk_id)
    return int(doc_id) * 10_000_000 + int(page_id) * 10_000 + chunk_val


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class StoredPoint:
    """One indexed item (a page or chunk) ready to be written to the store.

    Attributes:
        point_id:  Deterministic integer key (output of make_point_id).
        vector:    Multi-vector embedding, shape (n_tokens, dim).  CPU tensor.
        payload:   Arbitrary metadata dict stored alongside the vector.
                   Must be JSON-serialisable.
    """
    point_id: int
    vector: torch.Tensor          # (n_tokens, dim)
    payload: Dict[str, Any]       # doc_id, page_id, chunk_id, metadata, …


@dataclass
class SearchHit:
    """One result returned by VectorStore.search().

    Attributes:
        point_id: Integer key that matches StoredPoint.point_id.
        score:    Relevance score (higher is better for all backends).
        payload:  Full payload dict as stored at index time.
    """
    point_id: int
    score: float
    payload: Dict[str, Any]


@dataclass
class MultiVectorQuery:
    """A multi-vector query embedding ready to be sent to VectorStore.search().

    Attributes:
        vectors:           Query embeddings, shape (n_query_tokens, dim).  CPU tensor.
        filter_metadata:   Optional key/value dict for payload filtering.
    """
    vectors: torch.Tensor                        # (n_query_tokens, dim)
    filter_metadata: Optional[Dict[str, Any]] = field(default=None)


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------

class VectorStore(abc.ABC):
    """Abstract interface for a vector store backend.

    Concrete subclasses:
        LocalVectorStore   — in-memory + .pt/.json.gz files
        QdrantVectorStore  — embedded Qdrant (local path)
        MilvusVectorStore  — Milvus Lite (file-based)
        RemoteVectorStore  — future HTTP client to a remote DB server

    Lifecycle
    ---------
    Instances are created via make_vector_store() then initialised with open().
    open() is called once per index name (either creating or loading the store).
    close() is called on teardown (context manager not required but recommended).

    Backend implementations must be stateless with respect to the index name
    until open() is called — this allows the same VectorStore object to be
    reused for multiple index names by calling open() again after close().
    """

    backend_name: ClassVar[str]
    """Identifier string, e.g. "local", "qdrant", "milvus"."""

    supports_multivector_native: ClassVar[bool]
    """True if the backend natively scores multi-vector queries (e.g. Qdrant MAX_SIM).
    False means the implementation uses an approximation (mean-pool + re-rank)."""

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def open(
        self,
        index_name: str,
        index_root: Path,
        *,
        create: bool,
        dim: Optional[int] = None,
    ) -> None:
        """Initialise or connect to the store for a given index.

        Parameters:
            index_name: Logical name for the collection / index directory.
            index_root: Root directory where local files live.
            create:     If True, create the collection if it does not exist.
                        dim must be provided when create=True and the collection
                        does not yet exist.
            dim:        Embedding dimension.  Required when create=True and the
                        collection doesn't exist yet.
        """

    @abc.abstractmethod
    def close(self) -> None:
        """Release any resources held by the store (clients, file handles)."""

    # ------------------------------------------------------------------
    # Collection management
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def collection_exists(self) -> bool:
        """Return True if the collection backing this store already exists."""

    @abc.abstractmethod
    def create_collection(self, dim: int) -> None:
        """Create the collection with the given embedding dimension.

        No-op if the collection already exists.
        """

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def upsert(self, points: List[StoredPoint]) -> None:
        """Write a list of StoredPoints into the store.

        Implementations must be idempotent: upserting the same point_id twice
        must not raise an error (last-write-wins semantics).
        """

    @abc.abstractmethod
    def point_exists(self, point_id: int) -> bool:
        """Return True if a point with the given ID already exists in the store."""

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def search(
        self,
        query: MultiVectorQuery,
        k: int,
    ) -> List[SearchHit]:
        """Execute a nearest-neighbour search and return up to k hits.

        Parameters:
            query:  Multi-vector query embedding + optional metadata filter.
            k:      Maximum number of results to return.

        Returns:
            List of SearchHit, sorted by descending score.
        """

    @abc.abstractmethod
    def fetch_vector(self, point_id: int) -> Optional[torch.Tensor]:
        """Retrieve the stored multi-vector tensor for a given point.

        Returns:
            CPU tensor of shape (n_tokens, dim), or None if not found.

        Used by the heatmap / visual-grounding code path that needs the original
        per-token embeddings to compute patch attention scores.
        """

    # ------------------------------------------------------------------
    # Persistence helpers (called by ColPaliModel._export_index / from_index)
    # ------------------------------------------------------------------

    def export_sidecar(self, index_path: Path) -> None:
        """Persist any backend-specific sidecar data into index_path.

        Default implementation is a no-op (e.g. for remote backends where the
        data lives on the server).  Local/embedded backends override this to
        write .pt / .json.gz files.
        """

    def load_sidecar(self, index_path: Path) -> None:
        """Load backend-specific sidecar data from index_path.

        Default implementation is a no-op.  Matches export_sidecar().
        """

    # ------------------------------------------------------------------
    # Index-level bookkeeping (model name, doc metadata, file-name map,
    # per-embedding extras, …).  For local/embedded backends this lives in
    # local sidecar files written by ColPaliModel.  For the remote backend it
    # is round-tripped to the server so the client needs no local index dir.
    # ------------------------------------------------------------------

    def supports_remote_bookkeeping(self) -> bool:
        """Whether this backend persists ColPali bookkeeping on the server.

        Local/embedded backends return False (ColPaliModel keeps writing the
        local sidecar files).  RemoteVectorStore returns True so ColPaliModel
        routes the bookkeeping blob through export/load_bookkeeping instead of
        touching the local filesystem.
        """
        return False

    def export_bookkeeping(self, blob: Dict[str, Any]) -> None:
        """Persist the ColPali bookkeeping ``blob`` on the server.

        ``blob`` is a plain dict whose values are JSON-serialisable or
        torch.Tensors (it is transported with torch.save).  Default no-op;
        only RemoteVectorStore implements this.
        """

    def load_bookkeeping(self) -> Optional[Dict[str, Any]]:
        """Load the ColPali bookkeeping blob from the server.

        Returns ``None`` when no bookkeeping is stored (or for backends that
        do not support remote bookkeeping).  Default no-op.
        """
        return None
