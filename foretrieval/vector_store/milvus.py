"""MilvusVectorStore — Milvus Lite backend for FORetrieval.

Milvus does not support native multi-vector (multi-dimensional) storage like
Qdrant's MAX_SIM.  Instead, this implementation uses a two-collection layout
that closely mirrors the RAG_Orch orchestrator in origin/Adrien/Docker-DB:

    <index_name>__pages   — one row per page; vector = mean-pooled page embedding
    <index_name>__tokens  — one row per query-token; grouped by page_id

Search is a two-stage process:
    1. Candidate retrieval: ANN search against __pages using the mean-pooled
       query vector. Retrieves candidate_limit pages.
    2. Late-interaction reranking: for each query token, search the __tokens
       collection filtered to candidate page IDs. Aggregate max scores per
       page across all query tokens → approximate MAX_SIM score.
    3. Return top-k pages sorted by aggregated score.

This closely approximates Qdrant's MAX_SIM while being compatible with Milvus's
FLOAT_VECTOR type (which does not support multi-dimensional vectors).

Milvus Lite (file-based, included in pymilvus>=2.4) is used for local/dev usage.
Production deployment uses the same API but with a network URI.

Heatmap / fetch_vector
----------------------
fetch_vector() retrieves all token rows for a given page_id from __tokens and
reconstructs the (n_tokens, dim) tensor, enabling the same heatmap path as
Qdrant and local backends.

Storage layout
--------------
    <index_root>/<index_name>/milvus.db  — Milvus Lite file
    
No additional sidecar files for vectors (embed_id_to_extra etc. still managed
by ColPaliModel).
"""

from __future__ import annotations

import json
import logging
import uuid
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional, Tuple

import torch

from .base import (
    MultiVectorQuery,
    SearchHit,
    StoredPoint,
    VectorStore,
)

logger = logging.getLogger(__name__)

_SUFFIX_PAGES = "__pages"
_SUFFIX_TOKENS = "__tokens"
_PAGE_VECTOR_FIELD = "page_vector"
_TOKEN_VECTOR_FIELD = "token_vector"
_TOKEN_GROUP_FIELD = "page_id"
_PAYLOAD_FIELD = "payload_json"
_PRIMARY_KEY_MAX_LEN = 64
_PAYLOAD_JSON_MAX_LEN = 65535

# Default number of candidate pages to fetch before late-interaction rerank
_DEFAULT_CANDIDATE_LIMIT = 64

try:
    from pymilvus import DataType, MilvusClient
    _MILVUS_AVAILABLE = True
except ImportError:
    _MILVUS_AVAILABLE = False


def _require_milvus() -> None:
    if not _MILVUS_AVAILABLE:
        raise RuntimeError(
            "The Milvus storage backend requires the pymilvus package.\n"
            "Install it with:  pip install \"foretrieval[milvus]\"\n"
            "or:               uv add foretrieval --extra milvus"
        )


def _page_id_str(point_id: int) -> str:
    """Convert integer point_id to the string key used as Milvus page_id."""
    return str(point_id)


def _mean_pool(vectors: torch.Tensor) -> List[float]:
    """Return the mean of a 2-D tensor as a Python list (for Milvus insert)."""
    return vectors.float().mean(dim=0).tolist()


def _serialize_payload(payload: Dict[str, Any]) -> str:
    return json.dumps(payload)[:_PAYLOAD_JSON_MAX_LEN]


def _deserialize_payload(raw: Optional[str]) -> Dict[str, Any]:
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except Exception:
        return {}


def _collection_names(index_name: str) -> Tuple[str, str]:
    return f"{index_name}{_SUFFIX_PAGES}", f"{index_name}{_SUFFIX_TOKENS}"


class MilvusVectorStore(VectorStore):
    """Two-collection Milvus store using approximate late-interaction scoring."""

    backend_name: ClassVar[str] = "milvus"
    supports_multivector_native: ClassVar[bool] = False  # uses approximation

    def __init__(self, candidate_limit: int = _DEFAULT_CANDIDATE_LIMIT) -> None:
        self._client: Optional["MilvusClient"] = None
        self._index_name: Optional[str] = None
        self._db_path: Optional[Path] = None
        self._candidate_limit = candidate_limit

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
        _require_milvus()
        self._index_name = index_name
        index_dir = Path(index_root) / index_name
        index_dir.mkdir(parents=True, exist_ok=True)
        self._db_path = index_dir / "milvus.db"
        self._client = MilvusClient(uri=str(self._db_path))

        if create and dim is not None and not self.collection_exists():
            self.create_collection(dim)

    def close(self) -> None:
        self._client = None

    # ------------------------------------------------------------------
    # Collection management
    # ------------------------------------------------------------------

    def collection_exists(self) -> bool:
        if self._client is None or self._index_name is None:
            return False
        page_col, _ = _collection_names(self._index_name)
        return page_col in self._client.list_collections()

    def create_collection(self, dim: int) -> None:
        if self._client is None or self._index_name is None:
            raise RuntimeError("MilvusVectorStore.open() must be called first.")
        page_col, token_col = _collection_names(self._index_name)
        self._create_page_collection(page_col, dim)
        self._create_token_collection(token_col, dim)
        self._load_collections()

    def _create_page_collection(self, collection_name: str, dim: int) -> None:
        if collection_name in self._client.list_collections():
            return
        schema = MilvusClient.create_schema(
            auto_id=False, enable_dynamic_field=False
        )
        schema.add_field(
            field_name="id",
            datatype=DataType.VARCHAR,
            is_primary=True,
            max_length=_PRIMARY_KEY_MAX_LEN,
        )
        schema.add_field(
            field_name=_PAGE_VECTOR_FIELD,
            datatype=DataType.FLOAT_VECTOR,
            dim=dim,
        )
        schema.add_field(
            field_name=_PAYLOAD_FIELD,
            datatype=DataType.VARCHAR,
            max_length=_PAYLOAD_JSON_MAX_LEN,
        )
        index_params = self._client.prepare_index_params()
        index_params.add_index(
            field_name=_PAGE_VECTOR_FIELD,
            index_type="AUTOINDEX",
            metric_type="COSINE",
        )
        self._client.create_collection(
            collection_name=collection_name,
            schema=schema,
            index_params=index_params,
        )

    def _create_token_collection(self, collection_name: str, dim: int) -> None:
        if collection_name in self._client.list_collections():
            return
        schema = MilvusClient.create_schema(
            auto_id=False, enable_dynamic_field=False
        )
        schema.add_field(
            field_name="id",
            datatype=DataType.VARCHAR,
            is_primary=True,
            max_length=_PRIMARY_KEY_MAX_LEN,
        )
        schema.add_field(
            field_name=_TOKEN_GROUP_FIELD,
            datatype=DataType.VARCHAR,
            max_length=_PRIMARY_KEY_MAX_LEN,
        )
        schema.add_field(
            field_name=_TOKEN_VECTOR_FIELD,
            datatype=DataType.FLOAT_VECTOR,
            dim=dim,
        )
        index_params = self._client.prepare_index_params()
        index_params.add_index(
            field_name=_TOKEN_VECTOR_FIELD,
            index_type="AUTOINDEX",
            metric_type="COSINE",
        )
        self._client.create_collection(
            collection_name=collection_name,
            schema=schema,
            index_params=index_params,
        )

    def _load_collections(self) -> None:
        if self._client is None or self._index_name is None:
            return
        page_col, token_col = _collection_names(self._index_name)
        for col in (page_col, token_col):
            if col in self._client.list_collections():
                self._client.load_collection(collection_name=col)

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def upsert(self, points: List[StoredPoint]) -> None:
        if self._client is None or self._index_name is None:
            raise RuntimeError("MilvusVectorStore.open() must be called first.")
        page_col, token_col = _collection_names(self._index_name)
        self._load_collections()

        page_rows: List[Dict[str, Any]] = []
        token_rows: List[Dict[str, Any]] = []

        for sp in points:
            page_id_str = _page_id_str(sp.point_id)
            vectors = sp.vector.float()  # (n_tokens, dim)

            page_rows.append(
                {
                    "id": page_id_str,
                    _PAGE_VECTOR_FIELD: _mean_pool(vectors),
                    _PAYLOAD_FIELD: _serialize_payload(sp.payload),
                }
            )
            for vec in vectors:
                token_rows.append(
                    {
                        "id": str(uuid.uuid4()),
                        _TOKEN_GROUP_FIELD: page_id_str,
                        _TOKEN_VECTOR_FIELD: vec.tolist(),
                    }
                )

        if page_rows:
            self._client.upsert(collection_name=page_col, data=page_rows)
        if token_rows:
            self._client.upsert(collection_name=token_col, data=token_rows)

    def point_exists(self, point_id: int) -> bool:
        if self._client is None or self._index_name is None:
            return False
        if not self.collection_exists():
            return False
        page_col, _ = _collection_names(self._index_name)
        self._load_collections()
        result = self._client.get(
            collection_name=page_col,
            ids=[_page_id_str(point_id)],
            output_fields=["id"],
        )
        return bool(result)

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def search(
        self,
        query: MultiVectorQuery,
        k: int,
    ) -> List[SearchHit]:
        if self._client is None or self._index_name is None:
            raise RuntimeError("MilvusVectorStore.open() must be called first.")
        self._load_collections()

        q_vectors = query.vectors.float()  # (n_query_tokens, dim)
        filter_expr = self._build_filter_expr(query.filter_metadata)

        candidate_ids, candidate_payloads = self._fetch_candidates(
            q_vectors, k, filter_expr
        )
        if not candidate_ids:
            return []

        scores = self._late_interaction_rerank(q_vectors, candidate_ids)

        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:k]

        return [
            SearchHit(
                point_id=int(page_id_str),
                score=float(score),
                payload=candidate_payloads.get(page_id_str, {}),
            )
            for page_id_str, score in ranked
        ]

    def _fetch_candidates(
        self,
        q_vectors: torch.Tensor,
        k: int,
        filter_expr: Optional[str],
    ) -> Tuple[List[str], Dict[str, Dict[str, Any]]]:
        page_col, _ = _collection_names(self._index_name)
        pooled_q = _mean_pool(q_vectors)
        candidate_limit = max(k * 10, self._candidate_limit)

        search_kwargs: Dict[str, Any] = dict(
            collection_name=page_col,
            data=[pooled_q],
            anns_field=_PAGE_VECTOR_FIELD,
            limit=candidate_limit,
            output_fields=[_PAYLOAD_FIELD],
            search_params={"metric_type": "COSINE"},
        )
        if filter_expr:
            search_kwargs["filter"] = filter_expr

        results = self._client.search(**search_kwargs)
        rows = results[0] if results else []

        candidate_ids: List[str] = []
        candidate_payloads: Dict[str, Dict[str, Any]] = {}
        for row in rows:
            page_id = str(row.get("id", ""))
            entity = row.get("entity") or {}
            if not page_id:
                continue
            candidate_ids.append(page_id)
            candidate_payloads[page_id] = _deserialize_payload(
                entity.get(_PAYLOAD_FIELD)
            )
        return candidate_ids, candidate_payloads

    def _late_interaction_rerank(
        self,
        q_vectors: torch.Tensor,
        candidate_ids: List[str],
    ) -> Dict[str, float]:
        _, token_col = _collection_names(self._index_name)
        quoted = ", ".join(f'"{pid}"' for pid in candidate_ids)
        filter_expr = f"{_TOKEN_GROUP_FIELD} in [{quoted}]"

        aggregated: Dict[str, float] = {pid: 0.0 for pid in candidate_ids}

        for q_vec in q_vectors:
            results = self._client.search(
                collection_name=token_col,
                data=[q_vec.tolist()],
                anns_field=_TOKEN_VECTOR_FIELD,
                filter=filter_expr,
                limit=min(len(candidate_ids), 16384),
                output_fields=[_TOKEN_GROUP_FIELD],
                search_params={"metric_type": "COSINE"},
                group_by_field=_TOKEN_GROUP_FIELD,
            )
            for row in (results[0] if results else []):
                entity = row.get("entity") or {}
                pid = str(entity.get(_TOKEN_GROUP_FIELD, ""))
                if pid:
                    aggregated[pid] = aggregated.get(pid, 0.0) + float(
                        row.get("distance", 0.0)
                    )
        return aggregated

    def fetch_vector(self, point_id: int) -> Optional[torch.Tensor]:
        """Reconstruct the multi-vector tensor from stored token rows."""
        if self._client is None or self._index_name is None:
            return None
        _, token_col = _collection_names(self._index_name)
        self._load_collections()
        page_id_str = _page_id_str(point_id)

        rows = self._client.query(
            collection_name=token_col,
            filter=f'{_TOKEN_GROUP_FIELD} == "{page_id_str}"',
            output_fields=[_TOKEN_VECTOR_FIELD],
            limit=16384,
        )
        if not rows:
            return None
        token_vecs = [row[_TOKEN_VECTOR_FIELD] for row in rows]
        return torch.tensor(token_vecs)  # (n_tokens, dim)

    # ------------------------------------------------------------------
    # Filter helper
    # ------------------------------------------------------------------

    def _build_filter_expr(
        self, filter_metadata: Optional[Dict[str, Any]]
    ) -> Optional[str]:
        """Build a Milvus filter expression from a key/value metadata dict.

        Metadata values are stored inside the JSON payload field, so filtering
        is done by checking the payload fields decoded from the payload_json
        column.  However, Milvus does not support JSON sub-field filtering on
        VARCHAR columns natively without dynamic fields.

        Strategy: since we store the full payload as JSON in _PAYLOAD_FIELD we
        cannot use structured sub-field filters.  The metadata filtering is
        therefore implemented as a post-filter on the Python side:
        candidates are fetched without a server-side filter, then filtered
        locally by deserialising the payload JSON.
        """
        # Note: server-side metadata filtering is not available in this layout
        # because payload is stored as a serialised JSON string in a VARCHAR
        # column (no dynamic field enabled).  We return None here and apply
        # the filter in _apply_metadata_filter() after candidate retrieval.
        return None

    def _filter_candidates_by_metadata(
        self,
        candidate_ids: List[str],
        candidate_payloads: Dict[str, Dict[str, Any]],
        filter_metadata: Dict[str, Any],
    ) -> Tuple[List[str], Dict[str, Dict[str, Any]]]:
        """Post-filter candidates by metadata values in payload dicts."""
        from ..utils import _value_match
        from ..models_metadata import MetadataFilter

        f = (
            filter_metadata
            if isinstance(filter_metadata, MetadataFilter)
            else MetadataFilter(**filter_metadata)
        )

        filtered_ids = []
        for pid in candidate_ids:
            payload = candidate_payloads.get(pid, {})
            # Metadata lives under "metadata" key inside payload
            meta = payload.get("metadata", payload)
            if _value_match(meta, f):
                filtered_ids.append(pid)

        return filtered_ids, {
            pid: candidate_payloads[pid]
            for pid in filtered_ids
            if pid in candidate_payloads
        }

    # Override search to apply metadata post-filter
    def search(  # type: ignore[override]  # noqa: F811
        self,
        query: MultiVectorQuery,
        k: int,
    ) -> List[SearchHit]:
        if self._client is None or self._index_name is None:
            raise RuntimeError("MilvusVectorStore.open() must be called first.")
        self._load_collections()

        q_vectors = query.vectors.float()

        candidate_ids, candidate_payloads = self._fetch_candidates(
            q_vectors, k, filter_expr=None
        )
        if not candidate_ids:
            return []

        # Apply metadata filter post-retrieval if requested
        if query.filter_metadata:
            candidate_ids, candidate_payloads = self._filter_candidates_by_metadata(
                candidate_ids, candidate_payloads, query.filter_metadata
            )
            if not candidate_ids:
                logger.warning(
                    "Metadata filter matched no candidates — returning empty results."
                )
                return []

        scores = self._late_interaction_rerank(q_vectors, candidate_ids)
        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:k]

        return [
            SearchHit(
                point_id=int(page_id_str),
                score=float(score),
                payload=candidate_payloads.get(page_id_str, {}),
            )
            for page_id_str, score in ranked
        ]

    # ------------------------------------------------------------------
    # Client accessor
    # ------------------------------------------------------------------

    @property
    def client(self) -> Optional["MilvusClient"]:
        return self._client
