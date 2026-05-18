"""Tests for MilvusVectorStore.

Unit tests mock the MilvusClient.
The slow integration test uses Milvus Lite (file-based, no external server needed).
"""
from __future__ import annotations

import tempfile
import uuid
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest
import torch

from foretrieval.vector_store.base import (
    MultiVectorQuery,
    StoredPoint,
    make_point_id,
)
from foretrieval.vector_store.milvus import (
    MilvusVectorStore,
    _MILVUS_AVAILABLE,
    _collection_names,
    _mean_pool,
    _page_id_str,
    _deserialize_payload,
    _serialize_payload,
)


# ---------------------------------------------------------------------------
# Helper builders
# ---------------------------------------------------------------------------

def _point(doc_id: int, page_id: int, n_tokens: int = 4, dim: int = 8) -> StoredPoint:
    pid = make_point_id(doc_id, page_id)
    return StoredPoint(
        point_id=pid,
        vector=torch.rand(n_tokens, dim),
        payload={"doc_id": doc_id, "page_id": page_id, "chunk_id": None, "metadata": {}},
    )


def _make_mock_store(index_name: str = "test_idx") -> tuple[MilvusVectorStore, MagicMock]:
    store = MilvusVectorStore.__new__(MilvusVectorStore)
    mock_client = MagicMock()
    mock_client.list_collections.return_value = []
    store._client = mock_client
    store._index_name = index_name
    store._db_path = Path("/tmp/fake.db")
    store._candidate_limit = 64
    return store, mock_client


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------

class TestHelpers:
    def test_mean_pool_correct_shape(self):
        t = torch.ones(4, 8)
        pool = _mean_pool(t)
        assert len(pool) == 8

    def test_mean_pool_values(self):
        t = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        pool = _mean_pool(t)
        assert pool == pytest.approx([2.0, 3.0])

    def test_serialize_deserialize_roundtrip(self):
        payload = {"doc_id": 1, "page_id": 2, "metadata": {"cat": "A"}}
        raw = _serialize_payload(payload)
        result = _deserialize_payload(raw)
        assert result["doc_id"] == 1

    def test_collection_names_suffix(self):
        page_col, token_col = _collection_names("myindex")
        assert page_col == "myindex__pages"
        assert token_col == "myindex__tokens"

    def test_page_id_str(self):
        pid = make_point_id(5, 3)
        assert _page_id_str(pid) == str(pid)


# ---------------------------------------------------------------------------
# Optional-dependency guard
# ---------------------------------------------------------------------------

class TestOptionalDepGuard:
    def test_missing_milvus_raises_on_open(self, tmp_path):
        with patch("foretrieval.vector_store.milvus._MILVUS_AVAILABLE", False):
            store = MilvusVectorStore()
            with pytest.raises(RuntimeError, match="pymilvus"):
                store.open("idx", tmp_path, create=True, dim=8)


# ---------------------------------------------------------------------------
# Collection management
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _MILVUS_AVAILABLE, reason="pymilvus not installed")
class TestCollectionManagement:
    def test_collection_not_exists_empty_list(self):
        store, mock_client = _make_mock_store()
        mock_client.list_collections.return_value = []
        assert not store.collection_exists()

    def test_collection_exists_when_page_col_present(self):
        store, mock_client = _make_mock_store()
        page_col, _ = _collection_names("test_idx")
        mock_client.list_collections.return_value = [page_col]
        assert store.collection_exists()

    def test_create_collection_creates_both_collections(self):
        store, mock_client = _make_mock_store()
        mock_client.list_collections.return_value = []
        mock_client.create_collection = MagicMock()
        mock_client.prepare_index_params.return_value = MagicMock()
        store.create_collection(dim=8)
        assert mock_client.create_collection.call_count == 2


# ---------------------------------------------------------------------------
# Upsert
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _MILVUS_AVAILABLE, reason="pymilvus not installed")
class TestUpsert:
    def test_upsert_inserts_page_rows(self):
        store, mock_client = _make_mock_store()
        mock_client.list_collections.return_value = [
            "test_idx__pages", "test_idx__tokens"
        ]
        sp = _point(0, 1, n_tokens=3, dim=8)
        store.upsert([sp])
        page_col, token_col = _collection_names("test_idx")
        upsert_calls = {c.kwargs["collection_name"]: c for c in mock_client.upsert.call_args_list}
        assert page_col in upsert_calls

    def test_upsert_inserts_token_rows(self):
        store, mock_client = _make_mock_store()
        mock_client.list_collections.return_value = [
            "test_idx__pages", "test_idx__tokens"
        ]
        n_tokens = 5
        sp = _point(0, 1, n_tokens=n_tokens, dim=8)
        store.upsert([sp])
        page_col, token_col = _collection_names("test_idx")
        upsert_calls = {c.kwargs["collection_name"]: c for c in mock_client.upsert.call_args_list}
        assert token_col in upsert_calls
        token_rows = upsert_calls[token_col].kwargs["data"]
        assert len(token_rows) == n_tokens

    def test_upsert_page_vector_is_mean_pooled(self):
        store, mock_client = _make_mock_store()
        mock_client.list_collections.return_value = [
            "test_idx__pages", "test_idx__tokens"
        ]
        vec = torch.ones(3, 4)  # all ones → mean pool = [1,1,1,1]
        pid = make_point_id(0, 1)
        sp = StoredPoint(
            point_id=pid,
            vector=vec,
            payload={"doc_id": 0, "page_id": 1, "chunk_id": None, "metadata": {}},
        )
        store.upsert([sp])
        page_col, _ = _collection_names("test_idx")
        upsert_calls = {c.kwargs["collection_name"]: c for c in mock_client.upsert.call_args_list}
        page_row = upsert_calls[page_col].kwargs["data"][0]
        assert page_row["page_vector"] == pytest.approx([1.0, 1.0, 1.0, 1.0])

    def test_upsert_token_rows_share_page_id(self):
        store, mock_client = _make_mock_store()
        mock_client.list_collections.return_value = [
            "test_idx__pages", "test_idx__tokens"
        ]
        pid = make_point_id(0, 1)
        sp = _point(0, 1, n_tokens=3)
        store.upsert([sp])
        _, token_col = _collection_names("test_idx")
        upsert_calls = {c.kwargs["collection_name"]: c for c in mock_client.upsert.call_args_list}
        token_rows = upsert_calls[token_col].kwargs["data"]
        page_ids = {row["page_id"] for row in token_rows}
        assert page_ids == {_page_id_str(pid)}

    def test_point_exists_true(self):
        store, mock_client = _make_mock_store()
        mock_client.list_collections.return_value = ["test_idx__pages"]
        mock_client.get.return_value = [{"id": "100"}]
        assert store.point_exists(100)

    def test_point_exists_false(self):
        store, mock_client = _make_mock_store()
        mock_client.list_collections.return_value = ["test_idx__pages"]
        mock_client.get.return_value = []
        assert not store.point_exists(100)


# ---------------------------------------------------------------------------
# Search — two-stage retrieval logic
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _MILVUS_AVAILABLE, reason="pymilvus not installed")
class TestSearch:
    def _make_store_with_candidates(self, candidate_ids: list[str]) -> tuple[MilvusVectorStore, MagicMock]:
        store, mock_client = _make_mock_store()
        mock_client.list_collections.return_value = [
            "test_idx__pages", "test_idx__tokens"
        ]

        # _fetch_candidates: mock page search
        def page_search(**kwargs):
            rows = [
                {
                    "id": cid,
                    "entity": {
                        "payload_json": _serialize_payload(
                            {"doc_id": int(cid) // 10_000_000, "page_id": 1, "metadata": {}}
                        )
                    },
                }
                for cid in candidate_ids
            ]
            return [rows]

        # _late_interaction_rerank: mock token search, return first candidate as best
        def token_search(**kwargs):
            if not candidate_ids:
                return [[]]
            rows = [
                {
                    "entity": {"page_id": candidate_ids[0]},
                    "distance": 0.9,
                }
            ]
            return [rows]

        mock_client.search.side_effect = [page_search(**{}), token_search(**{})]
        return store, mock_client

    def test_search_returns_search_hits(self):
        candidate_ids = [str(make_point_id(i, 1)) for i in range(3)]
        store, mock_client = _make_mock_store()
        mock_client.list_collections.return_value = ["test_idx__pages", "test_idx__tokens"]

        # Mock page search
        page_rows = [
            {"id": cid, "entity": {"payload_json": _serialize_payload({"doc_id": i, "page_id": 1, "metadata": {}})}}
            for i, cid in enumerate(candidate_ids)
        ]
        # Mock token search (one per query token)
        token_rows = [
            {"entity": {"page_id": candidate_ids[0]}, "distance": 0.9}
        ]
        mock_client.search.side_effect = [[page_rows], [token_rows]]

        q = MultiVectorQuery(vectors=torch.rand(1, 8))
        results = store.search(q, k=2)
        assert len(results) > 0
        assert all(hasattr(r, "point_id") for r in results)

    def test_search_empty_candidates_returns_empty(self):
        store, mock_client = _make_mock_store()
        mock_client.list_collections.return_value = ["test_idx__pages", "test_idx__tokens"]
        mock_client.search.return_value = [[]]
        q = MultiVectorQuery(vectors=torch.rand(1, 8))
        results = store.search(q, k=5)
        assert results == []


# ---------------------------------------------------------------------------
# fetch_vector
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _MILVUS_AVAILABLE, reason="pymilvus not installed")
class TestFetchVector:
    def test_fetch_returns_tensor(self):
        store, mock_client = _make_mock_store()
        mock_client.list_collections.return_value = ["test_idx__tokens"]
        dim = 8
        n_tokens = 3
        token_rows = [{"token_vector": torch.rand(dim).tolist()} for _ in range(n_tokens)]
        mock_client.query.return_value = token_rows
        vec = store.fetch_vector(make_point_id(0, 1))
        assert vec is not None
        assert vec.shape == (n_tokens, dim)

    def test_fetch_returns_none_when_no_rows(self):
        store, mock_client = _make_mock_store()
        mock_client.list_collections.return_value = ["test_idx__tokens"]
        mock_client.query.return_value = []
        assert store.fetch_vector(make_point_id(0, 1)) is None


# ---------------------------------------------------------------------------
# Integration test: Milvus Lite (no external server)
# ---------------------------------------------------------------------------

@pytest.mark.slow
@pytest.mark.skipif(not _MILVUS_AVAILABLE, reason="pymilvus not installed")
class TestMilvusIntegration:
    """Full round-trip using Milvus Lite (file-based)."""

    def test_round_trip(self, tmp_path):
        dim = 8
        store = MilvusVectorStore()

        # open with create=True but no dim → collection not created yet
        store.open("test_rt", tmp_path, create=False)
        assert not store.collection_exists()

        # Create the collection explicitly
        store.create_collection(dim)
        assert store.collection_exists()

        pts = [
            StoredPoint(
                point_id=make_point_id(i, 1),
                vector=torch.rand(4, dim),
                payload={"doc_id": i, "page_id": 1, "chunk_id": None, "metadata": {}},
            )
            for i in range(5)
        ]
        store.upsert(pts)

        assert store.point_exists(pts[0].point_id)
        assert not store.point_exists(make_point_id(99, 99))

        q = MultiVectorQuery(vectors=torch.rand(2, dim))
        results = store.search(q, k=3)
        assert len(results) > 0

        vec = store.fetch_vector(pts[0].point_id)
        assert vec is not None
        assert vec.shape[1] == dim

        store.close()
