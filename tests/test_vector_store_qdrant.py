"""Tests for QdrantVectorStore.

Unit tests mock the QdrantClient.
The slow integration test runs a real embedded Qdrant (no external server needed).
"""
from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import pytest
import torch

from foretrieval.vector_store.base import (
    MultiVectorQuery,
    StoredPoint,
    make_point_id,
)
from foretrieval.vector_store.qdrant import QdrantVectorStore, _QDRANT_AVAILABLE


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _point(doc_id: int, page_id: int, dim: int = 8) -> StoredPoint:
    pid = make_point_id(doc_id, page_id)
    return StoredPoint(
        point_id=pid,
        vector=torch.rand(4, dim),
        payload={"doc_id": doc_id, "page_id": page_id, "chunk_id": None, "metadata": {}},
    )


def _make_mock_store() -> tuple[QdrantVectorStore, MagicMock]:
    """Return a QdrantVectorStore with a mock client already injected."""
    store = QdrantVectorStore.__new__(QdrantVectorStore)
    mock_client = MagicMock()
    mock_client.collection_exists.return_value = False
    store._client = mock_client
    store._collection_name = "test_idx"
    store._index_root = Path("/tmp/foretrieval_test")
    return store, mock_client


# ---------------------------------------------------------------------------
# Optional-dependency guard
# ---------------------------------------------------------------------------

class TestOptionalDepGuard:
    def test_missing_qdrant_raises_on_open(self, tmp_path):
        with patch("foretrieval.vector_store.qdrant._QDRANT_AVAILABLE", False):
            store = QdrantVectorStore()
            with pytest.raises(RuntimeError, match="qdrant-client"):
                store.open("idx", tmp_path, create=True, dim=8)


# ---------------------------------------------------------------------------
# collection_exists / create_collection
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _QDRANT_AVAILABLE, reason="qdrant-client not installed")
class TestCollectionManagement:
    def test_collection_exists_false_when_not_created(self):
        store, mock_client = _make_mock_store()
        mock_client.collection_exists.return_value = False
        assert not store.collection_exists()

    def test_collection_exists_true_when_created(self):
        store, mock_client = _make_mock_store()
        mock_client.collection_exists.return_value = True
        assert store.collection_exists()

    def test_create_collection_calls_client(self):
        store, mock_client = _make_mock_store()
        mock_client.collection_exists.return_value = False
        store.create_collection(dim=128)
        mock_client.create_collection.assert_called_once()
        call_kwargs = mock_client.create_collection.call_args[1]
        assert call_kwargs["collection_name"] == "test_idx"

    def test_create_collection_multivector_config(self):
        """Collection must be created with MultiVectorConfig(MAX_SIM)."""
        from qdrant_client.models import MultiVectorComparator
        store, mock_client = _make_mock_store()
        mock_client.collection_exists.return_value = False
        store.create_collection(dim=128)
        call_kwargs = mock_client.create_collection.call_args[1]
        vc = call_kwargs["vectors_config"]
        assert vc.multivector_config.comparator == MultiVectorComparator.MAX_SIM

    def test_create_collection_noop_if_already_exists(self):
        store, mock_client = _make_mock_store()
        mock_client.collection_exists.return_value = True
        store.create_collection(dim=128)
        mock_client.create_collection.assert_not_called()


# ---------------------------------------------------------------------------
# Upsert
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _QDRANT_AVAILABLE, reason="qdrant-client not installed")
class TestUpsert:
    def test_upsert_calls_client_upsert(self):
        store, mock_client = _make_mock_store()
        sp = _point(0, 1)
        store.upsert([sp])
        mock_client.upsert.assert_called_once()
        call_kwargs = mock_client.upsert.call_args[1]
        assert call_kwargs["collection_name"] == "test_idx"
        assert len(call_kwargs["points"]) == 1

    def test_upsert_payload_contains_doc_page(self):
        store, mock_client = _make_mock_store()
        sp = _point(5, 3)
        store.upsert([sp])
        point = mock_client.upsert.call_args[1]["points"][0]
        assert point.payload["doc_id"] == 5
        assert point.payload["page_id"] == 3

    def test_upsert_vector_is_list_of_floats(self):
        store, mock_client = _make_mock_store()
        sp = _point(0, 1)
        store.upsert([sp])
        point = mock_client.upsert.call_args[1]["points"][0]
        # Multivector: list of lists
        assert isinstance(point.vector, list)
        assert all(isinstance(row, list) for row in point.vector)

    def test_point_exists_returns_true(self):
        store, mock_client = _make_mock_store()
        mock_client.collection_exists.return_value = True
        mock_client.retrieve.return_value = [MagicMock()]
        assert store.point_exists(make_point_id(0, 1))

    def test_point_exists_returns_false(self):
        store, mock_client = _make_mock_store()
        mock_client.collection_exists.return_value = True
        mock_client.retrieve.return_value = []
        assert not store.point_exists(make_point_id(0, 1))


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _QDRANT_AVAILABLE, reason="qdrant-client not installed")
class TestSearch:
    def _make_hit(self, pid, score, doc_id, page_id):
        hit = MagicMock()
        hit.id = pid
        hit.score = score
        hit.payload = {"doc_id": doc_id, "page_id": page_id, "chunk_id": None, "metadata": {}}
        return hit

    def test_search_returns_search_hits(self):
        store, mock_client = _make_mock_store()
        hits = [self._make_hit(100, 0.9, 0, 1), self._make_hit(200, 0.5, 1, 1)]
        response = MagicMock()
        response.points = hits
        mock_client.query_points.return_value = response

        q = MultiVectorQuery(vectors=torch.rand(2, 8))
        results = store.search(q, k=2)
        assert len(results) == 2
        assert results[0].point_id == 100
        assert results[0].score == pytest.approx(0.9)

    def test_search_passes_k_to_client(self):
        store, mock_client = _make_mock_store()
        response = MagicMock()
        response.points = []
        mock_client.query_points.return_value = response
        q = MultiVectorQuery(vectors=torch.rand(2, 8))
        store.search(q, k=7)
        call_kwargs = mock_client.query_points.call_args[1]
        assert call_kwargs["limit"] == 7

    def test_search_with_metadata_filter_passes_filter(self):
        from qdrant_client.models import Filter
        store, mock_client = _make_mock_store()
        response = MagicMock()
        response.points = []
        mock_client.query_points.return_value = response
        q = MultiVectorQuery(vectors=torch.rand(2, 8), filter_metadata={"category": "A"})
        store.search(q, k=5)
        call_kwargs = mock_client.query_points.call_args[1]
        assert call_kwargs["query_filter"] is not None
        assert isinstance(call_kwargs["query_filter"], Filter)

    def test_search_no_filter_passes_none(self):
        store, mock_client = _make_mock_store()
        response = MagicMock()
        response.points = []
        mock_client.query_points.return_value = response
        q = MultiVectorQuery(vectors=torch.rand(2, 8))
        store.search(q, k=5)
        call_kwargs = mock_client.query_points.call_args[1]
        assert call_kwargs["query_filter"] is None


# ---------------------------------------------------------------------------
# fetch_vector
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _QDRANT_AVAILABLE, reason="qdrant-client not installed")
class TestFetchVector:
    def test_fetch_returns_tensor(self):
        store, mock_client = _make_mock_store()
        vec_data = [[0.1, 0.2, 0.3, 0.4]] * 4
        retrieved = MagicMock()
        retrieved.vector = vec_data
        mock_client.retrieve.return_value = [retrieved]
        result = store.fetch_vector(make_point_id(0, 1))
        assert result is not None
        assert isinstance(result, torch.Tensor)

    def test_fetch_returns_none_when_not_found(self):
        store, mock_client = _make_mock_store()
        mock_client.retrieve.return_value = []
        assert store.fetch_vector(make_point_id(99, 99)) is None


# ---------------------------------------------------------------------------
# Integration test: real embedded Qdrant
# ---------------------------------------------------------------------------

@pytest.mark.slow
@pytest.mark.skipif(not _QDRANT_AVAILABLE, reason="qdrant-client not installed")
class TestQdrantIntegration:
    """Full round-trip: open → upsert → search → fetch_vector → close."""

    def test_round_trip(self, tmp_path):
        dim = 8
        store = QdrantVectorStore()

        # Open with create=False first to test empty state
        store.open("test_rt", tmp_path, create=False)
        assert not store.collection_exists()

        # Create collection explicitly
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
        assert len(results) == 3

        vec = store.fetch_vector(pts[0].point_id)
        assert vec is not None
        assert vec.shape[1] == dim

        store.close()
