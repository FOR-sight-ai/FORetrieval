"""Tests for RemoteVectorStore — all HTTP calls mocked via a MagicMock client."""

from pathlib import Path
from unittest.mock import MagicMock
import pytest
import torch

from foretrieval.vector_store.remote import RemoteVectorStore
from foretrieval.vector_store.base import MultiVectorQuery, SearchHit, StoredPoint, make_point_id


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_client() -> MagicMock:
    client = MagicMock()
    client.open_collection.return_value = {"opened": True, "backend": "qdrant", "created": True}
    client.collection_exists.return_value = True
    client.point_exists.return_value = False
    client.search.return_value = []
    client.fetch_vector.return_value = None
    return client


def _make_store(backend="qdrant") -> tuple[RemoteVectorStore, MagicMock]:
    client = _make_mock_client()
    store = RemoteVectorStore(client, backend=backend)
    return store, client


def _make_point(doc_id=1, page_id=0, dim=8) -> StoredPoint:
    return StoredPoint(
        point_id=make_point_id(doc_id, page_id),
        vector=torch.randn(3, dim),
        payload={"doc_id": doc_id, "page_id": page_id},
    )


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------

class TestLifecycle:
    def test_open_calls_client(self):
        store, client = _make_store()
        store.open("idx", Path("."), create=True, dim=128)
        client.open_collection.assert_called_once_with(
            "idx", "qdrant", create=True, dim=128, storage_config=None
        )

    def test_is_opened_after_open(self):
        store, _ = _make_store()
        assert not store.is_opened
        store.open("idx", Path("."), create=True, dim=128)
        assert store.is_opened

    def test_close_marks_not_opened(self):
        store, _ = _make_store()
        store.open("idx", Path("."), create=True, dim=128)
        store.close()
        assert not store.is_opened

    def test_backend_name_class_attr(self):
        assert RemoteVectorStore.backend_name == "remote"


# ---------------------------------------------------------------------------
# collection_exists
# ---------------------------------------------------------------------------

class TestCollectionExists:
    def test_returns_false_before_open(self):
        store, client = _make_store()
        assert store.collection_exists() is False

    def test_delegates_to_client(self):
        store, client = _make_store()
        store.open("idx", Path("."), create=True)
        client.collection_exists.return_value = True
        assert store.collection_exists() is True
        client.collection_exists.assert_called_with("idx")


# ---------------------------------------------------------------------------
# upsert / point_exists
# ---------------------------------------------------------------------------

class TestWrite:
    def test_upsert_delegates_to_client(self):
        store, client = _make_store()
        store.open("idx", Path("."), create=True)
        pts = [_make_point()]
        store.upsert(pts)
        client.upsert.assert_called_once_with("idx", pts)

    def test_point_exists_delegates(self):
        store, client = _make_store()
        store.open("idx", Path("."), create=True)
        client.point_exists.return_value = True
        pid = make_point_id(1, 0)
        assert store.point_exists(pid) is True
        client.point_exists.assert_called_with("idx", pid)

    def test_point_exists_returns_false_before_open(self):
        store, _ = _make_store()
        assert store.point_exists(12345) is False

    def test_upsert_raises_without_open(self):
        store, _ = _make_store()
        with pytest.raises(RuntimeError, match="open"):
            store.upsert([_make_point()])


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------

class TestSearch:
    def test_search_delegates_to_client(self):
        store, client = _make_store()
        store.open("idx", Path("."), create=True)
        hit = SearchHit(point_id=10000, score=9.5, payload={"doc_id": 1, "page_id": 0})
        client.search.return_value = [hit]

        query = MultiVectorQuery(vectors=torch.randn(4, 8))
        results = store.search(query, k=3)

        client.search.assert_called_once_with("idx", query, 3)
        assert len(results) == 1
        assert results[0].score == pytest.approx(9.5)

    def test_search_raises_without_open(self):
        store, _ = _make_store()
        with pytest.raises(RuntimeError, match="open"):
            store.search(MultiVectorQuery(vectors=torch.randn(2, 8)), k=5)


# ---------------------------------------------------------------------------
# fetch_vector
# ---------------------------------------------------------------------------

class TestFetchVector:
    def test_fetch_returns_tensor(self):
        store, client = _make_store()
        store.open("idx", Path("."), create=True)
        t = torch.randn(3, 8)
        client.fetch_vector.return_value = t
        result = store.fetch_vector(make_point_id(1, 0))
        assert result is t

    def test_fetch_returns_none_on_miss(self):
        store, client = _make_store()
        store.open("idx", Path("."), create=True)
        client.fetch_vector.return_value = None
        assert store.fetch_vector(99999) is None

    def test_fetch_returns_none_before_open(self):
        store, _ = _make_store()
        assert store.fetch_vector(12345) is None


# ---------------------------------------------------------------------------
# Persistence no-ops
# ---------------------------------------------------------------------------

class TestPersistenceNoOps:
    def test_export_sidecar_noop(self, tmp_path):
        store, _ = _make_store()
        store.open("idx", Path("."), create=True)
        # Should not raise or create any files
        store.export_sidecar(tmp_path)
        assert list(tmp_path.iterdir()) == []

    def test_load_sidecar_noop(self, tmp_path):
        store, _ = _make_store()
        store.open("idx", Path("."), create=True)
        store.load_sidecar(tmp_path)  # no error
