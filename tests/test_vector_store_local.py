"""Tests for LocalVectorStore.

All tests run without GPU — processor.score() is mocked.
"""
from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from foretrieval.vector_store.base import (
    MultiVectorQuery,
    StoredPoint,
    make_point_id,
)
from foretrieval.vector_store.local import LocalVectorStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_store(processor=None) -> LocalVectorStore:
    store = LocalVectorStore()
    if processor is not None:
        store.set_processor(processor)
    return store


def _dummy_embedding(n_tokens: int = 4, dim: int = 8) -> torch.Tensor:
    return torch.rand(n_tokens, dim)


def _mock_processor(scores: list[float]):
    """Return a processor mock whose score() returns the given scores as a 2-D array."""
    proc = MagicMock()
    proc.score.return_value = torch.tensor([scores])
    return proc


def _point(doc_id: int, page_id: int, vec: torch.Tensor | None = None) -> StoredPoint:
    if vec is None:
        vec = _dummy_embedding()
    pid = make_point_id(doc_id, page_id)
    return StoredPoint(
        point_id=pid,
        vector=vec,
        payload={"doc_id": doc_id, "page_id": page_id, "chunk_id": None, "metadata": {}},
    )


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------

class TestLifecycle:
    def test_open_does_not_raise(self, tmp_path):
        store = LocalVectorStore()
        store.open("idx", tmp_path, create=True, dim=8)

    def test_collection_not_exists_before_upsert(self, tmp_path):
        store = LocalVectorStore()
        store.open("idx", tmp_path, create=True, dim=8)
        assert not store.collection_exists()

    def test_collection_exists_after_export(self, tmp_path):
        store = LocalVectorStore()
        store.open("idx", tmp_path, create=True, dim=8)
        store.upsert([_point(0, 1)])
        index_dir = tmp_path / "idx"
        index_dir.mkdir(parents=True, exist_ok=True)
        store.export_sidecar(index_dir)
        assert store.collection_exists()


# ---------------------------------------------------------------------------
# Upsert and point_exists
# ---------------------------------------------------------------------------

class TestUpsert:
    def test_upsert_single_point(self, tmp_path):
        store = LocalVectorStore()
        store.open("idx", tmp_path, create=True)
        sp = _point(0, 1)
        store.upsert([sp])
        assert store.point_exists(sp.point_id)

    def test_upsert_multiple_points(self, tmp_path):
        store = LocalVectorStore()
        store.open("idx", tmp_path, create=True)
        pts = [_point(i, 1) for i in range(5)]
        store.upsert(pts)
        for p in pts:
            assert store.point_exists(p.point_id)

    def test_point_not_exists_before_upsert(self, tmp_path):
        store = LocalVectorStore()
        store.open("idx", tmp_path, create=True)
        assert not store.point_exists(make_point_id(0, 1))

    def test_upsert_idempotent_last_write_wins(self, tmp_path):
        store = LocalVectorStore()
        store.open("idx", tmp_path, create=True)
        pid = make_point_id(0, 1)
        v1 = torch.zeros(4, 8)
        v2 = torch.ones(4, 8)
        sp1 = StoredPoint(point_id=pid, vector=v1, payload={"doc_id": 0, "page_id": 1, "chunk_id": None, "metadata": {}})
        sp2 = StoredPoint(point_id=pid, vector=v2, payload={"doc_id": 0, "page_id": 1, "chunk_id": None, "metadata": {}})
        store.upsert([sp1])
        store.upsert([sp2])
        fetched = store.fetch_vector(pid)
        assert fetched is not None
        assert torch.allclose(fetched, v2)


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

class TestSearch:
    def test_search_returns_k_results(self, tmp_path):
        scores = [0.5, 0.8, 0.3]
        proc = _mock_processor(scores)
        store = _make_store(proc)
        store.open("idx", tmp_path, create=True)
        for i in range(3):
            store.upsert([_point(i, 1)])
        q = MultiVectorQuery(vectors=_dummy_embedding(2, 8))
        results = store.search(q, k=2)
        assert len(results) == 2

    def test_search_sorted_descending_score(self, tmp_path):
        scores = [0.3, 0.9, 0.1]
        proc = _mock_processor(scores)
        store = _make_store(proc)
        store.open("idx", tmp_path, create=True)
        for i in range(3):
            store.upsert([_point(i, 1)])
        q = MultiVectorQuery(vectors=_dummy_embedding(2, 8))
        results = store.search(q, k=3)
        assert results[0].score >= results[1].score >= results[2].score

    def test_search_returns_correct_point_ids(self, tmp_path):
        scores = [0.1, 0.9, 0.5]
        proc = _mock_processor(scores)
        store = _make_store(proc)
        store.open("idx", tmp_path, create=True)
        pts = [_point(i, 1) for i in range(3)]
        for p in pts:
            store.upsert([p])
        q = MultiVectorQuery(vectors=_dummy_embedding(2, 8))
        results = store.search(q, k=1)
        # Best score is index 1 (score 0.9 → doc_id=1)
        assert results[0].payload["doc_id"] == 1

    def test_search_raises_without_processor(self, tmp_path):
        store = LocalVectorStore()
        store.open("idx", tmp_path, create=True)
        store.upsert([_point(0, 1)])
        q = MultiVectorQuery(vectors=_dummy_embedding(2, 8))
        with pytest.raises(RuntimeError, match="set_processor"):
            store.search(q, k=1)

    def test_search_empty_store_returns_empty(self, tmp_path):
        proc = MagicMock()
        store = _make_store(proc)
        store.open("idx", tmp_path, create=True)
        q = MultiVectorQuery(vectors=_dummy_embedding(2, 8))
        assert store.search(q, k=5) == []

    def test_search_with_metadata_filter(self, tmp_path):
        """Filter should restrict search to matching docs only."""
        # 3 docs: doc 0 has language=en, doc 1 has language=fr, doc 2 has language=en
        # We filter on language=fr → only doc 1 should be in the result pool.
        # The mock processor returns a single score for one embedding (the filtered one).
        store = LocalVectorStore()
        store.open("idx", tmp_path, create=True)
        store.set_doc_id_to_metadata({
            0: {"language": "en"},
            1: {"language": "fr"},
            2: {"language": "en"},
        })
        for i in range(3):
            store.upsert([_point(i, 1)])

        # After metadata filter only doc 1 is left → processor.score gets 1 embedding
        proc_filtered = MagicMock()
        proc_filtered.score.return_value = torch.tensor([[0.7]])
        store.set_processor(proc_filtered)

        q = MultiVectorQuery(
            vectors=_dummy_embedding(2, 8),
            filter_metadata={"language": "fr"},
        )
        results = store.search(q, k=5)
        assert len(results) == 1
        assert results[0].payload["doc_id"] == 1


# ---------------------------------------------------------------------------
# fetch_vector
# ---------------------------------------------------------------------------

class TestFetchVector:
    def test_fetch_returns_tensor(self, tmp_path):
        store = LocalVectorStore()
        store.open("idx", tmp_path, create=True)
        vec = _dummy_embedding(4, 8)
        sp = _point(0, 1, vec)
        store.upsert([sp])
        fetched = store.fetch_vector(sp.point_id)
        assert fetched is not None
        assert fetched.shape == (4, 8)
        assert torch.allclose(fetched, vec)

    def test_fetch_unknown_id_returns_none(self, tmp_path):
        store = LocalVectorStore()
        store.open("idx", tmp_path, create=True)
        assert store.fetch_vector(make_point_id(99, 99)) is None


# ---------------------------------------------------------------------------
# Persistence round-trip
# ---------------------------------------------------------------------------

class TestPersistence:
    def test_export_load_round_trip(self, tmp_path):
        store1 = LocalVectorStore()
        store1.open("idx", tmp_path, create=True)
        pts = [_point(i, 1) for i in range(3)]
        for p in pts:
            store1.upsert([p])
        index_path = tmp_path / "idx"
        index_path.mkdir(parents=True, exist_ok=True)
        store1.export_sidecar(index_path)

        # Load into a fresh store
        proc = MagicMock()
        scores = [0.5, 0.8, 0.3]
        proc.score.return_value = torch.tensor([scores])
        store2 = LocalVectorStore()
        store2.open("idx", tmp_path, create=False)
        store2.load_sidecar(index_path)
        store2.set_processor(proc)

        # All points should be present
        for p in pts:
            assert store2.point_exists(p.point_id)

        # Search should work
        q = MultiVectorQuery(vectors=_dummy_embedding(2, 8))
        results = store2.search(q, k=2)
        assert len(results) == 2
