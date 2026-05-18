"""Tests for make_vector_store() factory."""
from __future__ import annotations

import pytest

from foretrieval.vector_store.factory import make_vector_store, BACKEND_REGISTRY
from foretrieval.vector_store.local import LocalVectorStore
from foretrieval.vector_store.qdrant import QdrantVectorStore
from foretrieval.vector_store.milvus import MilvusVectorStore


class TestFactory:
    def test_local_backend_returns_local_store(self):
        vs = make_vector_store("local")
        assert isinstance(vs, LocalVectorStore)

    def test_qdrant_backend_returns_qdrant_store(self):
        vs = make_vector_store("qdrant")
        assert isinstance(vs, QdrantVectorStore)

    def test_milvus_backend_returns_milvus_store(self):
        vs = make_vector_store("milvus")
        assert isinstance(vs, MilvusVectorStore)

    def test_unknown_backend_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown storage backend"):
            make_vector_store("nonexistent")

    def test_error_message_lists_supported_backends(self):
        with pytest.raises(ValueError, match="local"):
            make_vector_store("bad")

    def test_none_backend_defaults_to_local(self):
        vs = make_vector_store(None)
        assert isinstance(vs, LocalVectorStore)

    def test_whitespace_and_uppercase_handled(self):
        vs = make_vector_store("  LOCAL  ")
        assert isinstance(vs, LocalVectorStore)

    def test_milvus_candidate_limit_passed_through(self):
        vs = make_vector_store("milvus", {"candidate_limit": 128})
        assert isinstance(vs, MilvusVectorStore)
        assert vs._candidate_limit == 128

    def test_backend_registry_contains_all_three(self):
        assert "local" in BACKEND_REGISTRY
        assert "qdrant" in BACKEND_REGISTRY
        assert "milvus" in BACKEND_REGISTRY

    def test_each_call_returns_new_instance(self):
        a = make_vector_store("local")
        b = make_vector_store("local")
        assert a is not b
