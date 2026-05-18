"""Tests for make_vector_store() factory."""
from __future__ import annotations

from unittest.mock import MagicMock, patch
import pytest

from foretrieval.vector_store.factory import make_vector_store, BACKEND_REGISTRY
from foretrieval.vector_store.local import LocalVectorStore
from foretrieval.vector_store.qdrant import QdrantVectorStore
from foretrieval.vector_store.milvus import MilvusVectorStore
from foretrieval.vector_store.remote import RemoteVectorStore


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

    def test_backend_registry_contains_all_four(self):
        assert "local" in BACKEND_REGISTRY
        assert "qdrant" in BACKEND_REGISTRY
        assert "milvus" in BACKEND_REGISTRY
        assert "remote" in BACKEND_REGISTRY

    def test_each_call_returns_new_instance(self):
        a = make_vector_store("local")
        b = make_vector_store("local")
        assert a is not b


class TestRemoteFactory:
    """Test make_vector_store("remote", ...) factory path."""

    def _make_remote(self, extra=None, auto_deploy=False):
        cfg = {
            "url": "http://localhost:18000",
            "backend": "qdrant",
            "auto_deploy": auto_deploy,
        }
        if extra:
            cfg.update(extra)
        # Patch VectorDBServerClient inside the module where it is imported
        with patch(
            "foretrieval.vector_db_server.client.VectorDBServerClient"
        ):
            with patch(
                "foretrieval.vector_store.factory._make_remote_vector_store",
            ) as mock_factory:
                mock_client_inst = MagicMock()
                remote_store = RemoteVectorStore(mock_client_inst, backend=cfg.get("backend", "qdrant"))
                mock_factory.return_value = remote_store
                vs = make_vector_store("remote", cfg)
        return vs, mock_client_inst

    def test_remote_backend_returns_remote_store(self):
        vs, _ = self._make_remote()
        assert isinstance(vs, RemoteVectorStore)

    def test_remote_store_has_correct_server_backend(self):
        vs, _ = self._make_remote()
        assert vs.server_backend == "qdrant"

    def test_remote_store_milvus_backend(self):
        vs, _ = self._make_remote(extra={"backend": "milvus"})
        assert vs.server_backend == "milvus"

    def test_remote_missing_url_raises(self):
        with pytest.raises(ValueError, match="url"):
            make_vector_store("remote", {"backend": "qdrant"})

    def test_auto_deploy_calls_manager_and_health_check(self):
        with (
            patch("foretrieval.vector_db_server.client.VectorDBServerClient") as MockClient,
            patch("foretrieval.vector_db_server.manager.VectorDBServerManager") as MockMgr,
            patch("foretrieval.vector_store.factory._wait_for_health") as mock_wait,
        ):
            mock_client_inst = MagicMock()
            MockClient.return_value = mock_client_inst
            mock_mgr_inst = MagicMock()
            MockMgr.return_value = mock_mgr_inst

            vs = make_vector_store(
                "remote",
                {
                    "url": "http://gpu-server:18000",
                    "backend": "qdrant",
                    "auto_deploy": True,
                    "ssh_host": "gpu-server",
                },
            )
        mock_mgr_inst.ensure_deployed.assert_called_once()
        mock_wait.assert_called_once()
        assert isinstance(vs, RemoteVectorStore)

    def test_candidate_limit_forwarded_as_server_storage_config(self):
        mock_client_inst = MagicMock()
        remote_store = RemoteVectorStore(
            mock_client_inst, backend="milvus",
            storage_config={"candidate_limit": 128}
        )
        with patch(
            "foretrieval.vector_store.factory._make_remote_vector_store",
            return_value=remote_store,
        ):
            vs = make_vector_store(
                "remote",
                {
                    "url": "http://localhost:18000",
                    "backend": "milvus",
                    "candidate_limit": 128,
                },
            )
        assert vs._storage_config == {"candidate_limit": 128}

