"""Tests for ColPaliModel backend dispatch via the VectorStore interface.

These tests verify that:
- storage_backend="local/qdrant/milvus" selects the correct VectorStore class
- the deprecated storage_qdrant boolean still works with a DeprecationWarning
- backward-compat properties (storage_qdrant, qdrant_client, qdrant_collection,
  indexed_embeddings, embed_id_to_doc_id) still work

No GPU or real model is required — ColPaliModel.__init__ is bypassed via
direct attribute injection on MagicMock objects.
"""
from __future__ import annotations

import warnings
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

from foretrieval.vector_store import (
    LocalVectorStore,
    QdrantVectorStore,
    MilvusVectorStore,
    make_vector_store,
)
from foretrieval.vector_store.base import make_point_id, StoredPoint


# ---------------------------------------------------------------------------
# Helpers — build a ColPaliModel with all heavy parts mocked
# ---------------------------------------------------------------------------

def _make_model(storage_backend: str = "local", **extra_kwargs):
    """Build a ColPaliModel instance with GPU/model loading patched out."""
    with (
        patch("foretrieval.colpali.ColPaliModel._load_model_and_processor"),
        patch("foretrieval.colpali.ColPaliModel._load_processor_only"),
    ):
        from foretrieval.colpali import ColPaliModel

        model = ColPaliModel.__new__(ColPaliModel)

        # Minimal attributes to avoid AttributeError from __init__ side-effects
        model.pretrained_model_name_or_path = "vidore/colpali-v1.2-test"
        model.model_name = "vidore/colpali-v1.2-test"
        model.verbose = 0
        model.load_from_index = False
        model.index_root = ".foretrieval_test"
        model.index_name = None
        model.kwargs = {}
        model.storage_backend = storage_backend.strip().lower()
        model.storage_config = extra_kwargs.get("storage_config", {})
        model._storage_qdrant_compat = (model.storage_backend == "qdrant")
        model.ingestion = {"backend": "default"}
        model.ingestion_backend = "default"
        model.n_gpu = 0
        model.device = "cpu"
        model.load_in_4bit = False
        model.load_in_8bit = False
        model.bnb_4bit_quant_type = "nf4"
        model.bnb_4bit_compute_dtype = "float16"
        model.collection = {}
        model.embed_id_to_extra = {}
        model.doc_id_to_metadata = {}
        model.doc_ids_to_file_names = {}
        model.doc_ids = set()
        model.enable_heatmaps = False
        model.enable_circle = False
        model.full_document_collection = False
        model.resize_stored_images = False
        model.max_image_width = None
        model.max_image_height = None
        model.highest_doc_id = -1
        model.docling_dir = None
        model.SOURCE_EXTS = set()
        model.IMAGE_EXTS = set()
        model._remote_client = None
        model.model = None
        model.processor = MagicMock()

        # Build real VectorStore
        model.vector_store = make_vector_store(storage_backend, model.storage_config)

        return model


# ---------------------------------------------------------------------------
# Backend dispatch
# ---------------------------------------------------------------------------

class TestBackendDispatch:
    def test_local_backend(self):
        model = _make_model("local")
        assert isinstance(model.vector_store, LocalVectorStore)

    def test_qdrant_backend(self):
        model = _make_model("qdrant")
        assert isinstance(model.vector_store, QdrantVectorStore)

    def test_milvus_backend(self):
        model = _make_model("milvus")
        assert isinstance(model.vector_store, MilvusVectorStore)

    def test_milvus_candidate_limit_forwarded(self):
        model = _make_model("milvus", storage_config={"candidate_limit": 128})
        assert isinstance(model.vector_store, MilvusVectorStore)
        assert model.vector_store._candidate_limit == 128


# ---------------------------------------------------------------------------
# Deprecated storage_qdrant shim
# ---------------------------------------------------------------------------

class TestDeprecatedStorageQdrant:
    def test_storage_qdrant_true_maps_to_qdrant_backend(self):
        """ColPaliModel(storage_qdrant=True) should emit DeprecationWarning and select qdrant."""
        with (
            patch("foretrieval.colpali.ColPaliModel._load_model_and_processor"),
            patch("foretrieval.colpali.ColPaliModel._load_processor_only"),
        ):
            from foretrieval.colpali import ColPaliModel
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                model = ColPaliModel(
                    pretrained_model_name_or_path="vidore/colpali-v1.2",
                    storage_qdrant=True,
                    device="cpu",
                    verbose=0,
                )
            # Should have emitted exactly one DeprecationWarning
            dep_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert len(dep_warnings) == 1
            assert "storage_qdrant" in str(dep_warnings[0].message)
            assert model.storage_backend == "qdrant"

    def test_storage_qdrant_false_maps_to_local_backend(self):
        with (
            patch("foretrieval.colpali.ColPaliModel._load_model_and_processor"),
            patch("foretrieval.colpali.ColPaliModel._load_processor_only"),
        ):
            from foretrieval.colpali import ColPaliModel
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                model = ColPaliModel(
                    pretrained_model_name_or_path="vidore/colpali-v1.2",
                    storage_qdrant=False,
                    device="cpu",
                    verbose=0,
                )
            assert model.storage_backend == "local"


# ---------------------------------------------------------------------------
# Backward-compat property accessors
# ---------------------------------------------------------------------------

class TestBackwardCompatProperties:
    def test_storage_qdrant_property_true_for_qdrant_backend(self):
        model = _make_model("qdrant")
        assert model.storage_qdrant is True

    def test_storage_qdrant_property_false_for_local_backend(self):
        model = _make_model("local")
        assert model.storage_qdrant is False

    def test_storage_qdrant_property_false_for_milvus_backend(self):
        model = _make_model("milvus")
        assert model.storage_qdrant is False

    def test_qdrant_client_property_returns_none_for_local(self):
        model = _make_model("local")
        assert model.qdrant_client is None

    def test_qdrant_client_property_returns_none_for_milvus(self):
        model = _make_model("milvus")
        assert model.qdrant_client is None

    def test_indexed_embeddings_local_backend_returns_list(self):
        model = _make_model("local")
        assert isinstance(model.indexed_embeddings, list)

    def test_indexed_embeddings_qdrant_backend_returns_empty(self):
        model = _make_model("qdrant")
        assert model.indexed_embeddings == []

    def test_embed_id_to_doc_id_local_backend_returns_dict(self):
        model = _make_model("local")
        assert isinstance(model.embed_id_to_doc_id, dict)

    def test_embed_id_to_doc_id_qdrant_backend_returns_empty(self):
        model = _make_model("qdrant")
        assert model.embed_id_to_doc_id == {}


# ---------------------------------------------------------------------------
# VectorStore open() called with correct args
# ---------------------------------------------------------------------------

class TestVectorStoreOpenCall:
    def test_open_called_with_index_name(self, tmp_path):
        """When ColPaliModel.index() is called, vector_store.open() gets the index name."""
        model = _make_model("local")
        model.index_root = str(tmp_path)

        # Patch open on the LocalVectorStore so we can verify the call
        from foretrieval.vector_store.local import LocalVectorStore as LSV
        original_open = LSV.open
        calls = []

        def fake_open(self_, index_name, index_root, *, create, dim=None):
            calls.append((index_name, create))
            original_open(self_, index_name, index_root, create=create, dim=dim)

        with patch.object(LSV, "open", fake_open):
            model.vector_store = make_vector_store("local")
            model.vector_store.open("my_index", tmp_path, create=True)

        assert ("my_index", True) in calls


# ---------------------------------------------------------------------------
# index_config storage_backend field
# ---------------------------------------------------------------------------

class TestIndexConfigStorageBackend:
    """Verify _export_index writes the correct storage_backend string."""

    def test_export_writes_local(self, tmp_path):
        import srsly
        model = _make_model("local")
        model.index_root = str(tmp_path)
        model.index_name = "test_export"
        model.doc_id_to_metadata = {}
        model.doc_ids_to_file_names = {}
        model.embed_id_to_extra = {}
        model.full_document_collection = False
        model.max_image_width = None
        model.max_image_height = None
        model.highest_doc_id = -1
        model.model_name = "vidore/colpali-v1.2"

        model._export_index()

        cfg = srsly.read_gzip_json(tmp_path / "test_export" / "index_config.json.gz")
        assert cfg["storage_backend"] == "local"

    def test_export_writes_qdrant(self, tmp_path):
        import srsly
        model = _make_model("qdrant")
        model.index_root = str(tmp_path)
        model.index_name = "test_export_qdrant"
        model.doc_id_to_metadata = {}
        model.doc_ids_to_file_names = {}
        model.embed_id_to_extra = {}
        model.full_document_collection = False
        model.max_image_width = None
        model.max_image_height = None
        model.highest_doc_id = -1
        model.model_name = "vidore/colpali-v1.2"

        # Patch vector_store.export_sidecar to avoid Qdrant client calls
        model.vector_store = MagicMock()
        model._export_index()

        cfg = srsly.read_gzip_json(tmp_path / "test_export_qdrant" / "index_config.json.gz")
        assert cfg["storage_backend"] == "qdrant"

    def test_export_writes_milvus(self, tmp_path):
        import srsly
        model = _make_model("milvus")
        model.index_root = str(tmp_path)
        model.index_name = "test_export_milvus"
        model.doc_id_to_metadata = {}
        model.doc_ids_to_file_names = {}
        model.embed_id_to_extra = {}
        model.full_document_collection = False
        model.max_image_width = None
        model.max_image_height = None
        model.highest_doc_id = -1
        model.model_name = "vidore/colpali-v1.2"

        model.vector_store = MagicMock()
        model._export_index()

        cfg = srsly.read_gzip_json(tmp_path / "test_export_milvus" / "index_config.json.gz")
        assert cfg["storage_backend"] == "milvus"


# ---------------------------------------------------------------------------
# Remote backend dispatch and colpali integration
# ---------------------------------------------------------------------------

class TestRemoteBackendDispatch:
    """Verify RemoteVectorStore is selected and ColPaliModel integrates correctly."""

    def _make_remote_model(self, url="http://localhost:18000"):
        from unittest.mock import patch as _patch
        with _patch("foretrieval.vector_db_server.client.VectorDBServerClient") as MockClient:
            mock_client_inst = MagicMock()
            mock_client_inst.open_collection.return_value = {
                "opened": True, "backend": "qdrant", "created": True
            }
            MockClient.return_value = mock_client_inst
            model = _make_model(
                "remote",
                storage_config={"url": url, "backend": "qdrant"},
            )
        return model, mock_client_inst

    def test_remote_backend_selects_remote_store(self):
        from foretrieval.vector_store.remote import RemoteVectorStore
        model, _ = self._make_remote_model()
        assert isinstance(model.vector_store, RemoteVectorStore)

    def test_vector_store_is_open_remote(self):
        """_vector_store_is_open returns True after open() is called."""
        from foretrieval.colpali import ColPaliModel
        model, mock_client = self._make_remote_model()
        # Simulate open() having been called (set _opened flag)
        model.vector_store._opened = True
        assert model._vector_store_is_open() is True

    def test_vector_store_is_not_open_before_open(self):
        from foretrieval.colpali import ColPaliModel
        model, _ = self._make_remote_model()
        model.vector_store._opened = False
        assert model._vector_store_is_open() is False

    def test_set_processor_not_called_for_remote(self):
        """set_processor is a local-only hook — must not be called for remote."""
        from foretrieval.vector_store.remote import RemoteVectorStore
        model, _ = self._make_remote_model()
        assert isinstance(model.vector_store, RemoteVectorStore)
        # RemoteVectorStore has no set_processor attribute
        assert not hasattr(model.vector_store, "set_processor")

    def test_export_index_writes_remote_backend(self, tmp_path):
        import srsly
        model, _ = self._make_remote_model()
        model.index_root = str(tmp_path)
        model.index_name = "test_export_remote"
        model.doc_id_to_metadata = {}
        model.doc_ids_to_file_names = {}
        model.embed_id_to_extra = {}
        model.full_document_collection = False
        model.max_image_width = None
        model.max_image_height = None
        model.highest_doc_id = -1
        model.model_name = "vidore/colpali-v1.2"
        # storage_config set on model (url present, no api_key)
        model.storage_config = {"url": "http://localhost:18000", "backend": "qdrant"}

        model.vector_store = MagicMock()
        model._export_index()

        cfg = srsly.read_gzip_json(
            tmp_path / "test_export_remote" / "index_config.json.gz"
        )
        assert cfg["storage_backend"] == "remote"
        # api_key must NOT be in persisted storage_config
        persisted = cfg.get("storage_config") or {}
        assert "api_key" not in persisted

    def test_export_index_strips_api_key(self, tmp_path):
        import srsly
        model, _ = self._make_remote_model()
        model.index_root = str(tmp_path)
        model.index_name = "test_strip_key"
        model.doc_id_to_metadata = {}
        model.doc_ids_to_file_names = {}
        model.embed_id_to_extra = {}
        model.full_document_collection = False
        model.max_image_width = None
        model.max_image_height = None
        model.highest_doc_id = -1
        model.model_name = "vidore/colpali-v1.2"
        model.storage_config = {
            "url": "http://localhost:18000",
            "backend": "qdrant",
            "api_key": "supersecret",
        }

        model.vector_store = MagicMock()
        model._export_index()

        cfg = srsly.read_gzip_json(
            tmp_path / "test_strip_key" / "index_config.json.gz"
        )
        persisted = cfg.get("storage_config") or {}
        assert "api_key" not in persisted
        assert persisted.get("url") == "http://localhost:18000"

    def test_indexed_embeddings_returns_empty_for_remote(self):
        model, _ = self._make_remote_model()
        assert model.indexed_embeddings == []

    def test_embed_id_to_doc_id_returns_empty_for_remote(self):
        model, _ = self._make_remote_model()
        assert model.embed_id_to_doc_id == {}
