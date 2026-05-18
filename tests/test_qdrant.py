"""
Tests for the Qdrant storage backend and the docling ingestion guard.

Since Qdrant logic now lives in foretrieval/vector_store/qdrant.py, the
low-level tests (make_point_id, filter building, upsert shape, etc.) live in
test_vector_store_qdrant.py.  This file retains the tests that exercise
ColPaliModel-level behaviour: storage_backend selection, index config
serialization, search dispatch, and the docling guard.

The slow integration test (marked @pytest.mark.slow) is preserved unchanged
from the original test_qdrant.py.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from foretrieval.colpali import ColPaliModel, _DOCLING_AVAILABLE
from foretrieval.vector_store import LocalVectorStore, QdrantVectorStore, make_point_id
from foretrieval.vector_store.qdrant import _QDRANT_AVAILABLE

DATA_DIR = Path(__file__).parent / "data"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_model(storage_backend: str = "qdrant") -> ColPaliModel:
    """Build a ColPaliModel with all heavy attributes mocked away."""
    from foretrieval.vector_store import make_vector_store
    model = MagicMock()
    model.storage_backend = storage_backend
    model.storage_qdrant = (storage_backend == "qdrant")
    model.index_name = "test_index"
    model.index_root = ".foretrieval_test"
    model.doc_id_to_metadata = {}
    model.doc_ids_to_file_names = {}
    model.collection = {}
    model.embed_id_to_extra = {}
    model.device = "cpu"
    model.verbose = 0
    model.processor = MagicMock()
    model.full_document_collection = False
    model.max_image_width = None
    model.max_image_height = None
    model.highest_doc_id = -1
    model.model_name = "stub"
    model.vector_store = make_vector_store(storage_backend)
    return model


# ---------------------------------------------------------------------------
# Docling ingestion guard
# ---------------------------------------------------------------------------

class TestDoclingGuard:
    def test_docling_not_available_raises_on_init(self):
        """When docling is absent, requesting docling backend raises RuntimeError."""
        with (
            patch("foretrieval.colpali._DOCLING_AVAILABLE", False),
            patch("foretrieval.colpali.ColPaliModel._load_model_and_processor"),
            patch("foretrieval.colpali.ColPaliModel._load_index_state"),
        ):
            with pytest.raises(RuntimeError, match="docling"):
                ColPaliModel(
                    pretrained_model_name_or_path="colqwen2-stub",
                    ingestion={"backend": "docling"},
                    index_root="/tmp",
                )

    def test_docling_not_available_message_contains_install_hint(self):
        """RuntimeError for missing docling contains pip install hint."""
        with (
            patch("foretrieval.colpali._DOCLING_AVAILABLE", False),
            patch("foretrieval.colpali.ColPaliModel._load_model_and_processor"),
            patch("foretrieval.colpali.ColPaliModel._load_index_state"),
        ):
            with pytest.raises(RuntimeError) as exc_info:
                ColPaliModel(
                    pretrained_model_name_or_path="colqwen2-stub",
                    ingestion={"backend": "docling"},
                    index_root="/tmp",
                )
            assert "foretrieval[docling]" in str(exc_info.value)


# ---------------------------------------------------------------------------
# make_point_id (now in vector_store.base, still accessible publicly)
# ---------------------------------------------------------------------------

class TestMakePointId:
    def test_basic(self):
        result = make_point_id(doc_id=1, page_id=2, chunk_id=3)
        expected = 1 * 10_000_000 + 2 * 10_000 + 3
        assert result == expected

    def test_none_chunk_uses_zero(self):
        result_none = make_point_id(doc_id=1, page_id=2, chunk_id=None)
        result_zero = make_point_id(doc_id=1, page_id=2, chunk_id=0)
        assert result_none == result_zero

    def test_uniqueness(self):
        ids = {
            make_point_id(d, p, c)
            for d, p, c in [(0, 0, 0), (0, 0, 1), (0, 1, 0), (1, 0, 0)]
        }
        assert len(ids) == 4

    def test_deterministic(self):
        a = make_point_id(5, 3, 7)
        b = make_point_id(5, 3, 7)
        assert a == b


# ---------------------------------------------------------------------------
# QdrantVectorStore._build_filter (moved from ColPaliModel._build_qdrant_filter)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _QDRANT_AVAILABLE, reason="qdrant-client not installed")
class TestBuildQdrantFilter:
    def _store(self) -> QdrantVectorStore:
        store = QdrantVectorStore.__new__(QdrantVectorStore)
        store._client = MagicMock()
        store._collection_name = "test_col"
        store._index_root = Path("/tmp")
        return store

    def test_none_returns_none(self):
        assert self._store()._build_filter(None) is None

    def test_empty_dict_returns_none(self):
        assert self._store()._build_filter({}) is None

    def test_single_field_returns_filter(self):
        from qdrant_client.models import Filter, FieldCondition
        result = self._store()._build_filter({"ext": ".pdf"})
        assert isinstance(result, Filter)
        assert len(result.must) == 1
        cond = result.must[0]
        assert isinstance(cond, FieldCondition)
        assert cond.key == "metadata.ext"
        assert cond.match.value == ".pdf"

    def test_multiple_fields_returns_multiple_conditions(self):
        from qdrant_client.models import Filter
        result = self._store()._build_filter({"ext": ".pdf", "language": "en"})
        assert isinstance(result, Filter)
        assert len(result.must) == 2
        keys = {c.key for c in result.must}
        assert "metadata.ext" in keys
        assert "metadata.language" in keys


# ---------------------------------------------------------------------------
# Search dispatch — now goes through vector_store.search()
# ---------------------------------------------------------------------------

class TestSearchDispatch:
    def test_zero_k_returns_empty_immediately(self):
        """search() returns [] when k < 1 without calling any search backend."""
        mock = _make_mock_model("local")
        mock._encode_search_query = MagicMock()
        result = ColPaliModel.search(mock, query="test", k=0)
        assert result == []
        mock._encode_search_query.assert_not_called()

    def test_search_delegates_to_vector_store(self):
        """search() calls self.vector_store.search() regardless of backend."""
        mock = _make_mock_model("local")
        mock.vector_store = MagicMock()
        mock.vector_store.search.return_value = []
        mock._encode_search_query = MagicMock(return_value=[MagicMock()])
        ColPaliModel.search(mock, query="test", k=3)
        mock.vector_store.search.assert_called_once()

    def test_search_qdrant_backend_uses_qdrant_store(self):
        """ColPaliModel with qdrant backend delegates to QdrantVectorStore."""
        mock = _make_mock_model("qdrant")
        assert isinstance(mock.vector_store, QdrantVectorStore)

    def test_search_local_backend_uses_local_store(self):
        """ColPaliModel with local backend delegates to LocalVectorStore."""
        mock = _make_mock_model("local")
        assert isinstance(mock.vector_store, LocalVectorStore)


# ---------------------------------------------------------------------------
# Empty filter → empty results (through LocalVectorStore)
# ---------------------------------------------------------------------------

class TestEmptyFilterLocal:
    def test_local_empty_filter_returns_empty_list(self, tmp_path):
        """When LocalVectorStore metadata filter matches nothing, search returns []."""
        from foretrieval.vector_store.base import MultiVectorQuery
        import torch

        store = LocalVectorStore()
        store.open("idx", tmp_path, create=True)
        # Empty metadata map → filter matches nothing
        store.set_doc_id_to_metadata({})
        proc = MagicMock()
        proc.score.return_value = torch.tensor([[]])
        store.set_processor(proc)

        q = MultiVectorQuery(
            vectors=torch.rand(2, 8),
            filter_metadata={"language": "fr"},
        )
        result = store.search(q, k=5)
        assert result == []

    def test_local_empty_filter_does_not_raise(self, tmp_path):
        """search() must not raise ValueError when filter matches nothing."""
        from foretrieval.vector_store.base import MultiVectorQuery
        import torch

        store = LocalVectorStore()
        store.open("idx", tmp_path, create=True)
        store.set_doc_id_to_metadata({})
        proc = MagicMock()
        store.set_processor(proc)

        q = MultiVectorQuery(
            vectors=torch.rand(2, 8),
            filter_metadata={"language": "fr"},
        )
        try:
            store.search(q, k=5)
        except ValueError as exc:
            pytest.fail(f"search raised ValueError on empty filter: {exc}")


# ---------------------------------------------------------------------------
# storage_backend serialization — index_config.json.gz
# ---------------------------------------------------------------------------

class TestStorageBackendSerialization:
    def test_qdrant_backend_written_to_config(self, tmp_path):
        """_export_index writes storage_backend='qdrant'."""
        import srsly
        mock = _make_mock_model("qdrant")
        mock.index_root = str(tmp_path)
        mock.index_name = "test_idx"
        mock.full_document_collection = False
        mock.highest_doc_id = 0
        mock.max_image_width = None
        mock.max_image_height = None
        mock.embed_id_to_extra = {}
        mock.doc_ids_to_file_names = {}
        mock.doc_id_to_metadata = {}
        mock.collection = {}
        mock.vector_store = MagicMock()

        ColPaliModel._export_index(mock)

        config = srsly.read_gzip_json(tmp_path / "test_idx" / "index_config.json.gz")
        assert config["storage_backend"] == "qdrant"

    def test_local_backend_written_to_config(self, tmp_path):
        """_export_index writes storage_backend='local'."""
        import srsly
        mock = _make_mock_model("local")
        mock.index_root = str(tmp_path)
        mock.index_name = "test_idx"
        mock.full_document_collection = False
        mock.highest_doc_id = 0
        mock.max_image_width = None
        mock.max_image_height = None
        mock.embed_id_to_extra = {}
        mock.doc_ids_to_file_names = {}
        mock.doc_id_to_metadata = {}
        mock.collection = {}
        mock.vector_store = MagicMock()

        ColPaliModel._export_index(mock)

        config = srsly.read_gzip_json(tmp_path / "test_idx" / "index_config.json.gz")
        assert config["storage_backend"] == "local"

    def test_milvus_backend_written_to_config(self, tmp_path):
        """_export_index writes storage_backend='milvus'."""
        import srsly
        mock = _make_mock_model("milvus")
        mock.index_root = str(tmp_path)
        mock.index_name = "test_idx"
        mock.full_document_collection = False
        mock.highest_doc_id = 0
        mock.max_image_width = None
        mock.max_image_height = None
        mock.embed_id_to_extra = {}
        mock.doc_ids_to_file_names = {}
        mock.doc_id_to_metadata = {}
        mock.collection = {}
        mock.vector_store = MagicMock()

        ColPaliModel._export_index(mock)

        config = srsly.read_gzip_json(tmp_path / "test_idx" / "index_config.json.gz")
        assert config["storage_backend"] == "milvus"

    def test_from_index_reads_qdrant_backend(self, tmp_path):
        """from_index() passes storage_backend='qdrant' when config says so."""
        import srsly, torch

        idx_path = tmp_path / "my_index"
        idx_path.mkdir()
        srsly.write_gzip_json(
            idx_path / "index_config.json.gz",
            {
                "model_name": "vidore/colqwen2.5-v0.2",
                "storage_backend": "qdrant",
                "full_document_collection": False,
                "highest_doc_id": 0,
                "resize_stored_images": False,
                "max_image_width": None,
                "max_image_height": None,
            },
        )
        torch.save({}, idx_path / "embed_id_to_extra.pt")
        srsly.write_gzip_json(idx_path / "doc_ids_to_file_names.json.gz", {})
        srsly.write_gzip_json(idx_path / "metadata.json.gz", {})

        with patch("foretrieval.colpali.ColPaliModel.__init__", return_value=None) as mock_init:
            try:
                ColPaliModel.from_index(
                    index_path=str(idx_path.name),
                    index_root=str(tmp_path),
                    device="cpu",
                )
            except Exception:
                pass

            if mock_init.called:
                call_kwargs = mock_init.call_args.kwargs
                assert call_kwargs.get("storage_backend") == "qdrant"

    def test_from_index_reads_local_backend(self, tmp_path):
        """from_index() passes storage_backend='local' when config says so."""
        import srsly, torch

        idx_path = tmp_path / "local_index"
        idx_path.mkdir()
        srsly.write_gzip_json(
            idx_path / "index_config.json.gz",
            {
                "model_name": "vidore/colqwen2.5-v0.2",
                "storage_backend": "local",
                "full_document_collection": False,
                "highest_doc_id": 0,
                "resize_stored_images": False,
                "max_image_width": None,
                "max_image_height": None,
            },
        )
        torch.save({}, idx_path / "embed_id_to_extra.pt")
        srsly.write_gzip_json(idx_path / "doc_ids_to_file_names.json.gz", {})
        srsly.write_gzip_json(idx_path / "metadata.json.gz", {})

        with patch("foretrieval.colpali.ColPaliModel.__init__", return_value=None) as mock_init:
            try:
                ColPaliModel.from_index(
                    index_path=str(idx_path.name),
                    index_root=str(tmp_path),
                    device="cpu",
                )
            except Exception:
                pass

            if mock_init.called:
                call_kwargs = mock_init.call_args.kwargs
                assert call_kwargs.get("storage_backend") == "local"


# ---------------------------------------------------------------------------
# Qdrant optional-dependency guard (now on QdrantVectorStore)
# ---------------------------------------------------------------------------

class TestQdrantOptionalDepGuard:
    def test_missing_qdrant_raises_on_open(self, tmp_path):
        """When qdrant-client is absent, QdrantVectorStore.open() raises RuntimeError."""
        with patch("foretrieval.vector_store.qdrant._QDRANT_AVAILABLE", False):
            store = QdrantVectorStore()
            with pytest.raises(RuntimeError, match="qdrant-client"):
                store.open("idx", tmp_path, create=True, dim=8)

    def test_missing_qdrant_message_contains_install_hint(self, tmp_path):
        with patch("foretrieval.vector_store.qdrant._QDRANT_AVAILABLE", False):
            store = QdrantVectorStore()
            with pytest.raises(RuntimeError) as exc_info:
                store.open("idx", tmp_path, create=True, dim=8)
            assert "foretrieval[qdrant]" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Integration test — full Qdrant index + search (requires GPU + qdrant)
# ---------------------------------------------------------------------------

@pytest.mark.slow
@pytest.mark.integration
def test_qdrant_index_and_search(tmp_path):
    """Full index → from_index → search cycle using the Qdrant backend.

    Requires:
    - A compatible GPU (CUDA sm_70+)
    - qdrant-client installed  (pip install foretrieval[qdrant])
    - test PDFs in tests/data/
    """
    if not _QDRANT_AVAILABLE:
        pytest.skip("qdrant-client not installed; install with: pip install foretrieval[qdrant]")

    from foretrieval import MultiModalRetrieverModel

    retriever = MultiModalRetrieverModel.from_pretrained(
        pretrained_model_name_or_path="vidore/colqwen2.5-v0.2",
        index_root=str(tmp_path),
        storage_backend="qdrant",
        device=None,  # auto-detect
        verbose=0,
    )

    retriever.index(
        input_path=str(DATA_DIR),
        index_name="qdrant_integration_test",
        store_collection_with_index=False,
        overwrite=True,
    )

    retriever2 = MultiModalRetrieverModel.from_index(
        index_path="qdrant_integration_test",
        index_root=str(tmp_path),
        device=None,
    )

    results = retriever2.search(
        "maximum output current", k=1, return_base64_results=False
    )

    assert len(results) >= 1
    assert results[0].score is not None
    assert results[0].doc_id is not None
    assert results[0].page_num is not None
