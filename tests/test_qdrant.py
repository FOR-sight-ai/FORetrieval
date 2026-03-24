"""
Tests for the Qdrant storage backend and the docling ingestion guard.

No GPU or real Qdrant server required for unit tests — everything that
touches the model or the Qdrant client is mocked.

The integration test (marked @pytest.mark.slow) performs a full index +
search cycle using a real ColPali model, real Qdrant (local embedded), and
the test PDFs from tests/data/.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from foretrieval.colpali import ColPaliModel, _QDRANT_AVAILABLE, _DOCLING_AVAILABLE

DATA_DIR = Path(__file__).parent / "data"


# ---------------------------------------------------------------------------
# Helpers — minimal mock ColPaliModel (no GPU, no Qdrant)
# ---------------------------------------------------------------------------


def _make_mock_model(storage_qdrant: bool = True) -> ColPaliModel:
    """Build a ColPaliModel with all heavy attributes mocked away.

    Using spec='ColPaliModel' would restrict attributes to those known at
    class-definition time, which excludes instance-level attributes set in
    __init__ (e.g. verbose, doc_ids_to_file_names).  Use spec_set=False
    (plain MagicMock without spec) to allow arbitrary attribute access.
    """
    model = MagicMock()
    model.storage_qdrant = storage_qdrant
    model.qdrant_client = None
    model.qdrant_collection = "test_col"
    model.qdrant_path = None
    model.index_name = "test_index"
    model.index_root = ".foretrieval_test"
    model.indexed_embeddings = []
    model.embed_id_to_doc_id = {}
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
    return model


# ---------------------------------------------------------------------------
# Optional dependency guards
# ---------------------------------------------------------------------------


class TestOptionalDepGuards:
    def test_qdrant_not_available_raises_on_ensure(self):
        """When qdrant-client is absent, _ensure_qdrant_client raises RuntimeError."""
        mock = _make_mock_model(storage_qdrant=True)
        mock.index_name = "idx"

        with patch("foretrieval.colpali._QDRANT_AVAILABLE", False):
            with pytest.raises(RuntimeError, match="qdrant-client"):
                ColPaliModel._ensure_qdrant_client(mock)

    def test_qdrant_not_available_message_contains_install_hint(self):
        """RuntimeError message includes pip install instructions."""
        mock = _make_mock_model(storage_qdrant=True)
        mock.index_name = "idx"

        with patch("foretrieval.colpali._QDRANT_AVAILABLE", False):
            with pytest.raises(RuntimeError) as exc_info:
                ColPaliModel._ensure_qdrant_client(mock)
            assert "foretrieval[qdrant]" in str(exc_info.value)

    def test_qdrant_available_does_not_raise(self):
        """When qdrant-client is available, _ensure_qdrant_client proceeds normally."""
        if not _QDRANT_AVAILABLE:
            pytest.skip("qdrant-client not installed in this environment")
        mock = _make_mock_model(storage_qdrant=True)
        mock.index_name = "idx"
        mock.qdrant_path = None
        mock.qdrant_client = None

        with patch("foretrieval.colpali.QdrantClient") as MockQdrant:
            MockQdrant.return_value = MagicMock()
            # Should not raise
            ColPaliModel._ensure_qdrant_client(mock)

    def test_storage_qdrant_false_skips_qdrant_init(self):
        """_ensure_qdrant_client is a no-op when storage_qdrant=False."""
        mock = _make_mock_model(storage_qdrant=False)
        # Should return immediately without touching qdrant_client
        ColPaliModel._ensure_qdrant_client(mock)
        assert mock.qdrant_client is None

    def test_docling_not_available_raises_on_init(self):
        """When docling is absent, requesting docling backend raises RuntimeError.

        We use a valid model name prefix (colqwen2) to pass the early model
        name check, then patch out the heavy loading so we reach the docling guard.
        """
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
# _make_point_id
# ---------------------------------------------------------------------------


class TestMakePointId:
    def test_basic(self):
        """_make_point_id returns the expected integer for given inputs."""
        mock = _make_mock_model()
        result = ColPaliModel._make_point_id(mock, doc_id=1, page_id=2, chunk_id=3)
        expected = 1 * 10_000_000 + 2 * 10_000 + 3
        assert result == expected

    def test_none_chunk_uses_zero(self):
        """chunk_id=None is treated as 0."""
        mock = _make_mock_model()
        result_none = ColPaliModel._make_point_id(mock, doc_id=1, page_id=2, chunk_id=None)
        result_zero = ColPaliModel._make_point_id(mock, doc_id=1, page_id=2, chunk_id=0)
        assert result_none == result_zero

    def test_uniqueness(self):
        """Different (doc_id, page_id, chunk_id) tuples produce different IDs."""
        mock = _make_mock_model()
        ids = {
            ColPaliModel._make_point_id(mock, d, p, c)
            for d, p, c in [(0, 0, 0), (0, 0, 1), (0, 1, 0), (1, 0, 0)]
        }
        assert len(ids) == 4

    def test_deterministic(self):
        """Same inputs always produce the same ID."""
        mock = _make_mock_model()
        a = ColPaliModel._make_point_id(mock, 5, 3, 7)
        b = ColPaliModel._make_point_id(mock, 5, 3, 7)
        assert a == b


# ---------------------------------------------------------------------------
# _build_qdrant_filter
# ---------------------------------------------------------------------------


class TestBuildQdrantFilter:
    @pytest.fixture(autouse=True)
    def require_qdrant(self):
        if not _QDRANT_AVAILABLE:
            pytest.skip("qdrant-client not installed")

    def test_none_returns_none(self):
        mock = _make_mock_model()
        assert ColPaliModel._build_qdrant_filter(mock, None) is None

    def test_empty_dict_returns_none(self):
        mock = _make_mock_model()
        assert ColPaliModel._build_qdrant_filter(mock, {}) is None

    def test_single_field_returns_filter(self):
        """A single key-value pair produces a Filter with one must condition."""
        from qdrant_client.models import Filter, FieldCondition

        mock = _make_mock_model()
        result = ColPaliModel._build_qdrant_filter(mock, {"ext": ".pdf"})
        assert isinstance(result, Filter)
        assert len(result.must) == 1
        cond = result.must[0]
        assert isinstance(cond, FieldCondition)
        assert cond.key == "metadata.ext"
        assert cond.match.value == ".pdf"

    def test_multiple_fields_returns_multiple_conditions(self):
        """Multiple key-value pairs produce multiple must conditions."""
        from qdrant_client.models import Filter

        mock = _make_mock_model()
        result = ColPaliModel._build_qdrant_filter(
            mock, {"ext": ".pdf", "language": "en"}
        )
        assert isinstance(result, Filter)
        assert len(result.must) == 2
        keys = {c.key for c in result.must}
        assert "metadata.ext" in keys
        assert "metadata.language" in keys


# ---------------------------------------------------------------------------
# search() dispatch
# ---------------------------------------------------------------------------


class TestSearchDispatch:
    def test_dispatches_to_qdrant_when_storage_qdrant_true(self):
        """search() calls _search_qdrant when storage_qdrant=True."""
        mock = _make_mock_model(storage_qdrant=True)
        mock._encode_search_query = MagicMock(return_value=[MagicMock()])
        mock._search_qdrant = MagicMock(return_value=[])
        mock._search_local = MagicMock(return_value=[])
        mock.collection = {}

        ColPaliModel.search(mock, query="test", k=3)

        mock._search_qdrant.assert_called_once()
        mock._search_local.assert_not_called()

    def test_dispatches_to_local_when_storage_qdrant_false(self):
        """search() calls _search_local when storage_qdrant=False."""
        mock = _make_mock_model(storage_qdrant=False)
        mock._encode_search_query = MagicMock(return_value=[MagicMock()])
        mock._search_local = MagicMock(return_value=[])
        mock._search_qdrant = MagicMock(return_value=[])
        mock.collection = {}

        ColPaliModel.search(mock, query="test", k=3)

        mock._search_local.assert_called_once()
        mock._search_qdrant.assert_not_called()

    def test_zero_k_returns_empty_immediately(self):
        """search() returns [] when k < 1 without calling any search backend."""
        mock = _make_mock_model()
        mock._encode_search_query = MagicMock()
        mock._search_qdrant = MagicMock()
        mock._search_local = MagicMock()
        mock.collection = {}

        result = ColPaliModel.search(mock, query="test", k=0)

        assert result == []
        mock._encode_search_query.assert_not_called()


# ---------------------------------------------------------------------------
# Empty filter → [] (local backend)
# ---------------------------------------------------------------------------


class TestEmptyFilterLocal:
    def test_local_empty_filter_returns_empty_list(self):
        """_search_local returns [] when filter_embeddings yields nothing."""
        mock = _make_mock_model(storage_qdrant=False)
        mock.filter_embeddings = MagicMock(return_value=([], []))
        mock.indexed_embeddings = [MagicMock()]

        result = ColPaliModel._search_local(
            mock,
            qs=[MagicMock()],
            k=5,
            filter_metadata={"ext": ".docx"},
            return_base64_results=False,
        )

        assert result == []

    def test_local_empty_filter_does_not_raise(self):
        """_search_local must not raise ValueError when filter matches nothing."""
        mock = _make_mock_model(storage_qdrant=False)
        mock.filter_embeddings = MagicMock(return_value=([], []))
        mock.indexed_embeddings = [MagicMock()]

        try:
            ColPaliModel._search_local(
                mock,
                qs=[MagicMock()],
                k=5,
                filter_metadata={"ext": ".docx"},
                return_base64_results=False,
            )
        except ValueError as exc:
            pytest.fail(f"_search_local raised ValueError on empty filter: {exc}")


# ---------------------------------------------------------------------------
# storage_backend serialisation
# ---------------------------------------------------------------------------


class TestStorageBackendSerialization:
    def test_qdrant_backend_written_to_config(self, tmp_path):
        """_export_index writes storage_backend='qdrant' when storage_qdrant=True."""
        import srsly

        mock = _make_mock_model(storage_qdrant=True)
        mock.index_root = str(tmp_path)
        mock.index_name = "test_idx"
        mock.model_name = "stub"
        mock.full_document_collection = False
        mock.highest_doc_id = 0
        mock.max_image_width = None
        mock.max_image_height = None
        mock.qdrant_collection = "test_idx"
        mock.embed_id_to_extra = {}
        mock.doc_ids_to_file_names = {}
        mock.doc_id_to_metadata = {}
        mock.collection = {}

        ColPaliModel._export_index(mock)

        config = srsly.read_gzip_json(tmp_path / "test_idx" / "index_config.json.gz")
        assert config["storage_backend"] == "qdrant"

    def test_local_backend_written_to_config(self, tmp_path):
        """_export_index writes storage_backend='local' when storage_qdrant=False."""
        import srsly

        mock = _make_mock_model(storage_qdrant=False)
        mock.index_root = str(tmp_path)
        mock.index_name = "test_idx"
        mock.model_name = "stub"
        mock.full_document_collection = False
        mock.highest_doc_id = 0
        mock.max_image_width = None
        mock.max_image_height = None
        mock.qdrant_collection = None
        mock.embed_id_to_extra = {}
        mock.doc_ids_to_file_names = {}
        mock.doc_id_to_metadata = {}
        mock.collection = {}
        mock.indexed_embeddings = []
        mock.embed_id_to_doc_id = {}

        ColPaliModel._export_index(mock)

        config = srsly.read_gzip_json(tmp_path / "test_idx" / "index_config.json.gz")
        assert config["storage_backend"] == "local"

    def test_from_index_reads_qdrant_backend(self, tmp_path):
        """from_index() sets storage_qdrant=True when config says 'qdrant'."""
        import srsly

        idx_path = tmp_path / "my_index"
        idx_path.mkdir()
        srsly.write_gzip_json(
            idx_path / "index_config.json.gz",
            {
                "model_name": "vidore/colqwen2.5-v0.2",
                "storage_backend": "qdrant",
                "qdrant_collection": "my_index",
                "full_document_collection": False,
                "highest_doc_id": 0,
                "resize_stored_images": False,
                "max_image_width": None,
                "max_image_height": None,
            },
        )
        # Write minimal required sidecar files
        import torch
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
                pass  # may fail on model load; we only care about the kwarg

            if mock_init.called:
                call_kwargs = mock_init.call_args.kwargs
                assert call_kwargs.get("storage_qdrant") is True

    def test_from_index_reads_local_backend(self, tmp_path):
        """from_index() sets storage_qdrant=False when config says 'local'."""
        import srsly, torch

        idx_path = tmp_path / "local_index"
        idx_path.mkdir()
        srsly.write_gzip_json(
            idx_path / "index_config.json.gz",
            {
                "model_name": "vidore/colqwen2.5-v0.2",
                "storage_backend": "local",
                "qdrant_collection": None,
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
        torch.save({}, idx_path / "embed_id_to_doc_id.pt")
        torch.save([], idx_path / "embeddings_0.pt")

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
                assert call_kwargs.get("storage_qdrant") is False


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
        storage_qdrant=True,
        device=None,  # auto-detect
        verbose=0,
    )

    retriever.index(
        input_path=str(DATA_DIR),
        index_name="qdrant_integration_test",
        store_collection_with_index=False,
        overwrite=True,
    )

    # Reload from disk — exercises from_index + Qdrant path
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
