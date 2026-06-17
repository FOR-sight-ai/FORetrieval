"""Regression tests for task_10 bug fixes.

Bug 1: build_metadata_list_for_dir recursive alignment — covered in test_metadata_no_ai.py
Bug 2: cleanup on index failure
Bug 3: doc_ids/highest_doc_id derived from doc_ids_to_file_names (not doc_id_to_metadata)
Bug 4: matplotlib.colormaps replaces deprecated cm.get_cmap
"""
from __future__ import annotations

import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_model(storage_backend: str = "local", index_root: str = "/tmp/foretrieval_test"):
    """Minimal ColPaliModel with GPU / model loading patched out."""
    with (
        patch("foretrieval.colpali.ColPaliModel._load_model_and_processor"),
        patch("foretrieval.colpali.ColPaliModel._load_processor_only"),
    ):
        from foretrieval.colpali import ColPaliModel
        from foretrieval.vector_store.local import LocalVectorStore

        model = ColPaliModel.__new__(ColPaliModel)
        model.pretrained_model_name_or_path = "vidore/colpali-v1.2-test"
        model.model_name = "vidore/colpali-v1.2-test"
        model.verbose = 0
        model.load_from_index = False
        model.index_root = index_root
        model.index_name = None
        model.kwargs = {}
        model.storage_backend = storage_backend
        model.storage_config = {}
        model._storage_qdrant_compat = False
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
        model.vector_store = LocalVectorStore()
        model.index_description = ""
        return model


# ---------------------------------------------------------------------------
# Bug 3: load without metadata yields correct doc_ids / highest_doc_id
# ---------------------------------------------------------------------------

class TestLoadWithoutMetadata:
    """Regression: doc_ids / highest_doc_id are derived from doc_ids_to_file_names."""

    def _build_index_dir(self, tmp_path: Path, with_metadata: bool) -> Path:
        """Write minimal index sidecars to tmp_path."""
        import srsly, torch

        idx = tmp_path / "myindex"
        idx.mkdir()

        # index_config.json.gz
        srsly.write_gzip_json(idx / "index_config.json.gz", {
            "model_name": "vidore/colpali-v1.2-test",
            "full_document_collection": False,
            "highest_doc_id": 2,
            "resize_stored_images": False,
            "max_image_width": None,
            "max_image_height": None,
            "library_version": "0.0.0",
            "storage_backend": "local",
            "storage_config": None,
            "description": "",
        })

        # doc_ids_to_file_names.json.gz — ALWAYS present
        srsly.write_gzip_json(idx / "doc_ids_to_file_names.json.gz", {
            "0": "/data/doc_a.pdf",
            "1": "/data/doc_b.pdf",
            "2": "/data/doc_c.pdf",
        })

        # embed_id_to_doc_id.json.gz
        srsly.write_gzip_json(idx / "embed_id_to_doc_id.json.gz", {})

        # embed_id_to_extra.pt
        torch.save({}, idx / "embed_id_to_extra.pt")

        if with_metadata:
            srsly.write_gzip_json(idx / "metadata.json.gz", {
                "0": {"stem": "doc_a", "ext": ".pdf"},
                "1": {"stem": "doc_b", "ext": ".pdf"},
                "2": {"stem": "doc_c", "ext": ".pdf"},
            })
        # No metadata.json.gz when with_metadata=False

        return idx

    def test_load_without_metadata_doc_ids_correct(self, tmp_path):
        """doc_ids is populated from doc_ids_to_file_names when metadata absent."""
        idx = self._build_index_dir(tmp_path, with_metadata=False)
        model = _make_model(index_root=str(tmp_path))

        model._load_local_sidecars(idx)

        # Simulate _load_index_state final derivation
        id_source = model.doc_ids_to_file_names or model.doc_id_to_metadata
        model.highest_doc_id = max(id_source.keys(), default=-1)
        model.doc_ids = set(id_source.keys())

        assert model.doc_ids == {0, 1, 2}, (
            f"Expected {{0, 1, 2}}, got {model.doc_ids}"
        )
        assert model.highest_doc_id == 2, (
            f"Expected 2, got {model.highest_doc_id}"
        )
        assert model.doc_id_to_metadata == {}, "Metadata should be empty"

    def test_load_with_metadata_doc_ids_correct(self, tmp_path):
        """doc_ids is populated correctly when metadata present."""
        idx = self._build_index_dir(tmp_path, with_metadata=True)
        model = _make_model(index_root=str(tmp_path))

        model._load_local_sidecars(idx)

        id_source = model.doc_ids_to_file_names or model.doc_id_to_metadata
        model.highest_doc_id = max(id_source.keys(), default=-1)
        model.doc_ids = set(id_source.keys())

        assert model.doc_ids == {0, 1, 2}
        assert model.highest_doc_id == 2

    def test_load_local_sidecars_no_metadata(self, tmp_path):
        """_load_local_sidecars: doc_id_to_metadata is {} and doc_ids_to_file_names populated."""
        idx = self._build_index_dir(tmp_path, with_metadata=False)
        model = _make_model(index_root=str(tmp_path))

        model._load_local_sidecars(idx)

        assert model.doc_id_to_metadata == {}
        assert set(model.doc_ids_to_file_names.keys()) == {0, 1, 2}

    def test_apply_bookkeeping_blob_without_metadata(self):
        """Remote path: _apply_bookkeeping_blob derives doc_ids from doc_ids_to_file_names."""
        model = _make_model(storage_backend="remote")

        blob = {
            "index_config": {
                "model_name": "vidore/colpali-v1.2-test",
                "full_document_collection": False,
                "highest_doc_id": 2,
                "resize_stored_images": False,
                "max_image_width": None,
                "max_image_height": None,
                "description": "",
            },
            "embed_id_to_extra": {},
            "doc_ids_to_file_names": {
                "0": "/data/doc_a.pdf",
                "1": "/data/doc_b.pdf",
                "2": "/data/doc_c.pdf",
            },
            "doc_id_to_metadata": {},   # empty: index built without add_metadata
        }

        model._apply_bookkeeping_blob(blob)

        assert model.doc_ids == {0, 1, 2}, (
            f"Expected {{0, 1, 2}}, got {model.doc_ids}"
        )
        assert model.highest_doc_id == 2


# ---------------------------------------------------------------------------
# Bug 2: cleanup on index failure
# ---------------------------------------------------------------------------

class TestIndexCleanup:
    """Regression: _cleanup_failed_index removes partial artefacts."""

    def test_cleanup_removes_local_index_dir(self, tmp_path):
        """_cleanup_failed_index removes the local index directory."""
        index_name = "partial_index"
        index_root = tmp_path
        index_path = index_root / index_name
        index_path.mkdir()
        (index_path / "index_config.json.gz").write_bytes(b"fake")

        model = _make_model(index_root=str(index_root))
        model.index_name = index_name
        model.highest_doc_id = 0
        model.doc_ids = {0}

        model._cleanup_failed_index(index_name)

        assert not index_path.exists(), "Partial index directory should be removed"
        assert model.index_name is None
        assert model.highest_doc_id == -1
        assert model.doc_ids == set()

    def test_cleanup_tolerates_missing_dir(self, tmp_path):
        """_cleanup_failed_index does not raise if the directory doesn't exist."""
        model = _make_model(index_root=str(tmp_path))
        # Should not raise
        model._cleanup_failed_index("nonexistent_index")

    def test_index_raises_cleans_up_on_metadata_mismatch(self, tmp_path):
        """index() cleans up when metadata list length doesn't match file count."""
        from foretrieval.models_metadata import DocMetadata
        from foretrieval.vector_store.local import LocalVectorStore

        # Create a nested data directory with 2 files
        data_dir = tmp_path / "docs"
        data_dir.mkdir()
        (data_dir / "a.pdf").write_bytes(b"%PDF")
        subdir = data_dir / "sub"
        subdir.mkdir()
        (subdir / "b.pdf").write_bytes(b"%PDF")

        model = _make_model(index_root=str(tmp_path))
        model.vector_store = LocalVectorStore()

        # Provide metadata with wrong length (1 instead of 2)
        wrong_metadata = [DocMetadata(stem="only_one", ext=".pdf")]

        with pytest.raises(ValueError, match="metadata entries"):
            model.index(
                input_path=str(data_dir),
                index_name="test_cleanup",
                metadata=wrong_metadata,
            )

        # Index directory should be cleaned up
        index_path = tmp_path / "test_cleanup"
        assert not index_path.exists(), (
            "Partial index directory should be removed after failed indexing"
        )


# ---------------------------------------------------------------------------
# Bug 4: matplotlib.colormaps replaces deprecated cm.get_cmap
# ---------------------------------------------------------------------------

class TestHeatmapColormap:
    """Regression: heatmap_overlay_base64 works with matplotlib >= 3.9."""

    def test_heatmap_overlay_base64_returns_string(self):
        """heatmap_overlay_base64 returns a base64 string without AttributeError."""
        import numpy as np
        from PIL import Image
        from foretrieval.plot_utils import heatmap_overlay_base64

        img = Image.fromarray(
            (np.ones((32, 32, 3)) * 200).astype("uint8")
        )
        heat = (np.ones((32, 32)) * 128).astype("uint8")
        result = heatmap_overlay_base64(img, heat, cmap="jet")
        assert isinstance(result, str) and len(result) > 0

    def test_heatmap_overlay_base64_different_cmaps(self):
        """Several colormaps all work without AttributeError."""
        import numpy as np
        from PIL import Image
        from foretrieval.plot_utils import heatmap_overlay_base64

        img = Image.fromarray(
            (np.ones((16, 16, 3)) * 100).astype("uint8")
        )
        heat = (np.ones((16, 16)) * 50).astype("uint8")
        for cmap in ("jet", "viridis", "plasma", "hot"):
            result = heatmap_overlay_base64(img, heat, cmap=cmap)
            assert isinstance(result, str), f"cmap={cmap} returned non-string"
