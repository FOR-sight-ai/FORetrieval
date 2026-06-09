"""Tests for recursive directory indexing in ColPaliModel.

Verifies that index(), _process_directory(), and update_index_from_folder()
traverse subdirectories recursively using rglob instead of iterdir.

No GPU or real model required — ColPaliModel is built with mocked loading
and _process_and_add_to_index is patched to avoid actual embedding work.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest

from foretrieval.vector_store import make_vector_store


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_model(tmp_path: Path, storage_backend: str = "local"):
    """Build a minimal ColPaliModel with GPU/model loading mocked."""
    from foretrieval.colpali import ColPaliModel

    model = ColPaliModel.__new__(ColPaliModel)
    model.pretrained_model_name_or_path = "vidore/colpali-v1.2-test"
    model.model_name = "vidore/colpali-v1.2-test"
    model.verbose = 0
    model.load_from_index = False
    model.index_root = str(tmp_path / "index_root")
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
    model.SOURCE_EXTS = {".docx", ".txt", ".png", ".jpg"}
    model.IMAGE_EXTS = {".png", ".jpg", ".jpeg"}
    model._remote_client = None
    model.model = None
    model.processor = MagicMock()
    model.vector_store = make_vector_store(storage_backend, {})
    return model


def _make_nested_corpus(base: Path) -> list[Path]:
    """Create a nested directory of dummy files and return their paths sorted."""
    # base/
    #   a.pdf
    #   sub1/
    #     b.pdf
    #     sub2/
    #       c.pdf
    (base / "sub1" / "sub2").mkdir(parents=True)
    files = [
        base / "a.pdf",
        base / "sub1" / "b.pdf",
        base / "sub1" / "sub2" / "c.pdf",
    ]
    for f in files:
        f.write_bytes(b"%PDF-1.4 dummy")
    return sorted(files, key=lambda p: p.relative_to(base))


# ---------------------------------------------------------------------------
# index() — recursive traversal
# ---------------------------------------------------------------------------

class TestIndexRecursive:
    def test_index_visits_all_nested_files(self, tmp_path):
        """index() must find files in subdirectories via rglob."""
        corpus = tmp_path / "corpus"
        corpus.mkdir()
        expected_files = _make_nested_corpus(corpus)

        model = _make_model(tmp_path)
        visited = []

        def fake_process(item, *args, **kwargs):
            visited.append(Path(item))
            model.doc_ids_to_file_names[kwargs.get("doc_id", len(visited))] = str(item)
            model.doc_ids.add(kwargs.get("doc_id", len(visited)))
            model.highest_doc_id = max(model.highest_doc_id, kwargs.get("doc_id", len(visited)))
            return item

        with (
            patch.object(model, "_process_and_add_to_index", side_effect=fake_process),
            patch.object(model, "_export_index"),
            patch.object(model.vector_store, "open"),
            patch.object(model.vector_store, "set_processor", create=True),
            patch.object(model.vector_store, "set_doc_id_to_metadata", create=True),
        ):
            model.index(
                input_path=corpus,
                index_name="nested_test",
                overwrite=False,
            )

        assert sorted(visited, key=lambda p: p.relative_to(corpus)) == expected_files

    def test_index_flat_dir_unchanged_behavior(self, tmp_path):
        """Flat directory: rglob and old iterdir produce identical file sets."""
        corpus = tmp_path / "flat"
        corpus.mkdir()
        files = []
        for name in ["x.pdf", "y.pdf", "z.pdf"]:
            f = corpus / name
            f.write_bytes(b"%PDF-1.4 dummy")
            files.append(f)
        files = sorted(files, key=lambda p: p.relative_to(corpus))

        model = _make_model(tmp_path)
        visited = []

        def fake_process(item, *args, **kwargs):
            visited.append(Path(item))
            model.doc_ids_to_file_names[kwargs.get("doc_id", len(visited))] = str(item)
            model.doc_ids.add(kwargs.get("doc_id", len(visited)))
            model.highest_doc_id = max(model.highest_doc_id, kwargs.get("doc_id", len(visited)))
            return item

        with (
            patch.object(model, "_process_and_add_to_index", side_effect=fake_process),
            patch.object(model, "_export_index"),
            patch.object(model.vector_store, "open"),
            patch.object(model.vector_store, "set_processor", create=True),
            patch.object(model.vector_store, "set_doc_id_to_metadata", create=True),
        ):
            model.index(input_path=corpus, index_name="flat_test", overwrite=False)

        assert sorted(visited, key=lambda p: p.relative_to(corpus)) == files

    def test_index_overwrite_false_no_existing_index_creates_new(self, tmp_path):
        """overwrite=False with no existing index directory must proceed normally."""
        corpus = tmp_path / "corpus"
        corpus.mkdir()
        (corpus / "doc.pdf").write_bytes(b"%PDF-1.4 dummy")

        model = _make_model(tmp_path)
        visited = []

        def fake_process(item, *args, **kwargs):
            visited.append(Path(item))
            model.doc_ids_to_file_names[kwargs.get("doc_id", 0)] = str(item)
            model.doc_ids.add(kwargs.get("doc_id", 0))
            model.highest_doc_id = max(model.highest_doc_id, kwargs.get("doc_id", 0))
            return item

        with (
            patch.object(model, "_process_and_add_to_index", side_effect=fake_process),
            patch.object(model, "_export_index"),
            patch.object(model.vector_store, "open"),
            patch.object(model.vector_store, "set_processor", create=True),
            patch.object(model.vector_store, "set_doc_id_to_metadata", create=True),
        ):
            # Must NOT raise and must NOT return None
            result = model.index(
                input_path=corpus,
                index_name="brand_new_index",
                overwrite=False,
            )

        assert result is not None, "index() returned None — did not create a new index"
        assert len(visited) == 1


# ---------------------------------------------------------------------------
# _process_directory() — recursive traversal
# ---------------------------------------------------------------------------

class TestProcessDirectoryRecursive:
    def test_process_directory_visits_nested_files(self, tmp_path):
        """_process_directory() must recurse into subdirectories."""
        directory = tmp_path / "docs"
        directory.mkdir()
        expected_files = _make_nested_corpus(directory)

        model = _make_model(tmp_path)
        model.index_name = "test_index"
        visited = []

        def fake_process(item, *args, **kwargs):
            visited.append(Path(item))
            return item

        with patch.object(model, "_process_and_add_to_index", side_effect=fake_process):
            model._process_directory(
                directory=directory,
                store_collection_with_index=False,
                base_doc_id=0,
                metadata=None,
                batch_size=1,
            )

        assert sorted(visited, key=lambda p: p.relative_to(directory)) == expected_files


# ---------------------------------------------------------------------------
# update_index_from_folder() — recursive traversal
# ---------------------------------------------------------------------------

class TestUpdateIndexFromFolderRecursive:
    def test_update_visits_nested_new_files(self, tmp_path):
        """update_index_from_folder() must recurse into subdirectories."""
        folder = tmp_path / "docs"
        folder.mkdir()
        expected_files = _make_nested_corpus(folder)

        model = _make_model(tmp_path)
        model.index_name = "test_index"
        model.doc_ids_to_file_names = {}
        model.doc_ids = set()
        model.highest_doc_id = -1
        visited = []

        def fake_process(item, *args, **kwargs):
            visited.append(Path(item))
            model.doc_ids_to_file_names[kwargs.get("doc_id", len(visited))] = str(item)
            model.doc_ids.add(kwargs.get("doc_id", len(visited)))
            model.highest_doc_id = max(model.highest_doc_id, kwargs.get("doc_id", len(visited)))
            return item

        with (
            patch.object(model, "_process_and_add_to_index", side_effect=fake_process),
            patch.object(model, "_export_index"),
        ):
            model.update_index_from_folder(folder=folder)

        assert sorted(visited, key=lambda p: p.relative_to(folder)) == expected_files

    def test_update_skips_already_indexed_nested_files(self, tmp_path):
        """update_index_from_folder() must skip already-indexed files in subdirs."""
        folder = tmp_path / "docs"
        folder.mkdir()
        (folder / "sub").mkdir()
        already_indexed = folder / "sub" / "old.pdf"
        new_file = folder / "sub" / "new.pdf"
        already_indexed.write_bytes(b"%PDF-1.4 old")
        new_file.write_bytes(b"%PDF-1.4 new")

        model = _make_model(tmp_path)
        model.index_name = "test_index"
        model.doc_ids_to_file_names = {0: str(already_indexed.resolve())}
        model.doc_ids = {0}
        model.highest_doc_id = 0
        visited = []

        def fake_process(item, *args, **kwargs):
            visited.append(Path(item))
            return item

        with (
            patch.object(model, "_process_and_add_to_index", side_effect=fake_process),
            patch.object(model, "_export_index"),
        ):
            model.update_index_from_folder(folder=folder)

        assert visited == [new_file], f"Expected only new.pdf, got {visited}"
