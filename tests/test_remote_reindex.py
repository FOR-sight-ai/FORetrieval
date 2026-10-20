"""Regression tests for re-indexing an existing remote collection (issue #5).

``index(..., overwrite=True)`` used to reopen the existing server collection
as-is: every file then collided with the stale points ("Document ID 0 ...
already exists"), was skipped as a faulty PDF, and the server bookkeeping was
replaced with an empty file map while the stale vectors stayed searchable.

These tests run the real vector-DB server app in-process (FastAPI TestClient,
server-side "local" backend in a temp dir) behind the real
VectorDBServerClient / RemoteVectorStore.  Only the model is stubbed: each
image is a solid colour and its embedding is the one-hot vector of that
colour, so search results reveal exactly which documents are stored.
"""
from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image

import foretrieval.vector_db_server.server as server_mod
from foretrieval.vector_db_server.client import VectorDBServerClient
from foretrieval.vector_db_server.config import VectorDBServerConfig
from foretrieval.vector_store.base import MultiVectorQuery, make_point_id
from foretrieval.vector_store.remote import RemoteVectorStore

INDEX = "reindex_test"
COLOURS = {"red": (255, 0, 0), "green": (0, 255, 0), "blue": (0, 0, 255)}
DIM = len(COLOURS)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

@pytest.fixture()
def server_client(tmp_path, monkeypatch):
    """VectorDBServerClient wired to an in-process server with a temp data dir."""
    monkeypatch.setattr(server_mod, "_DATA_DIR", tmp_path / "server_data")
    monkeypatch.setattr(server_mod, "_API_KEY", None)
    server_mod._registry.clear()
    server_mod._locks.clear()

    client = VectorDBServerClient(VectorDBServerConfig(url="http://testserver", backend="local"))
    client._client = TestClient(server_mod.app)
    yield client
    server_mod._registry.clear()
    server_mod._locks.clear()


def _one_hot(rgb) -> torch.Tensor:
    vec = torch.zeros(DIM)
    vec[list(COLOURS.values()).index(tuple(rgb))] = 1.0
    return vec


def _fake_embed_images(images):
    # Two identical "tokens" per image, one-hot on the image's colour.
    return [_one_hot(img.convert("RGB").getpixel((0, 0))).repeat(2, 1) for img in images]


def _fake_process_images(images):
    n = len(images)
    return {"input_ids": torch.zeros(n, 4, dtype=torch.long),
            "image_grid_thw": torch.ones(n, 3, dtype=torch.long)}


def _make_model(server_client, index_root: Path):
    """Remote-backend ColPaliModel with model loading and embedding stubbed."""
    with (
        patch("foretrieval.colpali.ColPaliModel._load_model_and_processor"),
        patch("foretrieval.colpali.ColPaliModel._load_processor_only"),
    ):
        from foretrieval.colpali import ColPaliModel

        model = ColPaliModel.__new__(ColPaliModel)
    model.pretrained_model_name_or_path = "athrael-soju/colqwen3.5-4.5B-v3"
    model.model_name = "athrael-soju/colqwen3.5-4.5B-v3"
    model.verbose = 0
    model.load_from_index = False
    model.index_root = str(index_root)
    model.index_name = None
    model.kwargs = {}
    model.storage_backend = "remote"
    model.storage_config = {"url": "http://testserver", "backend": "local"}
    model._storage_qdrant_compat = False
    model.ingestion = {"backend": "default"}
    model.ingestion_backend = "default"
    model.n_gpu = 0
    model.device = "cpu"
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
    model.model = None
    model.processor = MagicMock()
    model.processor.process_images.side_effect = _fake_process_images
    model._remote_client = MagicMock()
    model._remote_client.embed_images.side_effect = _fake_embed_images
    model.vector_store = RemoteVectorStore(server_client, backend="local")
    model.index_description = ""
    return model


def _make_data_dir(root: Path, colours) -> Path:
    root.mkdir(parents=True)
    for name in colours:
        Image.new("RGB", (8, 8), COLOURS[name]).save(root / f"{name}.png")
    return root


def _file_names(server_client) -> dict:
    bk = server_client.get_bookkeeping(INDEX) or {}
    return {int(k): Path(v).name for k, v in bk.get("doc_ids_to_file_names", {}).items()}


def _search_doc_ids(server_client, colour: str) -> set:
    query = MultiVectorQuery(vectors=_one_hot(COLOURS[colour]).unsqueeze(0))
    return {hit.payload["doc_id"] for hit in server_client.search(INDEX, query, k=10)}


@pytest.fixture()
def built_index(server_client, tmp_path):
    """An existing remote index with red.png (doc 0) and blue.png (doc 1)."""
    data_v1 = _make_data_dir(tmp_path / "v1", ["red", "blue"])
    model = _make_model(server_client, tmp_path / "idx")
    model.index(input_path=data_v1, index_name=INDEX)
    assert _file_names(server_client) == {0: "blue.png", 1: "red.png"}
    return model


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestOverwriteExistingRemoteIndex:
    def test_overwrite_from_fresh_instance_replaces_collection(
        self, server_client, built_index, tmp_path, caplog
    ):
        data_v2 = _make_data_dir(tmp_path / "v2", ["green"])
        model = _make_model(server_client, tmp_path / "idx")

        with caplog.at_level(logging.WARNING, logger="foretrieval.colpali"):
            result = model.index(input_path=data_v2, index_name=INDEX, overwrite=True)

        assert {k: Path(v).name for k, v in result.items()} == {0: "green.png"}
        assert "Skipping faulty PDF" not in caplog.text
        assert _file_names(server_client) == {0: "green.png"}
        # Stale points from the previous build are gone ...
        assert not server_client.point_exists(INDEX, make_point_id(1, 1))
        assert _search_doc_ids(server_client, "red") == {0}
        # ... and doc 0 now holds the new document's vector.
        stored = server_client.fetch_vector(INDEX, make_point_id(0, 1))
        assert torch.equal(stored[0], _one_hot(COLOURS["green"]))

    def test_overwrite_on_same_instance_resets_in_memory_state(
        self, server_client, built_index, tmp_path
    ):
        data_v2 = _make_data_dir(tmp_path / "v2", ["green"])

        built_index.index(input_path=data_v2, index_name=INDEX, overwrite=True)

        assert built_index.doc_ids == {0}
        assert {k: Path(v).name for k, v in built_index.doc_ids_to_file_names.items()} == {0: "green.png"}
        assert set(built_index.embed_id_to_extra) == {make_point_id(0, 1)}
        assert _file_names(server_client) == {0: "green.png"}

    def test_overwrite_on_missing_collection_builds_it(self, server_client, tmp_path):
        data = _make_data_dir(tmp_path / "v1", ["green"])
        model = _make_model(server_client, tmp_path / "idx")

        model.index(input_path=data, index_name=INDEX, overwrite=True)

        assert _file_names(server_client) == {0: "green.png"}


class TestNoOverwriteExistingRemoteIndex:
    def test_existing_index_is_left_untouched(self, server_client, built_index, tmp_path, caplog):
        data_v2 = _make_data_dir(tmp_path / "v2", ["green"])
        model = _make_model(server_client, tmp_path / "idx")

        with caplog.at_level(logging.WARNING, logger="foretrieval.colpali"):
            result = model.index(input_path=data_v2, index_name=INDEX, overwrite=False)

        assert result is None
        assert "already exists on the server" in caplog.text
        assert _file_names(server_client) == {0: "blue.png", 1: "red.png"}
        assert _search_doc_ids(server_client, "blue") >= {0}
        assert server_client.point_exists(INDEX, make_point_id(1, 1))


class TestNothingIndexedWarning:
    def test_warns_when_every_file_is_skipped(self, server_client, tmp_path, caplog):
        data = _make_data_dir(tmp_path / "v1", ["red", "green"])
        model = _make_model(server_client, tmp_path / "idx")
        model._remote_client.embed_images.side_effect = RuntimeError("embedding server down")

        with caplog.at_level(logging.WARNING, logger="foretrieval.colpali"):
            model.index(input_path=data, index_name=INDEX)

        assert "No documents were indexed." in caplog.text
