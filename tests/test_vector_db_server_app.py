"""Tests for the FastAPI vector-DB server app.

Uses FastAPI's TestClient for in-process HTTP calls — no real server needed.
All storage is local (temp dir) to keep tests fast and self-contained.
"""

import os
import tempfile
from unittest.mock import patch

import pytest
import torch
from fastapi.testclient import TestClient

# ── patch FOR_DB_DATA_DIR before importing the server ──────────────────────
_TMP = tempfile.mkdtemp(prefix="foretrieval_db_test_")

# We patch the module-level constants before the app is created.
import foretrieval.vector_db_server.server as _server_mod

_server_mod._DATA_DIR = __import__("pathlib").Path(_TMP)
_server_mod._API_KEY = None          # auth off by default
_server_mod._registry.clear()
_server_mod._locks.clear()

from foretrieval.vector_db_server.server import app  # noqa: E402
from foretrieval.vector_db_server.client import _dumps, _loads
from foretrieval.vector_store.base import make_point_id


@pytest.fixture(autouse=True)
def _clean_registry():
    """Clear in-memory registry and lock state between tests."""
    _server_mod._registry.clear()
    _server_mod._locks.clear()
    yield
    _server_mod._registry.clear()
    _server_mod._locks.clear()


@pytest.fixture()
def client():
    return TestClient(app)


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------

class TestHealth:
    def test_health_ok(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"


# ---------------------------------------------------------------------------
# Auth middleware
# ---------------------------------------------------------------------------

class TestAuth:
    def test_missing_auth_header_returns_401(self, client):
        _server_mod._API_KEY = "secret"
        try:
            r = client.get("/v1/collection/myidx/exists")
            assert r.status_code == 401
        finally:
            _server_mod._API_KEY = None

    def test_wrong_token_returns_401(self, client):
        _server_mod._API_KEY = "secret"
        try:
            r = client.get(
                "/v1/collection/myidx/exists",
                headers={"Authorization": "Bearer wrongtoken"},
            )
            assert r.status_code == 401
        finally:
            _server_mod._API_KEY = None

    def test_correct_token_accepted(self, client):
        _server_mod._API_KEY = "secret"
        try:
            r = client.get(
                "/v1/collection/myidx/exists",
                headers={"Authorization": "Bearer secret"},
            )
            assert r.status_code == 200
        finally:
            _server_mod._API_KEY = None

    def test_health_exempt_from_auth(self, client):
        _server_mod._API_KEY = "secret"
        try:
            r = client.get("/health")
            assert r.status_code == 200
        finally:
            _server_mod._API_KEY = None


# ---------------------------------------------------------------------------
# Collection management
# ---------------------------------------------------------------------------

class TestCollectionManagement:
    def test_collection_does_not_exist(self, client):
        r = client.get("/v1/collection/noexist/exists")
        assert r.status_code == 200
        assert r.json()["exists"] is False

    def test_open_creates_collection(self, client):
        r = client.post(
            "/v1/collection/open",
            json={"index_name": "test_open", "backend": "local", "create": True, "dim": 8},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["opened"] is True
        assert body["created"] is True

    def test_open_existing_collection(self, client):
        client.post(
            "/v1/collection/open",
            json={"index_name": "test_reopen", "backend": "local", "create": True, "dim": 8},
        )
        r = client.post(
            "/v1/collection/open",
            json={"index_name": "test_reopen", "backend": "local", "create": False},
        )
        assert r.status_code == 200
        assert r.json()["created"] is False

    def test_open_nonexistent_without_create_returns_404(self, client):
        r = client.post(
            "/v1/collection/open",
            json={"index_name": "missing", "backend": "local", "create": False},
        )
        assert r.status_code == 404

    def test_create_collection_endpoint(self, client):
        r = client.post(
            "/v1/collection",
            json={"index_name": "via_create", "backend": "local", "dim": 8},
        )
        assert r.status_code == 200
        assert r.json()["created"] is True

    def test_collection_exists_after_create(self, client):
        client.post(
            "/v1/collection",
            json={"index_name": "check_exists", "backend": "local", "dim": 8},
        )
        r = client.get("/v1/collection/check_exists/exists")
        assert r.json()["exists"] is True


# ---------------------------------------------------------------------------
# Upsert → search → fetch_vector full round-trip
# ---------------------------------------------------------------------------

class TestUpsertSearchFetch:
    def _open(self, client, name="rt_test", backend="local", dim=8):
        client.post(
            "/v1/collection/open",
            json={"index_name": name, "backend": backend, "create": True, "dim": dim},
        )

    def _upsert_points(self, client, name, n=3, dim=8):
        points = []
        for i in range(n):
            pid = make_point_id(1, i)
            points.append({
                "point_id": pid,
                "vector": torch.randn(5, dim),
                "payload": {"doc_id": 1, "page_id": i},
            })
        body = _dumps(points)
        return client.post(
            f"/v1/upsert/{name}",
            content=body,
            headers={"Content-Type": "application/octet-stream"},
        )

    def test_upsert_returns_count(self, client):
        self._open(client)
        r = self._upsert_points(client, "rt_test", n=2)
        assert r.status_code == 200
        assert r.json()["upserted"] == 2

    def test_point_exists_after_upsert(self, client):
        self._open(client, name="pe_test")
        self._upsert_points(client, "pe_test", n=1)
        pid = make_point_id(1, 0)
        r = client.get(f"/v1/point/pe_test/{pid}/exists")
        assert r.json()["exists"] is True

    def test_point_not_exists_before_upsert(self, client):
        self._open(client, name="pne_test")
        r = client.get(f"/v1/point/pne_test/9999999/exists")
        assert r.json()["exists"] is False

    def test_search_returns_hits(self, client):
        self._open(client, name="srch_test")
        self._upsert_points(client, "srch_test", n=3)

        query_vec = torch.randn(4, 8)
        wire = _dumps({"vectors": query_vec, "filter_metadata": None, "k": 2})
        r = client.post(
            "/v1/search/srch_test",
            content=wire,
            headers={"Content-Type": "application/octet-stream"},
        )
        assert r.status_code == 200
        hits = _loads(r.content)
        assert len(hits) <= 2
        for h in hits:
            assert "point_id" in h
            assert "score" in h

    def test_fetch_vector_returns_tensor(self, client):
        self._open(client, name="fv_test")
        self._upsert_points(client, "fv_test", n=1)
        pid = make_point_id(1, 0)
        r = client.get(f"/v1/vector/fv_test/{pid}")
        assert r.status_code == 200
        tensor = _loads(r.content)
        assert tensor.shape[-1] == 8

    def test_fetch_vector_missing_returns_404(self, client):
        self._open(client, name="fv_missing")
        r = client.get("/v1/vector/fv_missing/9999999")
        assert r.status_code == 404

    def test_search_on_unknown_collection_returns_404(self, client):
        wire = _dumps({"vectors": torch.randn(2, 8), "filter_metadata": None, "k": 3})
        r = client.post(
            "/v1/search/no_such_col",
            content=wire,
            headers={"Content-Type": "application/octet-stream"},
        )
        assert r.status_code == 404


# ---------------------------------------------------------------------------
# Delete collection
# ---------------------------------------------------------------------------

class TestDeleteCollection:
    def test_delete_removes_collection(self, client):
        client.post(
            "/v1/collection/open",
            json={"index_name": "to_delete", "backend": "local", "create": True, "dim": 8},
        )
        r = client.delete("/v1/collection/to_delete")
        assert r.status_code == 200
        assert r.json()["deleted"] is True

        r2 = client.get("/v1/collection/to_delete/exists")
        assert r2.json()["exists"] is False

    def test_delete_nonexistent_ok(self, client):
        r = client.delete("/v1/collection/ghost_col")
        assert r.status_code == 200


# ---------------------------------------------------------------------------
# Bookkeeping (server-side ColPali index metadata)
# ---------------------------------------------------------------------------

class TestBookkeeping:
    def _blob(self):
        return {
            "index_config": {
                "model_name": "vidore/colqwen-test",
                "highest_doc_id": 4,
                "description": "demo",
            },
            "embed_id_to_extra": {0: {"orig_size": (10, 20)}},
            "doc_ids_to_file_names": {1: "a.pdf"},
            "doc_id_to_metadata": {1: {"title": "A"}},
        }

    def test_get_missing_returns_404(self, client):
        r = client.get("/v1/collection/bk_missing/bookkeeping")
        assert r.status_code == 404

    def test_put_then_get_roundtrip(self, client):
        blob = self._blob()
        r = client.put(
            "/v1/collection/bk_rt/bookkeeping",
            content=_dumps(blob),
            headers={"Content-Type": "application/octet-stream"},
        )
        assert r.status_code == 200
        assert r.json()["stored"] is True

        r2 = client.get("/v1/collection/bk_rt/bookkeeping")
        assert r2.status_code == 200
        loaded = _loads(r2.content)
        assert loaded["index_config"]["model_name"] == "vidore/colqwen-test"
        assert loaded["doc_ids_to_file_names"][1] == "a.pdf"
        assert loaded["doc_id_to_metadata"][1]["title"] == "A"

    def test_put_overwrites(self, client):
        client.put(
            "/v1/collection/bk_ow/bookkeeping",
            content=_dumps(self._blob()),
            headers={"Content-Type": "application/octet-stream"},
        )
        updated = self._blob()
        updated["index_config"]["description"] = "v2"
        client.put(
            "/v1/collection/bk_ow/bookkeeping",
            content=_dumps(updated),
            headers={"Content-Type": "application/octet-stream"},
        )
        r = client.get("/v1/collection/bk_ow/bookkeeping")
        assert _loads(r.content)["index_config"]["description"] == "v2"
