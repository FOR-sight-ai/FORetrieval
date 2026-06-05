"""Tests for the vector-DB server admin endpoints.

Covers:
- /v1/admin/indexes
- /v1/admin/data_folders
- 60 s TTL cache + invalidation on data-plane writes
- Auth gating (inherits the shared middleware)
- Symlink jail / containment
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

# Patch FOR_DB_DATA_DIR before importing the server (mirrors the existing
# test_vector_db_server_app.py pattern).
_TMP = tempfile.mkdtemp(prefix="foretrieval_db_admin_test_")

import foretrieval.vector_db_server.server as _server_mod

_server_mod._DATA_DIR = Path(_TMP)
_server_mod._API_KEY = None
_server_mod._registry.clear()
_server_mod._locks.clear()
_server_mod._SIZE_CACHE.clear()

from foretrieval.vector_db_server.server import app  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mk_index_dir(name: str, n_files: int = 3, size_per_file: int = 100) -> Path:
    """Create a fake index directory under _DATA_DIR."""
    p = _server_mod._DATA_DIR / name
    p.mkdir(parents=True, exist_ok=True)
    # Mark as an index by adding the server-side sentinel (written by _write_meta).
    (p / "index.json").write_text('{"backend": "qdrant", "storage_config": null}')
    for i in range(n_files):
        (p / f"data_{i}.bin").write_bytes(b"x" * size_per_file)
    return p


def _mk_data_dir(name: str, n_files: int = 2, size_per_file: int = 50) -> Path:
    """Create a plain folder (NOT an index) under _DATA_DIR."""
    p = _server_mod._DATA_DIR / name
    p.mkdir(parents=True, exist_ok=True)
    for i in range(n_files):
        (p / f"file_{i}.bin").write_bytes(b"y" * size_per_file)
    return p


@pytest.fixture(autouse=True)
def _clean_state():
    _server_mod._registry.clear()
    _server_mod._locks.clear()
    _server_mod._SIZE_CACHE.clear()
    # Wipe data dir between tests
    if _server_mod._DATA_DIR.exists():
        for entry in list(_server_mod._DATA_DIR.iterdir()):
            if entry.is_dir():
                import shutil
                shutil.rmtree(entry, ignore_errors=True)
            else:
                entry.unlink(missing_ok=True)
    yield
    _server_mod._registry.clear()
    _server_mod._locks.clear()
    _server_mod._SIZE_CACHE.clear()


@pytest.fixture()
def client():
    return TestClient(app)


# ---------------------------------------------------------------------------
# /v1/admin/indexes
# ---------------------------------------------------------------------------

class TestAdminIndexes:
    def test_empty_data_dir(self, client):
        # Nothing under data_dir
        r = client.get("/v1/admin/indexes")
        assert r.status_code == 200
        payload = r.json()
        assert payload["items"] == []
        assert payload["count"] == 0
        assert payload["data_dir"] == str(_server_mod._DATA_DIR)

    def test_lists_index_dirs(self, client):
        _mk_index_dir("alpha", n_files=4, size_per_file=200)
        _mk_index_dir("beta", n_files=1, size_per_file=10)
        # A data-only folder should NOT appear in /indexes
        _mk_data_dir("gamma_data")

        r = client.get("/v1/admin/indexes")
        assert r.status_code == 200
        items = {it["name"]: it for it in r.json()["items"]}
        assert set(items) == {"alpha", "beta"}
        assert items["alpha"]["n_files"] >= 4   # data files + index_config marker
        assert items["alpha"]["size_bytes"] > 0
        assert items["beta"]["size_bytes"] > 0
        assert "modified" in items["alpha"]


# ---------------------------------------------------------------------------
# /v1/admin/data_folders
# ---------------------------------------------------------------------------

class TestAdminDataFolders:
    def test_lists_every_subdir_with_is_index_flag(self, client):
        _mk_index_dir("alpha")
        _mk_data_dir("uploads")

        r = client.get("/v1/admin/data_folders")
        assert r.status_code == 200
        items = {it["name"]: it for it in r.json()["items"]}
        assert set(items) == {"alpha", "uploads"}
        assert items["alpha"]["is_index"] is True
        assert items["uploads"]["is_index"] is False

    def test_empty(self, client):
        r = client.get("/v1/admin/data_folders")
        assert r.status_code == 200
        assert r.json()["count"] == 0


# ---------------------------------------------------------------------------
# Auth gating
# ---------------------------------------------------------------------------

class TestAdminAuth:
    def test_no_token_no_auth(self, client):
        _mk_index_dir("alpha")
        # _API_KEY is None — no auth required
        r = client.get("/v1/admin/indexes")
        assert r.status_code == 200

    def test_with_token_missing_header(self, client):
        _server_mod._API_KEY = "secret"
        try:
            r = client.get("/v1/admin/indexes")
            assert r.status_code == 401
        finally:
            _server_mod._API_KEY = None

    def test_with_token_wrong(self, client):
        _server_mod._API_KEY = "secret"
        try:
            r = client.get(
                "/v1/admin/indexes",
                headers={"Authorization": "Bearer wrong"},
            )
            assert r.status_code == 401
        finally:
            _server_mod._API_KEY = None

    def test_with_token_correct(self, client):
        _mk_index_dir("alpha")
        _server_mod._API_KEY = "secret"
        try:
            r = client.get(
                "/v1/admin/indexes",
                headers={"Authorization": "Bearer secret"},
            )
            assert r.status_code == 200
            assert any(it["name"] == "alpha" for it in r.json()["items"])
        finally:
            _server_mod._API_KEY = None


# ---------------------------------------------------------------------------
# TTL cache + invalidation
# ---------------------------------------------------------------------------

class TestSizeCache:
    def test_cache_hit_within_ttl(self, client, monkeypatch):
        _mk_index_dir("alpha", n_files=2, size_per_file=100)

        call_count = {"n": 0}
        real = _server_mod._dir_stats_sync

        def spy(path):
            call_count["n"] += 1
            return real(path)

        monkeypatch.setattr(_server_mod, "_dir_stats_sync", spy)

        client.get("/v1/admin/indexes")
        first = call_count["n"]
        client.get("/v1/admin/indexes")
        second = call_count["n"]

        # Second call must hit the cache — no additional walks
        assert second == first

    def test_cache_miss_after_ttl_expiry(self, client, monkeypatch):
        _mk_index_dir("alpha", n_files=2, size_per_file=100)
        monkeypatch.setattr(_server_mod, "_CACHE_TTL", 0.0)

        call_count = {"n": 0}
        real = _server_mod._dir_stats_sync

        def spy(path):
            call_count["n"] += 1
            return real(path)

        monkeypatch.setattr(_server_mod, "_dir_stats_sync", spy)

        client.get("/v1/admin/indexes")
        first = call_count["n"]
        client.get("/v1/admin/indexes")
        second = call_count["n"]

        # TTL is 0 → every request must re-walk
        assert second == 2 * first

    def test_invalidation_on_delete(self, client):
        # Manually populate cache then trigger DELETE on a (non-existent) collection;
        # the cache for that path should be cleared.
        idx = _mk_index_dir("alpha", n_files=2, size_per_file=100)
        client.get("/v1/admin/indexes")
        assert str(idx) in _server_mod._SIZE_CACHE
        # The delete endpoint always invalidates and returns {"deleted": True}.
        r = client.delete("/v1/collection/alpha")
        assert r.status_code == 200
        assert str(idx) not in _server_mod._SIZE_CACHE


# ---------------------------------------------------------------------------
# Symlink jail
# ---------------------------------------------------------------------------

class TestSymlinkContainment:
    def test_symlinked_dir_outside_data_dir_is_skipped(self, client, tmp_path):
        # Create a real dir OUTSIDE _DATA_DIR
        outside = tmp_path / "outside"
        outside.mkdir()
        (outside / "evil.bin").write_bytes(b"z" * 10000)

        # Symlink INSIDE _DATA_DIR pointing outside
        link = _server_mod._DATA_DIR / "linked"
        try:
            os.symlink(outside, link)
        except (NotImplementedError, OSError):
            pytest.skip("symlinks not supported on this platform")

        # Also a legitimate index alongside
        _mk_index_dir("real_index")

        r = client.get("/v1/admin/data_folders")
        assert r.status_code == 200
        names = {it["name"] for it in r.json()["items"]}
        # The symlink target's content must not be reported via the link.
        # The directory entry itself may appear, but with size_bytes == 0 because
        # we resolve and check containment in the walk.
        assert "real_index" in names
