"""Tests for VectorDBServerClient — HTTP interactions mocked via MagicMock."""

import io
from unittest.mock import MagicMock, patch
import pytest
import torch

from foretrieval.vector_db_server.client import VectorDBServerClient, _dumps, _loads
from foretrieval.vector_db_server.config import VectorDBServerConfig
from foretrieval.vector_store.base import MultiVectorQuery, StoredPoint, make_point_id


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(**kwargs) -> VectorDBServerConfig:
    defaults = {"url": "http://localhost:18000", "backend": "qdrant"}
    return VectorDBServerConfig(**{**defaults, **kwargs})


def _make_client(**cfg_kwargs) -> VectorDBServerClient:
    return VectorDBServerClient(_make_config(**cfg_kwargs))


def _make_mock_json_response(status_code: int, data: dict):
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = data
    resp.text = str(data)
    return resp


def _make_mock_bytes_response(status_code: int, content: bytes):
    resp = MagicMock()
    resp.status_code = status_code
    resp.content = content
    resp.text = "<binary>"
    resp.json.side_effect = Exception("not json")
    return resp


def _make_point(doc_id: int = 1, page_id: int = 0, dim: int = 4) -> StoredPoint:
    pid = make_point_id(doc_id, page_id)
    return StoredPoint(
        point_id=pid,
        vector=torch.randn(3, dim),
        payload={"doc_id": doc_id, "page_id": page_id},
    )


# ---------------------------------------------------------------------------
# Codec
# ---------------------------------------------------------------------------

class TestCodec:
    def test_roundtrip_tensor(self):
        t = torch.randn(5, 8)
        assert torch.allclose(t, _loads(_dumps(t)))

    def test_roundtrip_list_of_dicts(self):
        data = [{"point_id": 1, "vector": torch.randn(3, 4), "payload": {"a": 1}}]
        result = _loads(_dumps(data))
        assert result[0]["point_id"] == 1
        assert torch.allclose(result[0]["vector"], data[0]["vector"])


# ---------------------------------------------------------------------------
# Auth header
# ---------------------------------------------------------------------------

class TestAuthHeader:
    def test_api_key_added_to_headers(self):
        client = _make_client(api_key="mysecret")
        assert "Authorization" in client._client.headers
        assert client._client.headers["Authorization"] == "Bearer mysecret"

    def test_no_api_key_no_auth_header(self):
        client = _make_client()
        assert "Authorization" not in client._client.headers


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

class TestHealthCheck:
    def test_health_ok(self):
        client = _make_client()
        client._client.get = MagicMock(
            return_value=_make_mock_json_response(200, {"status": "ok"})
        )
        assert client.health_check() is True

    def test_health_non_200(self):
        client = _make_client()
        client._client.get = MagicMock(
            return_value=_make_mock_json_response(503, {})
        )
        assert client.health_check() is False

    def test_health_connection_error(self):
        import httpx
        client = _make_client()
        client._client.get = MagicMock(side_effect=httpx.ConnectError("refused"))
        assert client.health_check() is False


# ---------------------------------------------------------------------------
# collection_exists
# ---------------------------------------------------------------------------

class TestCollectionExists:
    def test_exists_true(self):
        client = _make_client()
        client._client.get = MagicMock(
            return_value=_make_mock_json_response(200, {"exists": True, "backend": "qdrant"})
        )
        assert client.collection_exists("my_index") is True

    def test_exists_false(self):
        client = _make_client()
        client._client.get = MagicMock(
            return_value=_make_mock_json_response(200, {"exists": False, "backend": None})
        )
        assert client.collection_exists("my_index") is False


# ---------------------------------------------------------------------------
# open_collection
# ---------------------------------------------------------------------------

class TestOpenCollection:
    def test_open_posts_correct_body(self):
        client = _make_client()
        expected = {"opened": True, "backend": "qdrant", "created": True}
        client._client.post = MagicMock(
            return_value=_make_mock_json_response(200, expected)
        )
        result = client.open_collection(
            "idx", "qdrant", create=True, dim=128, storage_config=None
        )
        assert result == expected
        call_args = client._client.post.call_args
        assert "/v1/collection/open" in call_args.args[0]
        body = call_args.kwargs["json"]
        assert body["index_name"] == "idx"
        assert body["create"] is True
        assert body["dim"] == 128


# ---------------------------------------------------------------------------
# upsert
# ---------------------------------------------------------------------------

class TestUpsert:
    def test_upsert_sends_octet_stream(self):
        client = _make_client()
        resp = _make_mock_json_response(200, {"upserted": 1})
        client._client.post = MagicMock(return_value=resp)

        point = _make_point()
        client.upsert("idx", [point])

        call_args = client._client.post.call_args
        assert "upsert/idx" in call_args.args[0]
        assert call_args.kwargs["headers"]["Content-Type"] == "application/octet-stream"
        # Body is binary — decode and verify it contains the tensor
        body_bytes = call_args.kwargs["content"]
        decoded = _loads(body_bytes)
        assert decoded[0]["point_id"] == point.point_id

    def test_upsert_timeout_raises(self):
        import httpx
        client = _make_client()
        client._client.post = MagicMock(side_effect=httpx.TimeoutException("timeout"))
        with pytest.raises(TimeoutError):
            client.upsert("idx", [_make_point()])

    def test_upsert_connection_error_raises(self):
        import httpx
        client = _make_client()
        client._client.post = MagicMock(side_effect=httpx.ConnectError("refused"))
        with pytest.raises(ConnectionError):
            client.upsert("idx", [_make_point()])


# ---------------------------------------------------------------------------
# point_exists
# ---------------------------------------------------------------------------

class TestPointExists:
    def test_point_exists_true(self):
        client = _make_client()
        client._client.get = MagicMock(
            return_value=_make_mock_json_response(200, {"exists": True})
        )
        assert client.point_exists("idx", 42) is True

    def test_point_exists_false(self):
        client = _make_client()
        client._client.get = MagicMock(
            return_value=_make_mock_json_response(200, {"exists": False})
        )
        assert client.point_exists("idx", 42) is False


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------

class TestSearch:
    def test_search_sends_query_and_parses_hits(self):
        client = _make_client()
        hits_wire = [
            {"point_id": 10000, "score": 9.5, "payload": {"doc_id": 1, "page_id": 0}}
        ]
        resp = _make_mock_bytes_response(200, _dumps(hits_wire))
        client._client.post = MagicMock(return_value=resp)

        query = MultiVectorQuery(vectors=torch.randn(5, 4))
        hits = client.search("idx", query, k=3)

        assert len(hits) == 1
        assert hits[0].point_id == 10000
        assert hits[0].score == pytest.approx(9.5)

        # Verify request body contains k and vectors
        body_bytes = client._client.post.call_args.kwargs["content"]
        wire = _loads(body_bytes)
        assert wire["k"] == 3
        assert wire["filter_metadata"] is None


# ---------------------------------------------------------------------------
# fetch_vector
# ---------------------------------------------------------------------------

class TestFetchVector:
    def test_fetch_returns_tensor(self):
        client = _make_client()
        tensor = torch.randn(3, 4)
        resp = _make_mock_bytes_response(200, _dumps(tensor))
        client._client.get = MagicMock(return_value=resp)

        result = client.fetch_vector("idx", 12345)
        assert result is not None
        assert torch.allclose(result, tensor)

    def test_fetch_returns_none_on_404(self):
        client = _make_client()
        resp = MagicMock()
        resp.status_code = 404
        client._client.get = MagicMock(return_value=resp)

        result = client.fetch_vector("idx", 99999)
        assert result is None


# ---------------------------------------------------------------------------
# Bookkeeping
# ---------------------------------------------------------------------------

class TestBookkeeping:
    def test_put_bookkeeping_sends_octet_stream(self):
        client = _make_client()
        resp = _make_mock_json_response(200, {"stored": True})
        client._client.put = MagicMock(return_value=resp)

        blob = {"index_config": {"model_name": "m"}, "doc_id_to_metadata": {1: {"t": "x"}}}
        client.put_bookkeeping("idx", blob)

        call = client._client.put.call_args
        assert call.args[0].endswith("/v1/collection/idx/bookkeeping")
        body = call.kwargs["content"]
        assert _loads(body)["index_config"]["model_name"] == "m"

    def test_get_bookkeeping_returns_blob(self):
        client = _make_client()
        blob = {"index_config": {"model_name": "m"}}
        resp = _make_mock_bytes_response(200, _dumps(blob))
        client._client.get = MagicMock(return_value=resp)

        out = client.get_bookkeeping("idx")
        assert out["index_config"]["model_name"] == "m"

    def test_get_bookkeeping_returns_none_on_404(self):
        client = _make_client()
        resp = MagicMock()
        resp.status_code = 404
        client._client.get = MagicMock(return_value=resp)

        assert client.get_bookkeeping("idx") is None
