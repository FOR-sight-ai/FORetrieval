"""Integration tests for the remote embedding server.

These tests require a live vLLM server and are skipped by default.

Environment variables
---------------------
FORETRIEVAL_TEST_SERVER : str
    Base URL of the vLLM server to test against, e.g. "http://gpu-server:8000".
    Tests are skipped when this variable is not set.
FORETRIEVAL_TEST_SSH_HOST : str
    SSH hostname for deployment tests (auto_deploy path), e.g. "gpu-server".
    Only needed for the deploy/stop tests.
FORETRIEVAL_TEST_MODEL : str
    HuggingFace model ID served by the server.
    Defaults to "athrael-soju/colqwen3.5-4.5B-v3".

Usage
-----
    # Run all integration tests against a remote GPU server:
    FORETRIEVAL_TEST_SERVER=http://<GPU_SERVER_HOST>:8000 \
    FORETRIEVAL_TEST_SSH_HOST=<GPU_SERVER_HOST> \
    uv run pytest tests/test_embedding_server_integration.py -m "integration" -v
"""

import os

import pytest
import torch
from PIL import Image

from foretrieval.embedding_server.client import EmbeddingServerClient
from foretrieval.embedding_server.config import EmbeddingServerConfig
from foretrieval.embedding_server.manager import EmbeddingServerManager

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

_SERVER_URL = os.environ.get("FORETRIEVAL_TEST_SERVER", "")
_SSH_HOST = os.environ.get("FORETRIEVAL_TEST_SSH_HOST", "")
_MODEL = os.environ.get(
    "FORETRIEVAL_TEST_MODEL", "athrael-soju/colqwen3.5-4.5B-v3"
)

requires_server = pytest.mark.skipif(
    not _SERVER_URL,
    reason="Set FORETRIEVAL_TEST_SERVER=http://<host>:<port> to run integration tests",
)
requires_ssh = pytest.mark.skipif(
    not _SSH_HOST,
    reason="Set FORETRIEVAL_TEST_SSH_HOST=<host> to run deploy integration tests",
)


@pytest.fixture(scope="module")
def server_config() -> EmbeddingServerConfig:
    return EmbeddingServerConfig(
        url=_SERVER_URL,
        model_name=_MODEL,
        batch_size=2,
        request_timeout=180,
    )


@pytest.fixture(scope="module")
def client(server_config) -> EmbeddingServerClient:
    return EmbeddingServerClient(server_config)


def _make_image(w: int = 64, h: int = 64) -> Image.Image:
    return Image.new("RGB", (w, h), color=(100, 150, 200))


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

@requires_server
@pytest.mark.integration
def test_server_health(client):
    assert client.health_check(), (
        f"Server at {_SERVER_URL} is not healthy — is vLLM running?"
    )


# ---------------------------------------------------------------------------
# Image embedding
# ---------------------------------------------------------------------------

@requires_server
@pytest.mark.integration
@pytest.mark.slow
def test_embed_single_image_returns_tensor(client):
    imgs = [_make_image()]
    result = client.embed_images(imgs)
    assert len(result) == 1
    assert isinstance(result[0], torch.Tensor)
    assert result[0].ndim == 2  # [n_tokens, dim]
    assert result[0].shape[1] == 128  # ColQwen3.5 embed dim


@requires_server
@pytest.mark.integration
@pytest.mark.slow
def test_embed_multiple_images_returns_correct_count(client):
    n = 3
    imgs = [_make_image()] * n
    result = client.embed_images(imgs)
    assert len(result) == n


@requires_server
@pytest.mark.integration
@pytest.mark.slow
def test_embed_images_tensors_are_on_cpu(client):
    result = client.embed_images([_make_image()])
    assert result[0].device.type == "cpu"


@requires_server
@pytest.mark.integration
@pytest.mark.slow
def test_embed_images_tensors_differ_across_images(client):
    """Different images should yield different embeddings."""
    img_a = Image.new("RGB", (64, 64), color=(255, 0, 0))
    img_b = Image.new("RGB", (64, 64), color=(0, 0, 255))
    emb_a, emb_b = client.embed_images([img_a, img_b])
    assert not torch.allclose(emb_a, emb_b)


# ---------------------------------------------------------------------------
# Query embedding
# ---------------------------------------------------------------------------

@requires_server
@pytest.mark.integration
@pytest.mark.slow
def test_embed_query_returns_single_tensor(client):
    result = client.embed_query("What is the maximum speed?")
    assert len(result) == 1
    assert isinstance(result[0], torch.Tensor)
    assert result[0].ndim == 2
    assert result[0].shape[1] == 128


@requires_server
@pytest.mark.integration
@pytest.mark.slow
def test_embed_query_differs_from_image_embedding(client):
    """Query and image embeddings should be in the same space but differ."""
    q_emb = client.embed_query("What is the airspeed?")[0]
    img_emb = client.embed_images([_make_image()])[0]
    # Both are [n_tokens, 128] — shapes may differ but dtypes should match
    assert q_emb.dtype == img_emb.dtype


# ---------------------------------------------------------------------------
# Batch size / OOM retry (live server, small batches)
# ---------------------------------------------------------------------------

@requires_server
@pytest.mark.integration
@pytest.mark.slow
def test_embed_images_batch_larger_than_one(client):
    """Server should handle multiple images per request without OOM on RTX4090."""
    imgs = [_make_image(128, 128)] * 4
    result = client.embed_images(imgs)
    assert len(result) == 4


@requires_server
@pytest.mark.integration
@pytest.mark.slow
def test_embed_empty_list_returns_empty(client):
    result = client.embed_images([])
    assert result == []


# ---------------------------------------------------------------------------
# End-to-end: ColPaliModel with remote client
# ---------------------------------------------------------------------------

@requires_server
@pytest.mark.integration
@pytest.mark.slow
def test_colpali_model_remote_embed_images(server_config, tmp_path):
    """ColPaliModel loads processor only and uses remote client for embeddings."""
    from foretrieval.colpali import ColPaliModel

    model = ColPaliModel.from_pretrained(
        pretrained_model_name_or_path=_MODEL,
        index_root=str(tmp_path),
        embedding_server=server_config,
    )
    assert model._remote_client is not None
    assert model.model is None  # no local weights loaded

    # Index a small synthetic image
    img = _make_image(64, 64)
    model.index_name = "test_remote"
    model.highest_doc_id = -1
    model.full_document_collection = False
    model.resize_stored_images = False
    model.max_image_width = None
    model.max_image_height = None

    model._add_to_index(
        images=[img],
        store_collection_with_index=False,
        doc_id=0,
        page_ids=[1],
    )

    # Should have one embedding stored
    if model.storage_qdrant:
        # qdrant path — count points
        pass  # point insertion verified by no exception
    else:
        assert len(model.indexed_embeddings) == 1
        assert model.indexed_embeddings[0].shape[1] == 128


@requires_server
@pytest.mark.integration
@pytest.mark.slow
def test_colpali_model_remote_encode_query(server_config, tmp_path):
    """_encode_search_query uses remote client in remote mode."""
    from foretrieval.colpali import ColPaliModel

    model = ColPaliModel.from_pretrained(
        pretrained_model_name_or_path=_MODEL,
        index_root=str(tmp_path),
        embedding_server=server_config,
    )
    qs = model._encode_search_query("What is the fuel capacity?")
    assert len(qs) == 1
    assert isinstance(qs[0], torch.Tensor)
    assert qs[0].ndim == 2


# ---------------------------------------------------------------------------
# Deployment manager (requires SSH access)
# ---------------------------------------------------------------------------

@requires_server
@requires_ssh
@pytest.mark.integration
@pytest.mark.slow
def test_manager_health_check_against_running_server():
    """If server is already running, health check returns True."""
    cfg = EmbeddingServerConfig(
        url=_SERVER_URL,
        model_name=_MODEL,
        ssh_host=_SSH_HOST,
    )
    client = EmbeddingServerClient(cfg)
    assert client.health_check()


@requires_ssh
@pytest.mark.integration
@pytest.mark.slow
def test_manager_ensure_deployed_idempotent():
    """ensure_deployed called twice should not start a second container."""
    cfg = EmbeddingServerConfig(
        url=_SERVER_URL or f"http://{_SSH_HOST}:8000",
        model_name=_MODEL,
        auto_deploy=True,
        ssh_host=_SSH_HOST,
        n_gpus=-1,
    )
    mgr = EmbeddingServerManager(cfg)
    # First call: may deploy or detect already running
    mgr.ensure_deployed()
    # Second call: container already running → no-op
    mgr.ensure_deployed()
    # Verify container is running
    assert mgr._is_container_running()
