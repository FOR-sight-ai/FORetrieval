"""Unit tests for the embedding server package.

All tests are offline — no real server, no GPU, no SSH required.
HTTP and SSH interactions are fully mocked.
"""

import base64
import io
import json
from unittest.mock import MagicMock, patch, call

import pytest
import torch
from PIL import Image

from foretrieval.embedding_server.config import EmbeddingServerConfig
from foretrieval.embedding_server.client import (
    EmbeddingServerClient,
    ServerOOMError,
    _pil_to_base64,
)
from foretrieval.embedding_server.manager import EmbeddingServerManager


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_TEST_SERVER_URL = "http://localhost:8000"
_TEST_MODEL = "athrael-soju/colqwen3.5-4.5B-v3"
_TEST_SSH_HOST = "test-gpu-server"


def _make_config(**kwargs) -> EmbeddingServerConfig:
    defaults = dict(
        url=_TEST_SERVER_URL,
        model_name=_TEST_MODEL,
        batch_size=4,
    )
    defaults.update(kwargs)
    return EmbeddingServerConfig(**defaults)


def _make_image(w=32, h=32) -> Image.Image:
    return Image.new("RGB", (w, h), color=(128, 64, 32))


def _fake_pooling_response(n_items: int, n_tokens: int = 10, dim: int = 128) -> dict:
    """Build a fake vLLM /pooling response with n_items token-embed outputs."""
    data = []
    for i in range(n_items):
        # shape [n_tokens, dim] as nested list
        token_embeds = [[float(i) * 0.1] * dim for _ in range(n_tokens)]
        data.append({"index": i, "object": "embedding", "data": token_embeds})
    return {"object": "list", "data": data, "model": "test-model"}


# ---------------------------------------------------------------------------
# EmbeddingServerConfig tests
# ---------------------------------------------------------------------------

class TestEmbeddingServerConfig:
    def test_basic_construction(self):
        cfg = _make_config()
        assert cfg.url == _TEST_SERVER_URL
        assert cfg.model_name == _TEST_MODEL
        assert cfg.n_gpus == -1
        assert cfg.batch_size == 4

    def test_trailing_slash_stripped(self):
        cfg = _make_config(url="http://localhost:8000/")
        assert cfg.url == "http://localhost:8000"

    def test_n_gpus_zero_invalid(self):
        with pytest.raises(Exception):
            _make_config(n_gpus=0)

    def test_n_gpus_negative_minus_one_valid(self):
        cfg = _make_config(n_gpus=-1)
        assert cfg.n_gpus == -1

    def test_n_gpus_positive_valid(self):
        cfg = _make_config(n_gpus=2)
        assert cfg.n_gpus == 2

    def test_auto_deploy_without_ssh_host_raises(self):
        with pytest.raises(Exception, match="ssh_host"):
            _make_config(auto_deploy=True, ssh_host=None)

    def test_auto_deploy_with_ssh_host_ok(self):
        cfg = _make_config(auto_deploy=True, ssh_host=_TEST_SSH_HOST)
        assert cfg.auto_deploy is True

    def test_from_dict(self):
        d = {"url": "http://host:9000", "model_name": "some/model"}
        cfg = EmbeddingServerConfig.from_dict(d)
        assert cfg.url == "http://host:9000"
        assert cfg.port == 8000  # default


# ---------------------------------------------------------------------------
# EmbeddingServerClient — request format
# ---------------------------------------------------------------------------

class TestEmbeddingServerClientRequestFormat:
    def _make_client(self, **kwargs) -> EmbeddingServerClient:
        return EmbeddingServerClient(_make_config(**kwargs))

    def test_embed_images_sends_correct_payload(self):
        client = self._make_client()
        images = [_make_image()]
        fake_resp = _fake_pooling_response(1)

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = fake_resp

        with patch.object(client._session, "post", return_value=mock_response) as mock_post:
            client.embed_images(images)

        mock_post.assert_called_once()
        _, kwargs = mock_post.call_args
        payload = kwargs["json"]

        assert payload["model"] == "athrael-soju/colqwen3.5-4.5B-v3"
        assert payload["task"] == "token_embed"
        assert len(payload["input"]) == 1
        item = payload["input"][0]
        assert item["type"] == "image_url"
        assert item["image_url"]["url"].startswith("data:image/png;base64,")

    def test_embed_query_sends_text_payload(self):
        client = self._make_client()
        fake_resp = _fake_pooling_response(1, n_tokens=5)

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = fake_resp

        with patch.object(client._session, "post", return_value=mock_response) as mock_post:
            client.embed_query("What is the speed?")

        _, kwargs = mock_post.call_args
        payload = kwargs["json"]
        assert payload["input"] == "What is the speed?"
        assert payload["task"] == "token_embed"

    def test_embed_images_uses_correct_endpoint(self):
        client = self._make_client()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = _fake_pooling_response(1)

        with patch.object(client._session, "post", return_value=mock_response) as mock_post:
            client.embed_images([_make_image()])

        url_called = mock_post.call_args[0][0]
        assert url_called == f"{_TEST_SERVER_URL}/pooling"

    def test_timeout_passed_to_requests(self):
        client = self._make_client(request_timeout=60)
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = _fake_pooling_response(1)

        with patch.object(client._session, "post", return_value=mock_response) as mock_post:
            client.embed_images([_make_image()])

        _, kwargs = mock_post.call_args
        assert kwargs["timeout"] == 60


# ---------------------------------------------------------------------------
# EmbeddingServerClient — response parsing
# ---------------------------------------------------------------------------

class TestEmbeddingServerClientResponseParsing:
    def _client_with_response(self, response_dict: dict) -> EmbeddingServerClient:
        client = EmbeddingServerClient(_make_config())
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = response_dict
        client._session = MagicMock()
        client._session.post.return_value = mock_response
        return client

    def test_embed_images_returns_correct_number_of_tensors(self):
        n = 3
        client = self._client_with_response(_fake_pooling_response(n))
        result = client.embed_images([_make_image()] * n)
        assert len(result) == n

    def test_embed_images_tensor_shape(self):
        n_tokens, dim = 12, 128
        client = self._client_with_response(_fake_pooling_response(1, n_tokens, dim))
        result = client.embed_images([_make_image()])
        assert result[0].shape == (n_tokens, dim)

    def test_embed_images_tensor_dtype_float32(self):
        client = self._client_with_response(_fake_pooling_response(1))
        result = client.embed_images([_make_image()])
        assert result[0].dtype == torch.float32

    def test_embed_query_returns_single_tensor(self):
        client = self._client_with_response(_fake_pooling_response(1, n_tokens=7, dim=128))
        result = client.embed_query("hello")
        assert len(result) == 1
        assert result[0].shape == (7, 128)

    def test_embed_empty_images_returns_empty(self):
        client = EmbeddingServerClient(_make_config())
        result = client.embed_images([])
        assert result == []


# ---------------------------------------------------------------------------
# EmbeddingServerClient — OOM retry
# ---------------------------------------------------------------------------

class TestEmbeddingServerClientOOMRetry:
    def test_oom_halves_batch_size_and_retries(self):
        """Server OOM on batch_size=4 → retries with 2 → succeeds."""
        client = EmbeddingServerClient(_make_config(batch_size=4))

        call_count = {"n": 0}

        def fake_post(url, json=None, timeout=None):
            call_count["n"] += 1
            n_items = len(json["input"]) if isinstance(json["input"], list) else 1
            if n_items > 2:
                resp = MagicMock()
                resp.status_code = 500
                resp.text = "CUDA out of memory trying to allocate tensor"
                return resp
            resp = MagicMock()
            resp.status_code = 200
            resp.json.return_value = _fake_pooling_response(n_items)
            return resp

        client._session = MagicMock()
        client._session.post.side_effect = fake_post

        images = [_make_image()] * 4
        result = client.embed_images(images)
        assert len(result) == 4
        # Should have called server multiple times (failed at 4, succeeded at batches of 2)
        assert call_count["n"] > 1

    def test_oom_at_batch_size_1_raises(self):
        """OOM even at batch_size=1 → raises ServerOOMError."""
        client = EmbeddingServerClient(_make_config(batch_size=1))

        def fake_post(url, json=None, timeout=None):
            resp = MagicMock()
            resp.status_code = 500
            resp.text = "CUDA out of memory"
            return resp

        client._session = MagicMock()
        client._session.post.side_effect = fake_post

        with pytest.raises(ServerOOMError):
            client.embed_images([_make_image()])

    def test_non_oom_500_raises_runtime_error(self):
        client = EmbeddingServerClient(_make_config())

        def fake_post(url, json=None, timeout=None):
            resp = MagicMock()
            resp.status_code = 500
            resp.text = "Internal server error: model not loaded"
            return resp

        client._session = MagicMock()
        client._session.post.side_effect = fake_post

        with pytest.raises(RuntimeError, match="HTTP 500"):
            client.embed_images([_make_image()])

    def test_connection_error_raises(self):
        import requests as req_lib
        client = EmbeddingServerClient(_make_config())
        client._session = MagicMock()
        client._session.post.side_effect = req_lib.ConnectionError("refused")

        with pytest.raises(ConnectionError):
            client.embed_images([_make_image()])

    def test_timeout_raises(self):
        import requests as req_lib
        client = EmbeddingServerClient(_make_config())
        client._session = MagicMock()
        client._session.post.side_effect = req_lib.Timeout("timed out")

        with pytest.raises(TimeoutError):
            client.embed_images([_make_image()])


# ---------------------------------------------------------------------------
# EmbeddingServerClient — health check
# ---------------------------------------------------------------------------

class TestHealthCheck:
    def test_health_check_true_on_200(self):
        client = EmbeddingServerClient(_make_config())
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        client._session = MagicMock()
        client._session.get.return_value = mock_resp
        assert client.health_check() is True

    def test_health_check_false_on_500(self):
        client = EmbeddingServerClient(_make_config())
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        client._session = MagicMock()
        client._session.get.return_value = mock_resp
        assert client.health_check() is False

    def test_health_check_false_on_connection_error(self):
        import requests as req_lib
        client = EmbeddingServerClient(_make_config())
        client._session = MagicMock()
        client._session.get.side_effect = req_lib.ConnectionError()
        assert client.health_check() is False


# ---------------------------------------------------------------------------
# EmbeddingServerManager — deploy logic (SSH mocked)
# ---------------------------------------------------------------------------

class TestEmbeddingServerManager:
    def _make_manager(self, **kwargs) -> EmbeddingServerManager:
        defaults = dict(auto_deploy=True, ssh_host=_TEST_SSH_HOST, n_gpus=-1)
        defaults.update(kwargs)
        cfg = _make_config(**defaults)
        return EmbeddingServerManager(cfg)

    def _mock_ssh(self, manager: EmbeddingServerManager, remote_outputs: dict):
        """Patch manager._run_remote to return values from remote_outputs dict.
        Keys are substrings to match in the command; fallback returns ("", "").
        """
        def fake_run(cmd):
            for key, val in remote_outputs.items():
                if key in cmd:
                    return val
            return ("", "")
        manager._run_remote = MagicMock(side_effect=fake_run)
        return manager

    def test_deploy_when_no_metadata(self):
        """No metadata file → deploy from scratch."""
        mgr = self._make_manager()
        self._mock_ssh(mgr, {
            "deployment.json": ("__MISSING__", ""),
            "nvidia-smi": ("2\n", ""),  # 2 GPUs
            "docker pull": ("", ""),
            "docker run": ("abc123\n", ""),
        })

        mgr.ensure_deployed()

        calls = [str(c) for c in mgr._run_remote.call_args_list]
        assert any("docker pull" in c for c in calls)
        assert any("docker run" in c for c in calls)

    def test_deploy_skipped_when_container_running(self):
        """Metadata present + container running → no redeploy."""
        mgr = self._make_manager()
        metadata = json.dumps({
            "model_name": "athrael-soju/colqwen3.5-4.5B-v3",
            "container_name": "foretrieval_embedding_server",
            "port": 8000,
        })
        self._mock_ssh(mgr, {
            "deployment.json": (metadata, ""),
            "docker inspect": ("true\n", ""),
        })

        mgr.ensure_deployed()

        calls = [str(c) for c in mgr._run_remote.call_args_list]
        assert not any("docker run" in c for c in calls)

    def test_redeploy_when_container_not_running(self):
        """Metadata present but container stopped → redeploy."""
        mgr = self._make_manager()
        metadata = json.dumps({
            "model_name": "athrael-soju/colqwen3.5-4.5B-v3",
            "container_name": "foretrieval_embedding_server",
            "port": 8000,
        })
        self._mock_ssh(mgr, {
            "deployment.json": (metadata, ""),
            "docker inspect": ("false\n", ""),
            "nvidia-smi": ("2\n", ""),
            "docker pull": ("", ""),
            "docker run": ("abc123\n", ""),
        })

        mgr.ensure_deployed()

        calls = [str(c) for c in mgr._run_remote.call_args_list]
        assert any("docker run" in c for c in calls)

    def test_n_gpus_minus1_uses_all_detected(self):
        """n_gpus=-1 → detect GPU count via nvidia-smi."""
        mgr = self._make_manager(n_gpus=-1)
        self._mock_ssh(mgr, {
            "deployment.json": ("__MISSING__", ""),
            "nvidia-smi": ("2\n", ""),
            "docker pull": ("", ""),
            "docker run": ("abc123\n", ""),
        })

        mgr.ensure_deployed()

        docker_run_calls = [
            str(c) for c in mgr._run_remote.call_args_list
            if "docker run" in str(c)
        ]
        assert docker_run_calls
        assert "tensor-parallel-size 2" in docker_run_calls[0]

    def test_n_gpus_explicit_used_directly(self):
        """n_gpus=1 → skip nvidia-smi, use 1 directly."""
        mgr = self._make_manager(n_gpus=1)
        self._mock_ssh(mgr, {
            "deployment.json": ("__MISSING__", ""),
            "docker pull": ("", ""),
            "docker run": ("abc123\n", ""),
        })

        mgr.ensure_deployed()

        docker_run_calls = [
            str(c) for c in mgr._run_remote.call_args_list
            if "docker run" in str(c)
        ]
        assert "tensor-parallel-size 1" in docker_run_calls[0]
        # nvidia-smi should NOT have been called
        nvidia_calls = [
            c for c in mgr._run_remote.call_args_list
            if "nvidia-smi" in str(c)
        ]
        assert not nvidia_calls

    def test_stop_removes_container_and_metadata(self):
        mgr = self._make_manager()
        mgr._run_remote = MagicMock(return_value=("", ""))
        mgr.stop()
        calls = [str(c) for c in mgr._run_remote.call_args_list]
        assert any("docker stop" in c for c in calls)
        assert any("docker rm" in c for c in calls)
        assert any("deployment.json" in c for c in calls)

    def test_missing_paramiko_raises_import_error(self):
        mgr = self._make_manager()
        with patch.dict("sys.modules", {"paramiko": None}):
            with pytest.raises(ImportError, match="paramiko"):
                mgr.ensure_deployed()


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

class TestPilToBase64:
    def test_returns_valid_base64_png(self):
        img = _make_image()
        b64 = _pil_to_base64(img)
        # Should decode without error
        raw = base64.b64decode(b64)
        # Should be a valid PNG (magic bytes)
        assert raw[:4] == b"\x89PNG"
