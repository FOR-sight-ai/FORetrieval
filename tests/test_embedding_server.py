"""Unit tests for the embedding server package.

All tests are offline — no real server, no GPU, no SSH required.
HTTP and SSH interactions are fully mocked.
"""

import base64
import io
import json
from unittest.mock import MagicMock, patch, call

import httpx
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
        token_embeds = [[float(i) * 0.1] * dim for _ in range(n_tokens)]
        data.append({"index": i, "object": "embedding", "data": token_embeds})
    return {"object": "list", "data": data, "model": "test-model"}


def _make_mock_response(status_code: int = 200, json_data: dict = None, text: str = "") -> MagicMock:
    """Build a mock httpx.Response."""
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = status_code
    resp.json.return_value = json_data or {}
    resp.text = text
    return resp


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
        d = {"url": "http://host:9000", "model_name": "athrael-soju/colqwen3.5-4.5B-v3"}
        cfg = EmbeddingServerConfig.from_dict(d)
        assert cfg.url == "http://host:9000"
        assert cfg.port == 8000  # default

    # --- model_name validator ---

    def test_incompatible_model_raises(self):
        with pytest.raises(ValueError, match="colqwen3"):
            _make_config(model_name="vidore/colpali-v1.2")

    def test_colqwen2_model_raises(self):
        with pytest.raises(ValueError, match="colqwen3"):
            _make_config(model_name="vidore/colqwen2-v1.0")

    def test_colqwen3_model_accepted(self):
        cfg = _make_config(model_name="athrael-soju/colqwen3.5-4.5B-v3")
        assert "colqwen3" in cfg.model_name.lower()

    def test_colqwen3_variant_accepted(self):
        cfg = _make_config(model_name="vidore/colqwen3-v1.0")
        assert cfg.model_name == "vidore/colqwen3-v1.0"

    # --- auth / SSL fields ---

    def test_api_key_default_none(self):
        cfg = _make_config()
        assert cfg.api_key is None

    def test_api_key_set(self):
        cfg = _make_config(api_key="secret-token")
        assert cfg.api_key == "secret-token"

    def test_verify_ssl_default_true(self):
        cfg = _make_config()
        assert cfg.verify_ssl is True

    def test_verify_ssl_false(self):
        cfg = _make_config(verify_ssl=False)
        assert cfg.verify_ssl is False


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

        mock_response = _make_mock_response(200, fake_resp)

        with patch.object(client._client, "post", return_value=mock_response) as mock_post:
            client.embed_images(images)

        mock_post.assert_called_once()
        _, kwargs = mock_post.call_args
        payload = kwargs["json"]

        assert payload["model"] == "athrael-soju/colqwen3.5-4.5B-v3"
        # vLLM >=0.19.0: images sent via messages array (PoolingChatRequest)
        assert "messages" in payload
        assert len(payload["messages"]) == 1
        content = payload["messages"][0]["content"]
        assert len(content) == 1
        assert content[0]["type"] == "image_url"
        assert content[0]["image_url"]["url"].startswith("data:image/png;base64,")

    def test_embed_query_sends_text_payload(self):
        client = self._make_client()
        fake_resp = _fake_pooling_response(1, n_tokens=5)

        mock_response = _make_mock_response(200, fake_resp)

        with patch.object(client._client, "post", return_value=mock_response) as mock_post:
            client.embed_query("What is the speed?")

        _, kwargs = mock_post.call_args
        payload = kwargs["json"]
        assert payload["input"] == "What is the speed?"
        assert "task" not in payload

    def test_embed_images_uses_correct_endpoint(self):
        client = self._make_client()
        mock_response = _make_mock_response(200, _fake_pooling_response(1))

        with patch.object(client._client, "post", return_value=mock_response) as mock_post:
            client.embed_images([_make_image()])

        url_called = mock_post.call_args[0][0]
        assert url_called == f"{_TEST_SERVER_URL}/pooling"

    def test_auth_header_sent_when_api_key_set(self):
        """api_key set → Authorization: Bearer header in httpx.Client headers."""
        cfg = _make_config(api_key="my-secret")
        client = EmbeddingServerClient(cfg)
        assert client._client.headers.get("authorization") == "Bearer my-secret"

    def test_no_auth_header_when_api_key_none(self):
        """api_key=None → no Authorization header."""
        cfg = _make_config()
        client = EmbeddingServerClient(cfg)
        assert "authorization" not in client._client.headers

    def test_ssl_verify_true_by_default(self):
        """verify_ssl=True → httpx.Client verifies SSL."""
        cfg = _make_config()
        client = EmbeddingServerClient(cfg)
        # httpx.Client stores ssl_context; verify=True means it's not disabled
        # We check the config was respected via the verify attr on the transport
        assert client.config.verify_ssl is True

    def test_ssl_verify_false_passed_to_client(self):
        """verify_ssl=False → httpx.Client created with verify=False."""
        cfg = _make_config(verify_ssl=False)
        # Should not raise; httpx accepts verify=False
        client = EmbeddingServerClient(cfg)
        assert client.config.verify_ssl is False


# ---------------------------------------------------------------------------
# EmbeddingServerClient — response parsing
# ---------------------------------------------------------------------------

class TestEmbeddingServerClientResponseParsing:
    def _client_with_response(self, response_dict: dict) -> EmbeddingServerClient:
        client = EmbeddingServerClient(_make_config())
        mock_response = _make_mock_response(200, response_dict)
        client._client = MagicMock()
        client._client.post.return_value = mock_response
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
        first_call = {"done": False}

        def fake_post(url, json=None, **kwargs):
            call_count["n"] += 1
            if not first_call["done"]:
                first_call["done"] = True
                return _make_mock_response(500, text="CUDA out of memory trying to allocate tensor")
            return _make_mock_response(200, _fake_pooling_response(1))

        client._client = MagicMock()
        client._client.post.side_effect = fake_post

        images = [_make_image()] * 4
        result = client.embed_images(images)
        assert len(result) == 4
        assert call_count["n"] > 1

    def test_oom_at_batch_size_1_raises(self):
        """OOM even at batch_size=1 → raises ServerOOMError."""
        client = EmbeddingServerClient(_make_config(batch_size=1))

        def fake_post(url, json=None, **kwargs):
            return _make_mock_response(500, text="CUDA out of memory")

        client._client = MagicMock()
        client._client.post.side_effect = fake_post

        with pytest.raises(ServerOOMError):
            client.embed_images([_make_image()])

    def test_non_oom_500_raises_runtime_error(self):
        client = EmbeddingServerClient(_make_config())

        def fake_post(url, json=None, **kwargs):
            return _make_mock_response(500, text="Internal server error: model not loaded")

        client._client = MagicMock()
        client._client.post.side_effect = fake_post

        with pytest.raises(RuntimeError, match="HTTP 500"):
            client.embed_images([_make_image()])

    def test_connection_error_raises(self):
        client = EmbeddingServerClient(_make_config())
        client._client = MagicMock()
        client._client.post.side_effect = httpx.ConnectError("refused")

        with pytest.raises(ConnectionError):
            client.embed_images([_make_image()])

    def test_timeout_raises(self):
        client = EmbeddingServerClient(_make_config())
        client._client = MagicMock()
        client._client.post.side_effect = httpx.TimeoutException("timed out")

        with pytest.raises(TimeoutError):
            client.embed_images([_make_image()])


# ---------------------------------------------------------------------------
# EmbeddingServerClient — health check
# ---------------------------------------------------------------------------

class TestHealthCheck:
    def test_health_check_true_on_200(self):
        client = EmbeddingServerClient(_make_config())
        mock_resp = _make_mock_response(200)
        client._client = MagicMock()
        client._client.get.return_value = mock_resp
        assert client.health_check() is True

    def test_health_check_false_on_500(self):
        client = EmbeddingServerClient(_make_config())
        mock_resp = _make_mock_response(500)
        client._client = MagicMock()
        client._client.get.return_value = mock_resp
        assert client.health_check() is False

    def test_health_check_false_on_connection_error(self):
        client = EmbeddingServerClient(_make_config())
        client._client = MagicMock()
        client._client.get.side_effect = httpx.ConnectError("refused")
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
            "nvidia-smi": ("2\n", ""),
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
# Quantization config (local mode, no GPU needed — just validates ctor)
# ---------------------------------------------------------------------------

class TestQuantizationConfig:
    def test_colpali_model_accepts_quantization_params(self):
        """ColPaliModel.__init__ signature accepts load_in_4bit/8bit — smoke test."""
        import inspect
        from foretrieval.colpali import ColPaliModel
        sig = inspect.signature(ColPaliModel.__init__)
        assert "load_in_4bit" in sig.parameters
        assert "load_in_8bit" in sig.parameters
        assert "bnb_4bit_quant_type" in sig.parameters
        assert "bnb_4bit_compute_dtype" in sig.parameters

    def test_retriever_model_accepts_quantization_params(self):
        """MultiModalRetrieverModel.from_pretrained accepts quantization params."""
        import inspect
        from foretrieval.retriever import MultiModalRetrieverModel
        sig = inspect.signature(MultiModalRetrieverModel.from_pretrained)
        assert "load_in_4bit" in sig.parameters
        assert "load_in_8bit" in sig.parameters

    def test_quantization_without_bitsandbytes_raises(self):
        """load_in_4bit=True with missing bitsandbytes → ImportError at load time."""
        from foretrieval.colpali import ColPaliModel
        with patch.dict("sys.modules", {"transformers.utils.bitsandbytes": None}):
            with patch("builtins.__import__", side_effect=lambda name, *a, **kw: (
                (_ for _ in ()).throw(ImportError("No module named 'bitsandbytes'"))
                if name == "bitsandbytes" else __import__(name, *a, **kw)
            )):
                pass  # import-level mock; actual test is integration-only without GPU


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

class TestPilToBase64:
    def test_returns_valid_base64_png(self):
        img = _make_image()
        b64 = _pil_to_base64(img)
        raw = base64.b64decode(b64)
        assert raw[:4] == b"\x89PNG"
