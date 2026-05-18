"""Tests for VectorDBServerConfig — pydantic validators and helpers."""

import pytest
from pydantic import ValidationError

from foretrieval.vector_db_server.config import VectorDBServerConfig


def _make_cfg(**kwargs) -> VectorDBServerConfig:
    defaults = {"url": "http://localhost:18000", "backend": "qdrant"}
    return VectorDBServerConfig(**{**defaults, **kwargs})


class TestURLValidation:
    def test_trailing_slash_stripped(self):
        cfg = _make_cfg(url="http://db-host:18000/")
        assert cfg.url == "http://db-host:18000"

    def test_multiple_trailing_slashes_stripped(self):
        cfg = _make_cfg(url="http://db-host:18000///")
        assert cfg.url == "http://db-host:18000"

    def test_url_preserved_without_trailing_slash(self):
        cfg = _make_cfg(url="http://db-host:18000")
        assert cfg.url == "http://db-host:18000"


class TestBackendValidation:
    def test_valid_backends(self):
        for b in ("local", "qdrant", "milvus"):
            cfg = _make_cfg(backend=b)
            assert cfg.backend == b

    def test_backend_normalised_to_lower(self):
        cfg = _make_cfg(backend="Qdrant")
        assert cfg.backend == "qdrant"

    def test_invalid_backend_raises(self):
        with pytest.raises(ValidationError, match="not a valid server-side backend"):
            _make_cfg(backend="redis")


class TestPortValidation:
    def test_default_port(self):
        cfg = _make_cfg()
        assert cfg.port == 18000

    def test_custom_port(self):
        cfg = _make_cfg(port=9999)
        assert cfg.port == 9999

    def test_zero_port_raises(self):
        with pytest.raises(ValidationError):
            _make_cfg(port=0)

    def test_out_of_range_port_raises(self):
        with pytest.raises(ValidationError):
            _make_cfg(port=99999)


class TestAutoDeployValidation:
    def test_auto_deploy_requires_ssh_host(self):
        with pytest.raises(ValidationError, match="ssh_host is required"):
            _make_cfg(auto_deploy=True, ssh_host=None)

    def test_auto_deploy_with_ssh_host_ok(self):
        cfg = _make_cfg(auto_deploy=True, ssh_host="gpu-server")
        assert cfg.auto_deploy is True
        assert cfg.ssh_host == "gpu-server"

    def test_auto_deploy_false_no_ssh_needed(self):
        cfg = _make_cfg(auto_deploy=False)
        assert cfg.auto_deploy is False


class TestDefaults:
    def test_api_key_default_none(self):
        cfg = _make_cfg()
        assert cfg.api_key is None

    def test_verify_ssl_default_true(self):
        cfg = _make_cfg()
        assert cfg.verify_ssl is True

    def test_request_timeout_default(self):
        cfg = _make_cfg()
        assert cfg.request_timeout == 120

    def test_data_dir_default(self):
        cfg = _make_cfg()
        assert cfg.data_dir == "/var/lib/foretrieval_db"


class TestFromDict:
    def test_from_dict_basic(self):
        cfg = VectorDBServerConfig.from_dict(
            {"url": "http://db-server:18000", "backend": "qdrant", "api_key": "tok"}
        )
        assert cfg.url == "http://db-server:18000"
        assert cfg.api_key == "tok"

    def test_from_dict_missing_url_raises(self):
        with pytest.raises((ValidationError, TypeError)):
            VectorDBServerConfig.from_dict({"backend": "qdrant"})
