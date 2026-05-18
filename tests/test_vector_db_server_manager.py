"""Tests for VectorDBServerManager — SSH/Docker operations mocked via MagicMock."""

import json
from unittest.mock import MagicMock, patch
import pytest

from foretrieval.vector_db_server.config import VectorDBServerConfig
from foretrieval.vector_db_server.manager import VectorDBServerManager, _CONTAINER_NAME


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_manager(**cfg_kwargs) -> VectorDBServerManager:
    defaults = {
        "url": "http://localhost:18000",
        "backend": "qdrant",
        "ssh_host": "gpu-server",
        "auto_deploy": True,
    }
    cfg = VectorDBServerConfig(**{**defaults, **cfg_kwargs})
    mgr = VectorDBServerManager(cfg)
    return mgr


def _fake_run_factory(responses: dict):
    """Return a _run_remote mock whose stdout depends on substrings in the command."""
    def _fake_run(cmd: str) -> tuple[str, str]:
        for substring, output in responses.items():
            if substring in cmd:
                return output, ""
        return "", ""
    return _fake_run


# ---------------------------------------------------------------------------
# Constructor
# ---------------------------------------------------------------------------

class TestManagerConstructor:
    def test_requires_ssh_host(self):
        cfg = VectorDBServerConfig(
            url="http://localhost:18000", backend="qdrant", ssh_host=None
        )
        with pytest.raises(ValueError, match="ssh_host"):
            VectorDBServerManager(cfg)

    def test_paramiko_import_error(self):
        mgr = _make_manager()
        mgr._run_remote = MagicMock()
        with patch.dict("sys.modules", {"paramiko": None}):
            with pytest.raises(ImportError, match="paramiko"):
                mgr.ensure_deployed()


# ---------------------------------------------------------------------------
# ensure_deployed — deploy from scratch
# ---------------------------------------------------------------------------

class TestDeployFromScratch:
    def test_deploys_when_no_metadata(self, tmp_path):
        mgr = _make_manager()

        # _run_remote returns __MISSING__ for metadata cat, "true" for docker inspect
        calls = []
        def fake_run(cmd: str):
            calls.append(cmd)
            if "cat" in cmd and "db_deployment" in cmd:
                return "__MISSING__", ""
            if "docker inspect" in cmd:
                return "false", ""
            if "nvidia-smi" in cmd:
                return "2\n", ""
            return "", ""

        mgr._run_remote = MagicMock(side_effect=fake_run)
        # Mock _upload_build_context to avoid real filesystem + SSH
        mgr._upload_build_context = MagicMock()

        # Stub paramiko import check
        with patch.dict("sys.modules", {"paramiko": MagicMock()}):
            mgr.ensure_deployed()

        # Should have called upload + docker build + docker run
        all_cmds = " ".join(calls)
        assert "docker build" in all_cmds
        assert "docker run" in all_cmds

    def test_no_op_when_container_running(self):
        mgr = _make_manager()

        meta = json.dumps({"deployed_at": "2026-01-01T00:00:00+00:00"})

        def fake_run(cmd: str):
            if "cat" in cmd and "db_deployment" in cmd:
                return meta, ""
            if "docker inspect" in cmd:
                return "true", ""
            return "", ""

        mgr._run_remote = MagicMock(side_effect=fake_run)
        mgr._upload_build_context = MagicMock()

        with patch.dict("sys.modules", {"paramiko": MagicMock()}):
            mgr.ensure_deployed()

        # _upload_build_context should NOT have been called
        mgr._upload_build_context.assert_not_called()

    def test_redeploys_when_container_stopped(self):
        mgr = _make_manager()

        meta = json.dumps({"deployed_at": "2026-01-01T00:00:00+00:00"})
        calls = []

        def fake_run(cmd: str):
            calls.append(cmd)
            if "cat" in cmd and "db_deployment" in cmd:
                return meta, ""
            if "docker inspect" in cmd:
                return "false", ""
            return "", ""

        mgr._run_remote = MagicMock(side_effect=fake_run)
        mgr._upload_build_context = MagicMock()

        with patch.dict("sys.modules", {"paramiko": MagicMock()}):
            mgr.ensure_deployed()

        all_cmds = " ".join(calls)
        assert "docker build" in all_cmds


# ---------------------------------------------------------------------------
# _build_docker_run_cmd
# ---------------------------------------------------------------------------

class TestBuildDockerRunCmd:
    def test_port_in_command(self):
        mgr = _make_manager(port=18000)
        cmd = mgr._build_docker_run_cmd()
        assert "-p 18000:18000" in cmd

    def test_data_dir_mount(self):
        mgr = _make_manager(data_dir="/mnt/data")
        cmd = mgr._build_docker_run_cmd()
        assert "/mnt/data:/data" in cmd

    def test_api_key_in_env(self):
        mgr = _make_manager(api_key="mysecret")
        cmd = mgr._build_docker_run_cmd()
        assert "FOR_DB_API_KEY=mysecret" in cmd

    def test_no_api_key_not_in_cmd(self):
        mgr = _make_manager()
        cmd = mgr._build_docker_run_cmd()
        assert "FOR_DB_API_KEY" not in cmd

    def test_container_name_in_cmd(self):
        mgr = _make_manager()
        cmd = mgr._build_docker_run_cmd()
        assert f"--name {_CONTAINER_NAME}" in cmd


# ---------------------------------------------------------------------------
# stop
# ---------------------------------------------------------------------------

class TestStop:
    def test_stop_sends_correct_commands(self):
        mgr = _make_manager()
        calls = []
        mgr._run_remote = MagicMock(side_effect=lambda cmd: (calls.append(cmd), ("", ""))[1])
        mgr.stop()
        all_cmds = " ".join(calls)
        assert "docker stop" in all_cmds
        assert "docker rm" in all_cmds
        assert "db_deployment" in all_cmds
