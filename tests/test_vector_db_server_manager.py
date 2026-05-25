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
    # Stub the remote-home resolution so tests that mock _run_remote
    # don't accidentally trigger a real SSH connection through
    # _remote_home() -> sftp.normalize('.').
    mgr._cached_home = "/home/testuser"
    # Stub UID/GID so _build_docker_run_cmd tests don't need SSH.
    # Tests that specifically exercise the --user flag override this.
    mgr._resolve_remote_uid_gid = MagicMock(return_value=(None, None))
    return mgr


def _fake_run_factory(responses: dict):
    """Return a _run_remote mock whose stdout depends on substrings in the command."""
    def _fake_run(cmd: str, on_line=None) -> tuple[str, str]:
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
        def fake_run(cmd: str, on_line=None):
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

        def fake_run(cmd: str, on_line=None):
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

        def fake_run(cmd: str, on_line=None):
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

    def test_user_flag_present_when_uid_resolved(self):
        """When UID/GID can be resolved, --user uid:gid must appear."""
        mgr = _make_manager()
        mgr._resolve_remote_uid_gid = MagicMock(return_value=(1000, 1001))
        cmd = mgr._build_docker_run_cmd()
        assert "--user 1000:1001" in cmd

    def test_user_flag_absent_when_uid_unknown(self):
        """When resolution fails (returns None, None), no --user flag."""
        mgr = _make_manager()
        # _resolve_remote_uid_gid already returns (None, None) from _make_manager
        cmd = mgr._build_docker_run_cmd()
        assert "--user" not in cmd


# ---------------------------------------------------------------------------
# stop
# ---------------------------------------------------------------------------

class TestStop:
    def test_stop_sends_correct_commands(self):
        mgr = _make_manager()
        calls = []
        mgr._run_remote = MagicMock(side_effect=lambda cmd, on_line=None: (calls.append(cmd), ("", ""))[1])
        mgr.stop()
        all_cmds = " ".join(calls)
        assert "docker stop" in all_cmds
        assert "docker rm" in all_cmds
        assert "db_deployment" in all_cmds


# ---------------------------------------------------------------------------
# redeploy + on_line streaming
# ---------------------------------------------------------------------------

class TestRedeploy:
    def test_redeploy_calls_deploy(self):
        mgr = _make_manager()
        mgr._deploy = MagicMock()
        with patch.dict("sys.modules", {"paramiko": MagicMock()}):
            mgr.redeploy()
        mgr._deploy.assert_called_once()

    def test_redeploy_forwards_on_line_callback(self):
        mgr = _make_manager()
        mgr._deploy = MagicMock()
        cb = MagicMock()
        with patch.dict("sys.modules", {"paramiko": MagicMock()}):
            mgr.redeploy(on_line=cb)
        mgr._deploy.assert_called_once()
        # The callback was threaded through
        assert mgr._deploy.call_args.kwargs.get("on_line") is cb

    def test_redeploy_unconditionally_redeploys_running_container(self):
        """Unlike ensure_deployed, redeploy() always rebuilds."""
        mgr = _make_manager()

        # _is_container_running returns True; ensure_deployed would no-op,
        # but redeploy must still trigger _deploy.
        mgr._is_container_running = MagicMock(return_value=True)
        mgr._read_remote_metadata = MagicMock(return_value={"deployed_at": "x"})
        mgr._deploy = MagicMock()
        with patch.dict("sys.modules", {"paramiko": MagicMock()}):
            mgr.redeploy()
        mgr._deploy.assert_called_once()


class TestOnLineStreaming:
    def test_run_remote_streams_lines_when_callback_given(self):
        """When on_line is provided, stdout is read line-by-line."""
        mgr = _make_manager()

        # Build a fake SSH client whose exec_command returns three lines
        class _FakeChannel:
            def recv_exit_status(self):
                return 0

        class _FakeStdout:
            def __init__(self, lines):
                self._lines = list(lines)
                self.channel = _FakeChannel()

            def readline(self):
                if self._lines:
                    return self._lines.pop(0)
                return ""

            def read(self):
                return b""

        fake_stdout = _FakeStdout(["Step 1/3\n", "Step 2/3\n", "Done\n"])
        fake_stderr = MagicMock()
        fake_stderr.read.return_value = b""
        fake_ssh = MagicMock()
        fake_ssh.exec_command.return_value = (None, fake_stdout, fake_stderr)
        mgr._get_ssh = MagicMock(return_value=fake_ssh)

        captured = []
        stdout, stderr = mgr._run_remote("docker build .", on_line=captured.append)

        assert captured == ["Step 1/3", "Step 2/3", "Done"]
        assert "Step 1/3" in stdout
        assert stderr == ""

    def test_run_remote_buffered_when_no_callback(self):
        """Default behaviour (no on_line) still reads the whole stdout at once."""
        mgr = _make_manager()

        class _FakeChannel:
            def recv_exit_status(self):
                return 0

        class _FakeStdout:
            channel = _FakeChannel()
            def read(self):
                return b"full output\n"

        fake_stderr = MagicMock()
        fake_stderr.read.return_value = b""
        fake_ssh = MagicMock()
        fake_ssh.exec_command.return_value = (None, _FakeStdout(), fake_stderr)
        mgr._get_ssh = MagicMock(return_value=fake_ssh)

        stdout, _ = mgr._run_remote("echo hi")
        assert "full output" in stdout


# ---------------------------------------------------------------------------
# _get_ssh uses ssh_utils.open_ssh_client
# ---------------------------------------------------------------------------

class TestGetSshUsesSshUtils:
    def test_get_ssh_delegates_to_open_ssh_client(self):
        """_get_ssh must call foretrieval.ssh_utils.open_ssh_client.

        This is the regression test for the bug where paramiko was
        called with a raw alias string and DNS-failed for entries that
        only existed in ~/.ssh/config.
        """
        mgr = _make_manager(ssh_host="pf01", ssh_user="alice", ssh_key_path="/k")
        fake_client = MagicMock()
        with patch("foretrieval.ssh_utils.open_ssh_client",
                   return_value=fake_client) as mock_open:
            result = mgr._get_ssh()
        mock_open.assert_called_once_with(
            ssh_host="pf01",
            ssh_user="alice",
            ssh_key_path="/k",
        )
        assert result is fake_client

    def test_get_ssh_is_cached(self):
        mgr = _make_manager()
        fake_client = MagicMock()
        with patch("foretrieval.ssh_utils.open_ssh_client",
                   return_value=fake_client) as mock_open:
            r1 = mgr._get_ssh()
            r2 = mgr._get_ssh()
        assert r1 is r2
        assert mock_open.call_count == 1


# ---------------------------------------------------------------------------
# Remote home / metadata path resolution
# ---------------------------------------------------------------------------

class TestRemoteHome:
    def test_remote_home_uses_sftp_normalize(self):
        """_remote_home returns the path produced by sftp.normalize('.')."""
        mgr = _make_manager()
        mgr._cached_home = None  # reset the stub from _make_manager

        fake_sftp = MagicMock()
        fake_sftp.normalize.return_value = "/home/alice"
        fake_ssh = MagicMock()
        fake_ssh.open_sftp.return_value = fake_sftp
        mgr._get_ssh = MagicMock(return_value=fake_ssh)

        assert mgr._remote_home() == "/home/alice"
        fake_sftp.normalize.assert_called_once_with(".")
        fake_sftp.close.assert_called_once()

    def test_remote_home_caches_result(self):
        mgr = _make_manager()
        mgr._cached_home = None

        fake_sftp = MagicMock()
        fake_sftp.normalize.return_value = "/home/alice"
        fake_ssh = MagicMock()
        fake_ssh.open_sftp.return_value = fake_sftp
        mgr._get_ssh = MagicMock(return_value=fake_ssh)

        a = mgr._remote_home()
        b = mgr._remote_home()
        assert a == b == "/home/alice"
        fake_ssh.open_sftp.assert_called_once()


class TestMetadataPathAbsolute:
    def test_metadata_path_is_absolute(self):
        mgr = _make_manager()  # _cached_home = "/home/testuser"
        path = mgr._metadata_path()
        assert path.startswith("/home/testuser/")
        assert "~" not in path
        assert path.endswith("db_deployment.json")

    def test_stop_uses_absolute_metadata_path(self):
        """Regression: stop() must rm an absolute path, not '~/...'."""
        mgr = _make_manager()
        calls = []
        mgr._run_remote = MagicMock(
            side_effect=lambda cmd, on_line=None: (calls.append(cmd), ("", ""))[1]
        )
        mgr.stop()
        rm_calls = [c for c in calls if c.startswith("rm -f")]
        assert rm_calls, "stop() did not call rm -f"
        # No tilde in the path passed to rm
        assert all("~" not in c for c in rm_calls), rm_calls
        assert any("/home/testuser/" in c for c in rm_calls), rm_calls


class TestUploadBuildContextAbsolutePath:
    def test_sftp_put_uses_absolute_remote_path(self, tmp_path, monkeypatch):
        """Regression for the SFTP ENOENT bug.

        The previous implementation passed
            f"{'~/foretrieval_db_build'.replace('~', '')}/build_context.tar.gz"
        which resolved to '/foretrieval_db_build/...' (root-relative).
        The new code must pass an absolute path rooted at the remote home.
        """
        from foretrieval.vector_db_server import manager as mod

        mgr = _make_manager()  # _cached_home = "/home/testuser"

        # Mock _run_remote (mkdir, tar extract)
        runs: list[str] = []
        mgr._run_remote = MagicMock(
            side_effect=lambda cmd, on_line=None: (runs.append(cmd), ("", ""))[1]
        )

        # Mock SSH + SFTP
        sftp_calls: list[tuple] = []

        class _FakeSftp:
            def put(self, local, remote):
                sftp_calls.append(("put", local, remote))
            def close(self):
                sftp_calls.append(("close",))

        fake_sftp = _FakeSftp()
        fake_ssh = MagicMock()
        fake_ssh.open_sftp.return_value = fake_sftp
        mgr._get_ssh = MagicMock(return_value=fake_ssh)

        # Avoid real disk + real tar — short-circuit _upload_build_context's
        # source-locating step by pointing it at the actual repo (it exists).
        # tarfile is small so let it run.
        mgr._upload_build_context()

        # Verify SFTP put destination
        put_calls = [c for c in sftp_calls if c[0] == "put"]
        assert len(put_calls) == 1
        remote_dest = put_calls[0][2]
        assert remote_dest == "/home/testuser/foretrieval_db_build/build_context.tar.gz", remote_dest

        # Verify the mkdir uses the same absolute path
        mkdir_calls = [c for c in runs if c.startswith("mkdir -p")]
        assert any("/home/testuser/foretrieval_db_build" in c for c in mkdir_calls), mkdir_calls

        # Verify tar extraction targets the same absolute path
        tar_calls = [c for c in runs if "tar -xzf" in c]
        assert any("/home/testuser/foretrieval_db_build" in c for c in tar_calls), tar_calls

        # _remote_build_dir stashed for _deploy() to reuse
        assert mgr._remote_build_dir == "/home/testuser/foretrieval_db_build"
