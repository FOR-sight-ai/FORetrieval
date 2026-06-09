"""Remote vector-DB server deployment manager.

Handles Docker-based deployment of the FORetrieval vector-DB server on a
remote host via SSH.  Mirrors the structure of EmbeddingServerManager.

Deployment flow:
1. Check remote metadata file (``~/.foretrieval/db_deployment.json``).
2. If absent → build image and run container from scratch.
3. If present → health-check the running container; redeploy if down.

The Docker image is built **on the remote host** from the local
``foretrieval/`` package source, transferred via SFTP.  This avoids needing
a public image registry.
"""

from __future__ import annotations

import json
import logging
import os
import tarfile
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

from .config import VectorDBServerConfig

logger = logging.getLogger(__name__)

# Remote paths and container constants.
#
# Paths are intentionally relative.  They are joined to the SSH user's
# absolute home directory (obtained via ``sftp.normalize('.')`` so that
# SFTP put / get commands receive a real absolute path — SFTP does NOT
# expand ``~``, unlike the remote shell.
_REMOTE_METADATA_SUBPATH = ".foretrieval/db_deployment.json"
_REMOTE_BUILD_SUBDIR = "foretrieval_db_build"
_CONTAINER_NAME = "foretrieval_vector_db_server"
_IMAGE_NAME = "foretrieval-vector-db:local"

# Port the server listens on *inside* the container.
# This is fixed by the Dockerfile CMD / server_main.py and is always 18000.
# VectorDBServerConfig.port controls only the *host-side* binding
# (the left-hand side of Docker's -p HOST:CONTAINER mapping), allowing
# callers to expose the service on any host port without rebuilding the image.
_CONTAINER_INTERNAL_PORT = 18000


class VectorDBServerManager:
    """Manages deployment of the FORetrieval vector-DB server via SSH + Docker.

    Parameters
    ----------
    config:
        VectorDBServerConfig with ssh_host, port, data_dir, api_key, etc.
    """

    def __init__(self, config: VectorDBServerConfig) -> None:
        if not config.ssh_host:
            raise ValueError("VectorDBServerManager requires ssh_host in config")
        self.config = config
        self._ssh: Optional[object] = None  # paramiko.SSHClient, lazy
        self._cached_home: Optional[str] = None  # remote $HOME, lazy
        # Absolute remote build dir, set by _upload_build_context() once
        # _remote_home() has been resolved.  Read by _deploy() and tests.
        self._remote_build_dir: Optional[str] = None

    # ------------------------------------------------------------------
    # Remote path helpers
    # ------------------------------------------------------------------

    def _remote_home(self) -> str:
        """Return the SSH user's absolute home directory on the remote host.

        Resolved once per manager instance via ``sftp.normalize('.')``.
        """
        if self._cached_home is not None:
            return self._cached_home
        ssh = self._get_ssh()
        sftp = ssh.open_sftp()
        try:
            self._cached_home = sftp.normalize(".")
        finally:
            sftp.close()
        return self._cached_home

    def _metadata_path(self) -> str:
        """Return the absolute path to the remote deployment-metadata file."""
        return f"{self._remote_home()}/{_REMOTE_METADATA_SUBPATH}"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ensure_deployed(self) -> None:
        """Ensure the vector-DB server is running on the remote host.

        Flow:
        1. Check remote metadata file.
        2. If absent → build image and deploy from scratch.
        3. If present → health-check container; redeploy if unhealthy.
        """
        try:
            import paramiko  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "paramiko is required for auto_deploy. "
                "Install it with: pip install 'foretrieval[vector_db_server]'"
            ) from exc

        logger.info(
            "Ensuring vector-DB server is deployed on %s", self.config.ssh_host
        )
        metadata = self._read_remote_metadata()

        if metadata is None:
            logger.info("No deployment metadata found — deploying from scratch")
            self._deploy()
        else:
            logger.info(
                "Found existing deployment (deployed_at=%s)",
                metadata.get("deployed_at"),
            )
            if self._is_container_running():
                logger.info("Container is running — nothing to do")
            else:
                logger.warning("Container not running — redeploying")
                self._deploy()

    def redeploy(self, on_line: Optional[Callable[[str], None]] = None) -> None:
        """Force a fresh build + container restart regardless of current state.

        Unlike :py:meth:`ensure_deployed`, this always rebuilds the image and
        restarts the container.  Use it after pulling new FORetrieval source
        on the local machine.

        Args:
            on_line: Optional callback invoked with every line of remote
                stdout (Docker pull / build / run output).  Best-effort:
                callback exceptions are swallowed.
        """
        try:
            import paramiko  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "paramiko is required for auto_deploy. "
                "Install it with: pip install 'foretrieval[vector_db_server]'"
            ) from exc

        logger.info(
            "Force redeploying vector-DB server on %s", self.config.ssh_host
        )
        self._deploy(on_line=on_line)

    def is_running(self) -> bool:
        """Return True iff the container is currently up."""
        try:
            return self._is_container_running()
        except Exception:  # noqa: BLE001
            return False

    def get_remote_metadata(self) -> Optional[dict]:
        """Return the remote deployment metadata, or None if not deployed."""
        try:
            return self._read_remote_metadata()
        except Exception:  # noqa: BLE001
            return None

    def stop(self) -> None:
        """Stop and remove the Docker container; delete metadata file."""
        logger.info("Stopping vector-DB server on %s", self.config.ssh_host)
        self._run_remote(f"docker stop {_CONTAINER_NAME} 2>/dev/null || true")
        self._run_remote(f"docker rm {_CONTAINER_NAME} 2>/dev/null || true")
        self._run_remote(f"rm -f {self._metadata_path()}")
        logger.info("Vector-DB server stopped")

    # ------------------------------------------------------------------
    # Deploy
    # ------------------------------------------------------------------

    def _deploy(self, on_line: Optional[Callable[[str], None]] = None) -> None:
        """Upload source, build image, run container, write metadata.

        Args:
            on_line: Optional callback invoked with every line of remote
                stdout produced by the long-running build/run commands.
        """
        # Stop stale container
        self._run_remote(f"docker stop {_CONTAINER_NAME} 2>/dev/null || true", on_line=on_line)
        self._run_remote(f"docker rm {_CONTAINER_NAME} 2>/dev/null || true", on_line=on_line)

        # Upload foretrieval source + Dockerfile to remote build dir
        if on_line is not None:
            try:
                on_line("Uploading build context …")
            except Exception:  # noqa: BLE001
                pass
        self._upload_build_context()

        # Build Docker image on the remote host
        logger.info("Building Docker image '%s' on %s …", _IMAGE_NAME, self.config.ssh_host)
        if on_line is not None:
            try:
                on_line(f"Building image {_IMAGE_NAME} …")
            except Exception:  # noqa: BLE001
                pass
        self._run_remote(
            f"cd {self._remote_build_dir} && "
            f"docker build -t {_IMAGE_NAME} -f Dockerfile.vector_db .",
            on_line=on_line,
        )

        # Create data directory on remote
        self._run_remote(f"mkdir -p {self.config.data_dir}", on_line=on_line)

        # Run container
        cmd = self._build_docker_run_cmd()
        logger.info("Starting container: %s", cmd)
        if on_line is not None:
            try:
                on_line("Starting container …")
            except Exception:  # noqa: BLE001
                pass
        self._run_remote(cmd, on_line=on_line)

        # Write metadata
        metadata = {
            "container_name": _CONTAINER_NAME,
            "image": _IMAGE_NAME,
            "port": self.config.port,
            "data_dir": self.config.data_dir,
            "deployed_at": datetime.now(timezone.utc).isoformat(),
        }
        self._write_remote_metadata(metadata)
        logger.info(
            "Deployment complete — server starting on port %d", self.config.port
        )
        if on_line is not None:
            try:
                on_line("Deployment complete.")
            except Exception:  # noqa: BLE001
                pass

    def _build_docker_run_cmd(self) -> str:
        cfg = self.config
        env_parts = [
            "-e FOR_DB_DATA_DIR=/data",
            # FOR_DB_PORT sets the port the server listens on inside the container.
            # This must always match _CONTAINER_INTERNAL_PORT — not cfg.port, which
            # is the host-side binding.  cfg.port is used below in -p HOST:CONTAINER.
            f"-e FOR_DB_PORT={_CONTAINER_INTERNAL_PORT}",
        ]
        if cfg.api_key:
            env_parts.append(f"-e FOR_DB_API_KEY={cfg.api_key}")

        # Run the container as the SSH user so files written under the
        # bind-mounted data_dir have the same ownership as SFTP-uploaded
        # files.  Falls back to the Docker default (root) if UID resolution
        # fails, so existing deployments without SSH are unaffected.
        uid, gid = self._resolve_remote_uid_gid()
        user_flag = f"--user {uid}:{gid}" if uid is not None else ""

        parts = [
            "docker run -d",
            f"--name {_CONTAINER_NAME}",
        ]
        if user_flag:
            parts.append(user_flag)
        parts += [
            # cfg.port  → host port (configurable, chosen by the caller)
            # _CONTAINER_INTERNAL_PORT → container port (fixed by the image)
            f"-p {cfg.port}:{_CONTAINER_INTERNAL_PORT}",
            f"-v {cfg.data_dir}:/data",
            " ".join(env_parts),
            "--restart unless-stopped",
            _IMAGE_NAME,
        ]
        return " ".join(parts)

    def _resolve_remote_uid_gid(self) -> tuple:
        """Return ``(uid, gid)`` of the SSH user on the remote host.

        Uses a single ``id -u && id -g`` call.  Returns ``(None, None)``
        on any failure so callers can omit the ``--user`` flag gracefully.
        """
        try:
            stdout, _ = self._run_remote("id -u && id -g")
            lines = [ln.strip() for ln in stdout.strip().splitlines() if ln.strip()]
            if len(lines) >= 2:
                return int(lines[0]), int(lines[1])
        except Exception:  # noqa: BLE001
            pass
        return None, None

    # ------------------------------------------------------------------
    # Build context upload
    # ------------------------------------------------------------------

    def _upload_build_context(self) -> None:
        """Create a tar of foretrieval/ + Dockerfile.vector_db and upload to remote."""

        # Locate the local foretrieval package root (parent of vector_db_server/)
        package_root = Path(__file__).parent.parent.parent  # …/FORetrieval/
        foretrieval_src = package_root / "foretrieval"
        dockerfile_src = Path(__file__).parent / "Dockerfile.vector_db"

        if not foretrieval_src.is_dir():
            raise RuntimeError(
                f"Could not locate foretrieval source at {foretrieval_src}"
            )
        if not dockerfile_src.exists():
            raise RuntimeError(
                f"Dockerfile.vector_db not found at {dockerfile_src}"
            )

        with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp:
            tmp_path = tmp.name

        logger.info("Creating build context archive …")
        with tarfile.open(tmp_path, "w:gz") as tar:
            tar.add(foretrieval_src, arcname="foretrieval")
            tar.add(dockerfile_src, arcname="Dockerfile.vector_db")
            # Include pyproject.toml if available (for pip install -e .)
            pyproject = package_root / "pyproject.toml"
            if pyproject.exists():
                tar.add(pyproject, arcname="pyproject.toml")

        # Resolve the absolute remote build directory.  SFTP does NOT expand
        # ``~`` (unlike the remote shell), so we must compute the absolute
        # path explicitly via the SSH user's home directory.
        remote_dir = f"{self._remote_home()}/{_REMOTE_BUILD_SUBDIR}"
        self._remote_build_dir = remote_dir

        logger.info("Uploading build context to %s:%s …", self.config.ssh_host, remote_dir)
        ssh = self._get_ssh()
        # Ensure remote dir exists (shell expansion not needed any more,
        # but a missing parent would still ENOENT below).
        self._run_remote(f"mkdir -p {remote_dir}")
        sftp = ssh.open_sftp()
        try:
            sftp.put(tmp_path, f"{remote_dir}/build_context.tar.gz")
        finally:
            sftp.close()
            os.unlink(tmp_path)

        # Extract on remote
        self._run_remote(
            f"cd {remote_dir} && "
            f"tar -xzf build_context.tar.gz && "
            f"rm build_context.tar.gz"
        )
        logger.info("Build context uploaded and extracted.")

    # ------------------------------------------------------------------
    # Health / container status
    # ------------------------------------------------------------------

    def _is_container_running(self) -> bool:
        """Return True if the Docker container exists and is running."""
        stdout, _ = self._run_remote(
            f"docker inspect --format='{{{{.State.Running}}}}' "
            f"{_CONTAINER_NAME} 2>/dev/null || echo false"
        )
        return stdout.strip().lower() == "true"

    # ------------------------------------------------------------------
    # Remote metadata
    # ------------------------------------------------------------------

    def _read_remote_metadata(self) -> Optional[dict]:
        path = self._metadata_path()
        stdout, _ = self._run_remote(
            f"cat {path} 2>/dev/null || echo '__MISSING__'"
        )
        text = stdout.strip()
        if text == "__MISSING__" or not text:
            return None
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            logger.warning("Could not parse remote metadata: %s", text[:200])
            return None

    def _write_remote_metadata(self, metadata: dict) -> None:
        path = self._metadata_path()
        json_str = json.dumps(metadata).replace("'", "'\\''")
        self._run_remote(
            f"mkdir -p $(dirname {path}) && "
            f"echo '{json_str}' > {path}"
        )

    # ------------------------------------------------------------------
    # SSH helpers
    # ------------------------------------------------------------------

    def _get_ssh(self):
        """Return a connected paramiko SSHClient (lazy init).

        Honours ``~/.ssh/config`` (Host aliases, User, Port, IdentityFile,
        ProxyCommand, ProxyJump) via :py:func:`foretrieval.ssh_utils.open_ssh_client`.
        """
        if self._ssh is not None:
            return self._ssh

        from ..ssh_utils import open_ssh_client
        self._ssh = open_ssh_client(
            ssh_host=self.config.ssh_host,
            ssh_user=self.config.ssh_user,
            ssh_key_path=self.config.ssh_key_path,
        )
        return self._ssh

    def _run_remote(
        self,
        cmd: str,
        on_line: Optional[Callable[[str], None]] = None,
    ) -> tuple[str, str]:
        """Run a shell command on the remote host; return (stdout, stderr).

        When ``on_line`` is provided, stdout is streamed line by line and the
        callback is invoked for each.  Useful for surfacing Docker build
        progress in interactive UIs.  Callback exceptions are swallowed.
        """
        ssh = self._get_ssh()
        logger.debug("Remote: %s", cmd)
        _, stdout_f, stderr_f = ssh.exec_command(cmd)

        if on_line is None:
            exit_code = stdout_f.channel.recv_exit_status()
            stdout = stdout_f.read().decode("utf-8", errors="replace")
            stderr = stderr_f.read().decode("utf-8", errors="replace")
        else:
            collected: list[str] = []
            # Stream stdout
            for raw in iter(stdout_f.readline, ""):
                if not raw:
                    break
                collected.append(raw)
                try:
                    on_line(raw.rstrip("\n"))
                except Exception:  # noqa: BLE001
                    pass
            exit_code = stdout_f.channel.recv_exit_status()
            stdout = "".join(collected)
            stderr = stderr_f.read().decode("utf-8", errors="replace")

        if stderr:
            logger.debug("Remote stderr: %s", stderr[:300])
        if exit_code != 0 and "|| true" not in cmd and "2>/dev/null" not in cmd:
            raise RuntimeError(
                f"Remote command failed (exit {exit_code}): {cmd}\n"
                f"stderr: {stderr[:500]}"
            )
        return stdout, stderr

    def __del__(self) -> None:
        if self._ssh is not None:
            try:
                self._ssh.close()
            except Exception:
                pass
