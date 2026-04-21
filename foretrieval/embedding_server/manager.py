"""Remote embedding server deployment manager.

Handles Docker-based deployment of a vLLM embedding server on a remote GPU
machine via SSH.

Deployment metadata is stored at ~/.foretrieval/deployment.json on the *remote*
server.  This file acts as the authoritative record of what is running:
- If the file is absent → deploy from scratch.
- If the file is present → health-check the running container; redeploy if down.

The manager always requires auto_deploy=True to trigger any SSH activity;
callers that only want to USE an existing server need not instantiate this.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from typing import Optional

from .config import EmbeddingServerConfig

logger = logging.getLogger(__name__)

# Remote path where deployment metadata is stored.
_REMOTE_METADATA_PATH = "~/.foretrieval/deployment.json"
_CONTAINER_NAME = "foretrieval_embedding_server"

# vLLM Docker image.
_VLLM_IMAGE = "vllm/vllm-openai:latest"


class EmbeddingServerManager:
    """Manages deployment of the vLLM embedding server via SSH + Docker.

    Parameters
    ----------
    config:
        EmbeddingServerConfig with ssh_host, ssh_user, n_gpus, port, etc.
    """

    def __init__(self, config: EmbeddingServerConfig) -> None:
        if not config.ssh_host:
            raise ValueError("EmbeddingServerManager requires ssh_host in config")
        self.config = config
        self._ssh: Optional[object] = None  # paramiko.SSHClient, lazy

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ensure_deployed(self) -> None:
        """Ensure the embedding server is running on the remote host.

        Flow:
        1. Check remote metadata file.
        2. If absent → deploy from scratch.
        3. If present → health-check; redeploy if unhealthy.
        """
        try:
            import paramiko  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "paramiko is required for auto_deploy. "
                "Install it with: pip install 'foretrieval[embedding_server]'"
            ) from exc

        logger.info("Ensuring embedding server is deployed on %s", self.config.ssh_host)
        metadata = self._read_remote_metadata()

        if metadata is None:
            logger.info("No deployment metadata found — deploying from scratch")
            self._deploy()
        else:
            logger.info(
                "Found existing deployment (model=%s, deployed_at=%s)",
                metadata.get("model_name"),
                metadata.get("deployed_at"),
            )
            if self._is_container_running():
                logger.info("Container is running and healthy — nothing to do")
            else:
                logger.warning("Container not running — redeploying")
                self._deploy()

    def stop(self) -> None:
        """Stop and remove the Docker container, delete metadata file."""
        logger.info("Stopping embedding server on %s", self.config.ssh_host)
        self._run_remote(f"docker stop {_CONTAINER_NAME} 2>/dev/null || true")
        self._run_remote(f"docker rm {_CONTAINER_NAME} 2>/dev/null || true")
        self._run_remote(f"rm -f {_REMOTE_METADATA_PATH}")
        logger.info("Embedding server stopped")

    # ------------------------------------------------------------------
    # Deploy
    # ------------------------------------------------------------------

    def _deploy(self) -> None:
        """Pull image, resolve GPU count, run container, write metadata."""
        # Resolve GPU count.
        n_gpus = self._resolve_n_gpus()
        logger.info("Using %d GPU(s) for tensor parallelism", n_gpus)

        # Stop any stale container first.
        self._run_remote(f"docker stop {_CONTAINER_NAME} 2>/dev/null || true")
        self._run_remote(f"docker rm {_CONTAINER_NAME} 2>/dev/null || true")

        # Pull image (no-op if already present).
        logger.info("Pulling %s", _VLLM_IMAGE)
        self._run_remote(f"docker pull {_VLLM_IMAGE}")

        # Build docker run command.
        cmd = self._build_docker_run_cmd(n_gpus)
        logger.info("Starting container: %s", cmd)
        self._run_remote(cmd)

        # Write metadata.
        metadata = {
            "model_name": self.config.model_name,
            "container_name": _CONTAINER_NAME,
            "port": self.config.port,
            "n_gpus": n_gpus,
            "image": _VLLM_IMAGE,
            "deployed_at": datetime.now(timezone.utc).isoformat(),
        }
        self._write_remote_metadata(metadata)
        logger.info("Deployment complete — server starting up on port %d", self.config.port)

    def _build_docker_run_cmd(self, n_gpus: int) -> str:
        cfg = self.config
        gpu_flag = "--gpus all" if cfg.n_gpus == -1 else f"--gpus {cfg.n_gpus}"

        hf_home = "/opt/huggingface"
        env_parts = [f"-e HF_HOME={hf_home}"]
        if cfg.hf_token:
            env_parts.append(f"-e HF_TOKEN={cfg.hf_token}")

        vol_parts = [f"-v {hf_home}:{hf_home}"]

        # vLLM >=0.19.0: image entrypoint is already "vllm serve".
        # Pass model + flags directly. --task removed; use --runner pooling + --convert embed.
        # --max-model-len 8192: caps encoder cache budget to avoid OOM on 24GB GPUs.
        # --gpu-memory-utilization 0.7: leaves headroom for KV cache allocation.
        model_args = (
            f"{cfg.model_name} "
            f"--runner pooling "
            f"--convert embed "
            f"--tensor-parallel-size {n_gpus} "
            f"--gpu-memory-utilization 0.7 "
            f"--max-model-len 8192 "
            f"--trust-remote-code"
        )

        return (
            f"docker run -d "
            f"--name {_CONTAINER_NAME} "
            f"{gpu_flag} "
            f"-p {cfg.port}:8000 "
            f"{' '.join(env_parts)} "
            f"{' '.join(vol_parts)} "
            f"--restart unless-stopped "
            f"--ipc=host "
            f"{_VLLM_IMAGE} "
            f"{model_args}"
        )

    # ------------------------------------------------------------------
    # GPU detection
    # ------------------------------------------------------------------

    def _resolve_n_gpus(self) -> int:
        """Return actual GPU count: query remote if n_gpus=-1, else use config value."""
        if self.config.n_gpus != -1:
            return self.config.n_gpus
        stdout, _ = self._run_remote(
            "nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l"
        )
        try:
            count = int(stdout.strip())
        except ValueError:
            count = 1
        if count < 1:
            count = 1
        logger.info("Detected %d GPU(s) on remote host", count)
        return count

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
        stdout, stderr = self._run_remote(
            f"cat {_REMOTE_METADATA_PATH} 2>/dev/null || echo '__MISSING__'"
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
        json_str = json.dumps(metadata).replace("'", "'\\''")
        self._run_remote(
            f"mkdir -p $(dirname {_REMOTE_METADATA_PATH}) && "
            f"echo '{json_str}' > {_REMOTE_METADATA_PATH}"
        )

    # ------------------------------------------------------------------
    # SSH helpers
    # ------------------------------------------------------------------

    def _get_ssh(self):
        """Return a connected paramiko SSHClient (lazy init)."""
        import paramiko

        if self._ssh is not None:
            return self._ssh

        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        connect_kwargs: dict = {
            "hostname": self.config.ssh_host,
            "username": self.config.ssh_user or os.environ.get("USER", "root"),
        }
        if self.config.ssh_key_path:
            connect_kwargs["key_filename"] = self.config.ssh_key_path

        client.connect(**connect_kwargs)
        self._ssh = client
        return client

    def _run_remote(self, cmd: str) -> tuple[str, str]:
        """Run a shell command on the remote host and return (stdout, stderr).

        Raises RuntimeError if the exit code is non-zero (for commands that
        don't have their own || true fallback).
        """
        ssh = self._get_ssh()
        logger.debug("Remote: %s", cmd)
        _, stdout_f, stderr_f = ssh.exec_command(cmd)
        exit_code = stdout_f.channel.recv_exit_status()
        stdout = stdout_f.read().decode("utf-8", errors="replace")
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
