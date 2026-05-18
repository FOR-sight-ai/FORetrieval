"""Configuration model for the remote vector-DB server."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, field_validator, model_validator

# Valid backend names that the server can use on its own store.
_SERVER_BACKENDS = ("local", "qdrant", "milvus")


class VectorDBServerConfig(BaseModel):
    """Configuration for a remote FORetrieval vector-DB server.

    When ``storage_backend="remote"`` is passed to ``make_vector_store()``,
    FORetrieval will talk to this server instead of maintaining a local
    vector store.  The server itself runs one of ``local``, ``qdrant``, or
    ``milvus`` as its underlying backend.

    Attributes:
        url: Full base URL of the server, e.g. ``"http://gpu-server:18000"``.
            Trailing slashes are stripped automatically.
        backend: Backend used by the server for the collection.
            One of ``"local"``, ``"qdrant"``, or ``"milvus"``.
            Default ``"qdrant"``.  Once a collection is created on the server
            with a given backend the backend cannot be changed without
            recreating the collection.
        storage_config: Optional backend-specific kwargs forwarded to
            the server (e.g. ``{"candidate_limit": 128}`` for Milvus).
        auto_deploy: When True, FORetrieval will SSH to ssh_host and
            deploy a Docker container if the server is not already running.
            Requires ``ssh_host`` to be set.
        ssh_host: Hostname or IP of the server (SSH target).
            Required when ``auto_deploy=True``.
        ssh_user: SSH username.  Defaults to the current OS user at
            deploy time.
        ssh_key_path: Path to SSH private key.  If None the SSH agent or
            default keys (``~/.ssh/id_rsa`` etc.) are used.
        port: Port the server listens on.  Default ``18000``.
        api_key: Optional bearer token for server authentication.
            When set, requests include ``"Authorization: Bearer <api_key>"``.
            Start the server with ``FOR_DB_API_KEY=<api_key>`` to match.
        verify_ssl: Whether to verify SSL certificates.  Default ``True``.
        request_timeout: HTTP request timeout in seconds.  Default ``120``.
        data_dir: Path **on the remote host** where the server persists its
            data (bind-mounted into the Docker container as ``/data``).
            Default ``/var/lib/foretrieval_db``.
    """

    url: str
    backend: str = "qdrant"
    storage_config: Optional[dict] = None
    auto_deploy: bool = False
    ssh_host: Optional[str] = None
    ssh_user: Optional[str] = None
    ssh_key_path: Optional[str] = None
    port: int = 18000
    api_key: Optional[str] = None
    verify_ssl: bool = True
    request_timeout: int = 120
    data_dir: str = "/var/lib/foretrieval_db"

    @field_validator("url")
    @classmethod
    def strip_trailing_slash(cls, v: str) -> str:
        return v.rstrip("/")

    @field_validator("backend")
    @classmethod
    def validate_backend(cls, v: str) -> str:
        lower = v.lower()
        if lower not in _SERVER_BACKENDS:
            raise ValueError(
                f"backend '{v}' is not a valid server-side backend. "
                f"Valid choices: {', '.join(_SERVER_BACKENDS)}."
            )
        return lower

    @field_validator("port")
    @classmethod
    def validate_port(cls, v: int) -> int:
        if not (1 <= v <= 65535):
            raise ValueError("port must be in range 1–65535")
        return v

    @model_validator(mode="after")
    def auto_deploy_requires_ssh_host(self) -> "VectorDBServerConfig":
        if self.auto_deploy and not self.ssh_host:
            raise ValueError("ssh_host is required when auto_deploy=True")
        return self

    @classmethod
    def from_dict(cls, d: dict) -> "VectorDBServerConfig":
        """Convenience constructor from a plain config dict (e.g. from JSON)."""
        return cls(**d)
