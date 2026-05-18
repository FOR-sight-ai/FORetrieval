"""Factory for VectorStore instances.

Usage:
    from foretrieval.vector_store import make_vector_store

    # Local backends
    vs = make_vector_store("qdrant")
    vs.open("my_index", Path(".foretrieval"), create=True, dim=128)

    # Remote server backend
    vs = make_vector_store(
        "remote",
        {
            "url": "http://gpu-server:18000",
            "backend": "qdrant",   # server-side backend
            "api_key": "secret",   # optional bearer token
        }
    )
    vs.open("my_index", Path(".foretrieval"), create=True, dim=128)

Supported backend names:
    "local"  — LocalVectorStore
    "qdrant" — QdrantVectorStore  (requires foretrieval[qdrant])
    "milvus" — MilvusVectorStore  (requires foretrieval[milvus])
    "remote" — RemoteVectorStore  (requires foretrieval[vector_db_server] for
                                   auto_deploy; plain httpx for client-only use)

Future backends can be registered via BACKEND_REGISTRY without changing callers.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional, Type

from .base import VectorStore
from .local import LocalVectorStore
from .qdrant import QdrantVectorStore
from .milvus import MilvusVectorStore
from .remote import RemoteVectorStore

logger = logging.getLogger(__name__)

# Registry maps backend name → class.  Add new entries here to register future
# backends without touching make_vector_store().
BACKEND_REGISTRY: Dict[str, Type[VectorStore]] = {
    "local": LocalVectorStore,
    "qdrant": QdrantVectorStore,
    "milvus": MilvusVectorStore,
    "remote": RemoteVectorStore,
}

# Config keys consumed at the factory level for "remote"; not forwarded as
# server-side storage_config.
_REMOTE_CLIENT_KEYS = {
    "url", "api_key", "verify_ssl", "request_timeout",
    "auto_deploy", "ssh_host", "ssh_user", "ssh_key_path",
    "port", "data_dir",
}
# Health-check wait parameters for auto_deploy
_HEALTH_CHECK_RETRIES = 30
_HEALTH_CHECK_INTERVAL = 2  # seconds


def make_vector_store(
    backend: str,
    storage_config: Optional[Dict[str, Any]] = None,
) -> VectorStore:
    """Instantiate a VectorStore for the given backend name.

    Parameters:
        backend:        Backend identifier: "local", "qdrant", "milvus", or
                        "remote".
        storage_config: Optional dict of backend-specific keyword arguments.

                        For "remote", the following keys are consumed at the
                        factory level and used to build the client/config:
                            url         (required) — base URL of the server
                            backend     (str, default "qdrant") — server-side
                                         backend to use for the collection
                            api_key     (str, optional)
                            verify_ssl  (bool, default True)
                            request_timeout (int, default 120)
                            auto_deploy (bool, default False)
                            ssh_host    (str, optional) — required if auto_deploy
                            ssh_user    (str, optional)
                            ssh_key_path (str, optional)
                            port        (int, default 18000)
                            data_dir    (str, optional) — remote data directory

                        Any remaining keys are forwarded as server-side
                        storage_config (e.g. candidate_limit for Milvus).

                        For "milvus", ``candidate_limit`` is accepted.

    Returns:
        An uninitialised VectorStore.  Call .open() before using it.

    Raises:
        ValueError:   Unknown backend name.
        RuntimeError: Required optional dependency not installed.
    """
    key = (backend or "local").strip().lower()
    cls = BACKEND_REGISTRY.get(key)
    if cls is None:
        supported = ", ".join(sorted(BACKEND_REGISTRY))
        raise ValueError(
            f"Unknown storage backend {backend!r}. "
            f"Supported backends: {supported}."
        )

    kwargs = dict(storage_config or {})

    if cls is LocalVectorStore:
        return LocalVectorStore()

    if cls is QdrantVectorStore:
        return QdrantVectorStore()

    if cls is MilvusVectorStore:
        candidate_limit = kwargs.get("candidate_limit", 64)
        return MilvusVectorStore(candidate_limit=int(candidate_limit))

    if cls is RemoteVectorStore:
        return _make_remote_vector_store(kwargs)

    # Generic fallback for future registered backends
    return cls(**kwargs)  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# Remote store construction helper
# ---------------------------------------------------------------------------

def _make_remote_vector_store(kwargs: Dict[str, Any]) -> RemoteVectorStore:
    """Build a RemoteVectorStore from a storage_config dict."""
    from ..vector_db_server.config import VectorDBServerConfig
    from ..vector_db_server.client import VectorDBServerClient

    # Split kwargs into client-level config and server-side storage_config
    client_kwargs: Dict[str, Any] = {}
    server_storage_config: Dict[str, Any] = {}
    for k, v in kwargs.items():
        if k in _REMOTE_CLIENT_KEYS or k == "backend":
            client_kwargs[k] = v
        else:
            server_storage_config[k] = v

    if "url" not in client_kwargs:
        raise ValueError(
            "storage_config must include 'url' when using the 'remote' backend."
        )

    cfg = VectorDBServerConfig.from_dict(client_kwargs)
    server_backend = cfg.backend
    srv_storage = server_storage_config if server_storage_config else None

    # Auto-deploy: SSH to remote host and ensure server container is running
    if cfg.auto_deploy:
        from ..vector_db_server.manager import VectorDBServerManager
        manager = VectorDBServerManager(cfg)
        manager.ensure_deployed()
        logger.info("Auto-deploy complete. Waiting for server to be ready …")
        _wait_for_health(cfg)

    client = VectorDBServerClient(cfg)
    return RemoteVectorStore(client, backend=server_backend, storage_config=srv_storage)


def _wait_for_health(cfg: Any, retries: int = _HEALTH_CHECK_RETRIES) -> None:
    """Poll /health until the server responds or raise TimeoutError."""
    from ..vector_db_server.client import VectorDBServerClient
    client = VectorDBServerClient(cfg)
    for attempt in range(retries):
        if client.health_check():
            logger.info("Vector-DB server is healthy.")
            client.close()
            return
        logger.debug(
            "Health check attempt %d/%d failed — retrying in %ds …",
            attempt + 1, retries, _HEALTH_CHECK_INTERVAL,
        )
        time.sleep(_HEALTH_CHECK_INTERVAL)
    client.close()
    raise TimeoutError(
        f"Vector-DB server at {cfg.url} did not become healthy "
        f"after {retries * _HEALTH_CHECK_INTERVAL}s."
    )
