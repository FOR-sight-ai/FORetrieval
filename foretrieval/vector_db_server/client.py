"""HTTP client for the remote FORetrieval vector-DB server.

Communicates with a FastAPI server exposing the VectorStore HTTP API.
Multi-vector tensors are transported as raw bytes via torch.save/torch.load
(``Content-Type: application/octet-stream``) for efficiency and dtype
preservation.  Control responses (health, exists checks, create) use JSON.

Auth: when VectorDBServerConfig.api_key is set, every request includes an
``Authorization: Bearer <api_key>`` header.

SSL: ``verify_ssl=False`` disables certificate verification (self-signed certs).
"""

from __future__ import annotations

import io
import logging
from typing import Any, Dict, List, Optional

import httpx
import torch

from ..vector_store.base import MultiVectorQuery, SearchHit, StoredPoint
from .config import VectorDBServerConfig

logger = logging.getLogger(__name__)

# Endpoint paths
_HEALTH_ENDPOINT = "/health"
_COLLECTION_OPEN = "/v1/collection/open"
_COLLECTION_EXISTS = "/v1/collection/{name}/exists"
_COLLECTION_CREATE = "/v1/collection"
_COLLECTION_DELETE = "/v1/collection/{name}"
_POINT_EXISTS = "/v1/point/{name}/{point_id}/exists"
_UPSERT = "/v1/upsert/{name}"
_SEARCH = "/v1/search/{name}"
_VECTOR = "/v1/vector/{name}/{point_id}"
_ADMIN_INDEXES = "/v1/admin/indexes"
_ADMIN_DATA_FOLDERS = "/v1/admin/data_folders"


# ---------------------------------------------------------------------------
# Tensor codec (torch.save / torch.load via BytesIO)
# ---------------------------------------------------------------------------

def _dumps(obj: Any) -> bytes:
    """Serialise an arbitrary object (tensors, dicts, lists) via torch.save."""
    buf = io.BytesIO()
    torch.save(obj, buf)
    return buf.getvalue()


def _loads(data: bytes) -> Any:
    """Deserialise bytes produced by _dumps."""
    return torch.load(io.BytesIO(data), map_location="cpu", weights_only=False)


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

class VectorDBServerClient:
    """HTTP client that talks to the FORetrieval vector-DB server.

    Parameters
    ----------
    config:
        VectorDBServerConfig with url, api_key, verify_ssl, request_timeout, etc.
    """

    def __init__(self, config: VectorDBServerConfig) -> None:
        self.config = config
        self._client = httpx.Client(
            verify=config.verify_ssl,
            headers=self._build_headers(),
            timeout=config.request_timeout,
        )

    def _build_headers(self) -> dict:
        headers: Dict[str, str] = {}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        return headers

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------

    def health_check(self) -> bool:
        """Return True if the server is reachable and healthy."""
        try:
            resp = self._client.get(self.config.url + _HEALTH_ENDPOINT, timeout=10)
            return resp.status_code == 200
        except httpx.HTTPError:
            return False

    # ------------------------------------------------------------------
    # Collection management
    # ------------------------------------------------------------------

    def open_collection(
        self,
        index_name: str,
        backend: str,
        *,
        create: bool,
        dim: Optional[int] = None,
        storage_config: Optional[dict] = None,
    ) -> dict:
        """Ask the server to open (or create) a collection.

        Returns the server's JSON response (``opened``, ``backend``, ``created``).
        """
        payload: Dict[str, Any] = {
            "index_name": index_name,
            "backend": backend,
            "create": create,
        }
        if dim is not None:
            payload["dim"] = dim
        if storage_config:
            payload["storage_config"] = storage_config
        resp = self._post_json(_COLLECTION_OPEN, payload)
        return resp

    def collection_exists(self, index_name: str) -> bool:
        """Return True if the named collection exists on the server."""
        url = self.config.url + _COLLECTION_EXISTS.format(name=index_name)
        try:
            resp = self._client.get(url)
        except httpx.HTTPError as exc:
            raise ConnectionError(
                f"Cannot reach vector-DB server at {self.config.url}"
            ) from exc
        _raise_for_status(resp)
        return bool(resp.json().get("exists", False))

    def create_collection(
        self,
        index_name: str,
        backend: str,
        dim: int,
        storage_config: Optional[dict] = None,
    ) -> None:
        """Create a new collection on the server (no-op if already exists)."""
        payload: Dict[str, Any] = {
            "index_name": index_name,
            "backend": backend,
            "dim": dim,
        }
        if storage_config:
            payload["storage_config"] = storage_config
        self._post_json(_COLLECTION_CREATE, payload)

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def upsert(self, index_name: str, points: List[StoredPoint]) -> None:
        """Upsert a list of StoredPoints to the named collection.

        Tensors are serialised via torch.save for efficient binary transport.
        """
        # Convert to a serialisable list of dicts
        wire = [
            {
                "point_id": sp.point_id,
                "vector": sp.vector.cpu(),
                "payload": sp.payload,
            }
            for sp in points
        ]
        data = _dumps(wire)
        url = self.config.url + _UPSERT.format(name=index_name)
        try:
            resp = self._client.post(
                url,
                content=data,
                headers={"Content-Type": "application/octet-stream"},
            )
        except httpx.TimeoutException as exc:
            raise TimeoutError(
                f"Vector-DB server timed out after {self.config.request_timeout}s"
            ) from exc
        except httpx.HTTPError as exc:
            raise ConnectionError(
                f"Cannot reach vector-DB server at {self.config.url}"
            ) from exc
        _raise_for_status(resp)

    def point_exists(self, index_name: str, point_id: int) -> bool:
        """Return True if the given point exists in the named collection."""
        url = self.config.url + _POINT_EXISTS.format(
            name=index_name, point_id=point_id
        )
        try:
            resp = self._client.get(url)
        except httpx.HTTPError as exc:
            raise ConnectionError(
                f"Cannot reach vector-DB server at {self.config.url}"
            ) from exc
        _raise_for_status(resp)
        return bool(resp.json().get("exists", False))

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def search(
        self,
        index_name: str,
        query: MultiVectorQuery,
        k: int,
    ) -> List[SearchHit]:
        """Execute a nearest-neighbour search and return up to k hits.

        Query tensor and optional filter metadata are serialised via torch.save.
        Results are deserialised from the same binary format.
        """
        wire = {
            "vectors": query.vectors.cpu(),
            "filter_metadata": query.filter_metadata,
            "k": k,
        }
        data = _dumps(wire)
        url = self.config.url + _SEARCH.format(name=index_name)
        try:
            resp = self._client.post(
                url,
                content=data,
                headers={"Content-Type": "application/octet-stream"},
            )
        except httpx.TimeoutException as exc:
            raise TimeoutError(
                f"Vector-DB server timed out after {self.config.request_timeout}s"
            ) from exc
        except httpx.HTTPError as exc:
            raise ConnectionError(
                f"Cannot reach vector-DB server at {self.config.url}"
            ) from exc
        _raise_for_status(resp)
        raw_hits: List[dict] = _loads(resp.content)
        return [
            SearchHit(
                point_id=h["point_id"],
                score=float(h["score"]),
                payload=h["payload"],
            )
            for h in raw_hits
        ]

    def fetch_vector(
        self, index_name: str, point_id: int
    ) -> Optional[torch.Tensor]:
        """Retrieve the full multi-vector tensor for a given point.

        Returns None if the point does not exist.
        """
        url = self.config.url + _VECTOR.format(
            name=index_name, point_id=point_id
        )
        try:
            resp = self._client.get(url)
        except httpx.HTTPError as exc:
            raise ConnectionError(
                f"Cannot reach vector-DB server at {self.config.url}"
            ) from exc
        if resp.status_code == 404:
            return None
        _raise_for_status(resp)
        return _loads(resp.content)

    # ------------------------------------------------------------------
    # Admin
    # ------------------------------------------------------------------

    def delete_collection(self, index_name: str) -> None:
        """Delete a collection from the server."""
        url = self.config.url + _COLLECTION_DELETE.format(name=index_name)
        try:
            resp = self._client.delete(url)
        except httpx.HTTPError as exc:
            raise ConnectionError(
                f"Cannot reach vector-DB server at {self.config.url}"
            ) from exc
        _raise_for_status(resp)

    def list_indexes(self) -> dict:
        """Return the server's list of index directories under ``data_dir``.

        The returned payload has the shape
        ``{"items": [{"name": str, "path": str, "size_bytes": int,
        "n_files": int, "modified": float, "has_collection": bool,
        "backend": Optional[str]}, ...], "data_dir": str, "count": int}``.
        """
        try:
            resp = self._client.get(self.config.url + _ADMIN_INDEXES)
        except httpx.HTTPError as exc:
            raise ConnectionError(
                f"Cannot reach vector-DB server at {self.config.url}"
            ) from exc
        _raise_for_status(resp)
        return resp.json()

    def list_data_folders(self) -> dict:
        """Return every direct subdirectory under ``data_dir`` on the server.

        Each item carries ``is_index`` so the caller can filter client-side.
        Same envelope as :py:meth:`list_indexes`.
        """
        try:
            resp = self._client.get(self.config.url + _ADMIN_DATA_FOLDERS)
        except httpx.HTTPError as exc:
            raise ConnectionError(
                f"Cannot reach vector-DB server at {self.config.url}"
            ) from exc
        _raise_for_status(resp)
        return resp.json()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close the underlying HTTP client."""
        self._client.close()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _post_json(self, path: str, payload: dict) -> dict:
        """POST JSON to the given path and return parsed JSON response."""
        try:
            resp = self._client.post(self.config.url + path, json=payload)
        except httpx.TimeoutException as exc:
            raise TimeoutError(
                f"Vector-DB server timed out after {self.config.request_timeout}s"
            ) from exc
        except httpx.HTTPError as exc:
            raise ConnectionError(
                f"Cannot reach vector-DB server at {self.config.url}"
            ) from exc
        _raise_for_status(resp)
        return resp.json()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _raise_for_status(resp: httpx.Response) -> None:
    """Raise RuntimeError with message if the response is not 2xx."""
    if resp.status_code >= 400:
        try:
            detail = resp.json().get("detail", resp.text[:500])
        except Exception:
            detail = resp.text[:500]
        raise RuntimeError(
            f"Vector-DB server returned HTTP {resp.status_code}: {detail}"
        )
