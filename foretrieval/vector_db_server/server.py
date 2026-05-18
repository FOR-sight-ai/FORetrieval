"""FORetrieval vector-DB server — FastAPI application.

Entry point:
    uvicorn foretrieval.vector_db_server.server:app --host 0.0.0.0 --port 18000

Environment variables:
    FOR_DB_DATA_DIR   Root directory where collections are persisted (default: /data).
    FOR_DB_API_KEY    Optional bearer token. When set, all requests must include
                      ``Authorization: Bearer <key>``.
    FOR_DB_HOST       Bind host (default: 0.0.0.0)   — used by server_main.py only.
    FOR_DB_PORT       Bind port (default: 18000)       — used by server_main.py only.

Each named collection is backed by one of the three local VectorStore
implementations (local / qdrant / milvus).  The backend choice is fixed when
the collection is first created and stored in
``<data_dir>/<index_name>/index.json``.

Concurrency:
    A per-collection asyncio.Lock serialises all operations on the same index.
    This avoids issues with Qdrant-embedded's single-client-per-path constraint
    and Milvus Lite's internal thread safety.
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse

from ..vector_store.base import MultiVectorQuery, StoredPoint
from ..vector_store.factory import make_vector_store

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration from environment
# ---------------------------------------------------------------------------

_DATA_DIR = Path(os.environ.get("FOR_DB_DATA_DIR", "/data"))
_API_KEY: Optional[str] = os.environ.get("FOR_DB_API_KEY") or None

# ---------------------------------------------------------------------------
# Server-side MAX_SIM scorer (used when backend="local")
# ---------------------------------------------------------------------------

class _TorchScorer:
    """Minimal scorer for LocalVectorStore.set_processor() on the server side.

    LocalVectorStore needs a processor with a .score(queries, docs) method.
    On the server there is no ColPali model, so we compute MAX_SIM directly
    using pytorch — identical numerics to colpali_engine.
    """

    def score(
        self,
        queries: list,           # list of 1 tensor (n_q_tokens, dim)
        docs: list,              # list of tensors (n_tokens, dim)
    ) -> torch.Tensor:
        """Compute MAX_SIM scores: shape (1, n_docs)."""
        q = queries[0].float()   # (n_q, dim)
        results = []
        for d in docs:
            d_f = d.float()      # (n_d, dim)
            # Late-interaction: (n_q, n_d) → max over doc tokens → sum over query tokens
            sim = torch.matmul(q, d_f.T)   # (n_q, n_d)
            score = sim.max(dim=1).values.sum()
            results.append(score.item())
        return torch.tensor([results])   # (1, n_docs)


_TORCH_SCORER = _TorchScorer()

# ---------------------------------------------------------------------------
# Tensor codec (mirrors client.py — must stay in sync)
# ---------------------------------------------------------------------------


def _dumps(obj: Any) -> bytes:
    buf = io.BytesIO()
    torch.save(obj, buf)
    return buf.getvalue()


def _loads(data: bytes) -> Any:
    return torch.load(io.BytesIO(data), map_location="cpu", weights_only=False)


# ---------------------------------------------------------------------------
# Index registry
# ---------------------------------------------------------------------------

# Mapping index_name → VectorStore instance (opened, ready to use)
_registry: Dict[str, Any] = {}

# Per-collection locks: index_name → asyncio.Lock
_locks: Dict[str, asyncio.Lock] = {}


def _get_lock(index_name: str) -> asyncio.Lock:
    if index_name not in _locks:
        _locks[index_name] = asyncio.Lock()
    return _locks[index_name]


def _meta_path(index_name: str) -> Path:
    """Return the path to the collection metadata JSON file."""
    return _DATA_DIR / index_name / "index.json"


def _read_meta(index_name: str) -> Optional[Dict[str, Any]]:
    p = _meta_path(index_name)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def _write_meta(index_name: str, meta: Dict[str, Any]) -> None:
    p = _meta_path(index_name)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(meta))


def _inject_scorer(vs: Any) -> None:
    """Inject the server-side MAX_SIM scorer into a LocalVectorStore."""
    from ..vector_store.local import LocalVectorStore
    if isinstance(vs, LocalVectorStore):
        vs.set_processor(_TORCH_SCORER)


def _open_store(index_name: str, backend: str, storage_config: Optional[dict]) -> Any:
    """Instantiate and open a VectorStore for an existing collection."""
    vs = make_vector_store(backend, storage_config)
    vs.open(index_name, _DATA_DIR, create=False)
    vs.load_sidecar(_DATA_DIR / index_name)
    _inject_scorer(vs)
    return vs


def _get_or_load(index_name: str) -> Any:
    """Return the VectorStore for an existing, previously-opened collection.

    Raises HTTPException 404 if the collection doesn't exist.
    Raises HTTPException 409 if the collection exists on disk but was never
    opened in this process (auto-reloads it from disk).
    """
    if index_name in _registry:
        return _registry[index_name]

    # Try to reload from disk (e.g. after a server restart)
    meta = _read_meta(index_name)
    if meta is None:
        raise HTTPException(
            status_code=404,
            detail=f"Collection '{index_name}' does not exist on this server.",
        )
    backend = meta["backend"]
    storage_config = meta.get("storage_config")
    vs = _open_store(index_name, backend, storage_config)
    _registry[index_name] = vs
    logger.info(
        "Auto-reloaded collection '%s' (backend=%s) from disk.", index_name, backend
    )
    return vs


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(title="FORetrieval Vector-DB Server", version="0.1.0")


# ------------------------------------------------------------------
# Auth middleware
# ------------------------------------------------------------------

@app.middleware("http")
async def _auth_middleware(request: Request, call_next):
    if _API_KEY is not None and request.url.path != "/health":
        auth = request.headers.get("authorization", "")
        if not auth.lower().startswith("bearer "):
            return JSONResponse({"detail": "Missing Authorization header"}, status_code=401)
        token = auth.split(" ", 1)[1].strip()
        if token != _API_KEY:
            return JSONResponse({"detail": "Invalid API key"}, status_code=401)
    return await call_next(request)


# ------------------------------------------------------------------
# Health
# ------------------------------------------------------------------

@app.get("/health")
async def health():
    """Liveness check — always returns 200 OK."""
    return {"status": "ok"}


# ------------------------------------------------------------------
# Collection management
# ------------------------------------------------------------------

@app.post("/v1/collection/open")
async def open_collection(request: Request):
    """Open or create a named collection.

    Request JSON:
        index_name (str): Collection name.
        backend    (str): "local" | "qdrant" | "milvus".
        create     (bool): If True, create if not exists.
        dim        (int, optional): Embedding dimension (required when create=True
                                    and collection does not yet exist).
        storage_config (dict, optional): Backend-specific config (e.g. candidate_limit).
    """
    body = await request.json()
    index_name: str = body["index_name"]
    backend: str = body.get("backend", "qdrant")
    create: bool = body.get("create", False)
    dim: Optional[int] = body.get("dim")
    storage_config: Optional[dict] = body.get("storage_config")

    async with _get_lock(index_name):
        meta = _read_meta(index_name)
        already_exists = meta is not None

        if already_exists:
            # Ensure in-memory registry is populated
            _get_or_load(index_name)
            return JSONResponse({
                "opened": True,
                "backend": meta["backend"],
                "created": False,
            })

        if not create:
            raise HTTPException(
                status_code=404,
                detail=(
                    f"Collection '{index_name}' does not exist. "
                    "Pass create=true to create it."
                ),
            )

        # Create new collection
        vs = make_vector_store(backend, storage_config)
        vs.open(index_name, _DATA_DIR, create=True, dim=dim)
        if dim is not None:
            if not vs.collection_exists():
                vs.create_collection(dim)
        _inject_scorer(vs)
        _registry[index_name] = vs

        meta = {"backend": backend, "storage_config": storage_config}
        _write_meta(index_name, meta)

        logger.info(
            "Created collection '%s' (backend=%s, dim=%s).",
            index_name, backend, dim,
        )
        return JSONResponse({"opened": True, "backend": backend, "created": True})


@app.get("/v1/collection/{name}/exists")
async def collection_exists(name: str):
    """Check whether a named collection exists on this server."""
    meta = _read_meta(name)
    if meta is None:
        return {"exists": False, "backend": None}
    return {"exists": True, "backend": meta.get("backend")}


@app.post("/v1/collection")
async def create_collection(request: Request):
    """Create a new collection (no-op if already exists).

    Request JSON:
        index_name (str), backend (str), dim (int),
        storage_config (dict, optional).
    """
    body = await request.json()
    index_name: str = body["index_name"]
    backend: str = body.get("backend", "qdrant")
    dim: int = body["dim"]
    storage_config: Optional[dict] = body.get("storage_config")

    async with _get_lock(index_name):
        meta = _read_meta(index_name)
        if meta is not None:
            return JSONResponse({"created": False, "detail": "already exists"})

        vs = make_vector_store(backend, storage_config)
        vs.open(index_name, _DATA_DIR, create=True, dim=dim)
        vs.create_collection(dim)
        _inject_scorer(vs)
        _registry[index_name] = vs

        _write_meta(index_name, {"backend": backend, "storage_config": storage_config})
        logger.info("Created collection '%s' backend=%s dim=%d.", index_name, backend, dim)
        return JSONResponse({"created": True})


@app.delete("/v1/collection/{name}")
async def delete_collection(name: str):
    """Remove a collection from the registry and delete its data directory."""
    import shutil

    async with _get_lock(name):
        vs = _registry.pop(name, None)
        if vs is not None:
            try:
                vs.close()
            except Exception:
                pass

        coll_dir = _DATA_DIR / name
        if coll_dir.exists():
            shutil.rmtree(coll_dir)
            logger.info("Deleted collection directory '%s'.", coll_dir)

        _locks.pop(name, None)
        return {"deleted": True}


# ------------------------------------------------------------------
# Write
# ------------------------------------------------------------------

@app.post("/v1/upsert/{name}")
async def upsert(name: str, request: Request):
    """Upsert a list of StoredPoints (torch.save bytes body).

    The request body is a torch.save'd list of dicts with keys:
        point_id (int), vector (Tensor), payload (dict).
    """
    data = await request.body()
    raw_points = _loads(data)

    points = [
        StoredPoint(
            point_id=int(p["point_id"]),
            vector=p["vector"].cpu(),
            payload=p["payload"],
        )
        for p in raw_points
    ]

    async with _get_lock(name):
        vs = _get_or_load(name)

        # Lazy collection creation: create on first upsert if not yet created
        dim = points[0].vector.shape[-1] if points else None
        if dim is not None and not vs.collection_exists():
            meta = _read_meta(name)
            backend = meta["backend"] if meta else "local"
            vs.create_collection(dim)
            _inject_scorer(vs)
            logger.info(
                "Lazily created collection '%s' (backend=%s, dim=%d).",
                name, backend, dim,
            )

        vs.upsert(points)
        vs.export_sidecar(_DATA_DIR / name)

    return {"upserted": len(points)}


@app.get("/v1/point/{name}/{point_id}/exists")
async def point_exists(name: str, point_id: int):
    """Check if a point exists in the named collection."""
    async with _get_lock(name):
        vs = _get_or_load(name)
        exists = vs.point_exists(point_id)
    return {"exists": exists}


# ------------------------------------------------------------------
# Read
# ------------------------------------------------------------------

@app.post("/v1/search/{name}")
async def search(name: str, request: Request):
    """Execute a nearest-neighbour search (torch.save bytes body).

    The request body is a torch.save'd dict with keys:
        vectors (Tensor), filter_metadata (dict | None), k (int).

    Returns a torch.save'd list of dicts:
        [{point_id, score, payload}, ...]
    """
    data = await request.body()
    wire = _loads(data)

    query = MultiVectorQuery(
        vectors=wire["vectors"].cpu(),
        filter_metadata=wire.get("filter_metadata"),
    )
    k: int = int(wire.get("k", 10))

    async with _get_lock(name):
        vs = _get_or_load(name)
        hits = vs.search(query, k)

    result = [
        {"point_id": h.point_id, "score": h.score, "payload": h.payload}
        for h in hits
    ]
    return Response(content=_dumps(result), media_type="application/octet-stream")


@app.get("/v1/vector/{name}/{point_id}")
async def fetch_vector(name: str, point_id: int):
    """Retrieve the full multi-vector tensor for a point.

    Returns 404 if the point does not exist.
    Response body is torch.save'd tensor.
    """
    async with _get_lock(name):
        vs = _get_or_load(name)
        tensor = vs.fetch_vector(point_id)

    if tensor is None:
        raise HTTPException(
            status_code=404,
            detail=f"Point {point_id} not found in collection '{name}'.",
        )
    return Response(content=_dumps(tensor), media_type="application/octet-stream")
