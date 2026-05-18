"""LocalVectorStore — in-memory multi-vector store with .pt/.json.gz persistence.

This is a direct extraction of the "local" code-path from ColPaliModel into the
VectorStore interface.  All logic that previously lived in _search_local,
_load_local_index, and _export_index (local branch) is now here.

Storage layout on disk (mirrored from the existing format so old indexes remain
compatible):

    <index_root>/<index_name>/
        embeddings/
            embeddings_0.pt       — list[Tensor], up to 500 per file
            embeddings_500.pt
            …
        embed_id_to_doc_id.json.gz  — {str(embed_id): {"doc_id": int, "page_id": int, …}}

The embed_id is simply the zero-based position of the point in the flat
indexed_embeddings list.  This is intentionally different from the integer
point_id produced by make_point_id() — for backward-compatibility the local
backend keeps its own sequential embed_id scheme internally, but exports a
point_id in SearchHit.point_id by constructing it via make_point_id so that
ColPaliModel can look up embed_id_to_extra consistently.

For that reason LocalVectorStore also maintains an internal embed_id → point_id
and point_id → embed_id mapping so callers that only know the point_id can still
fetch vectors (used by the heatmap path).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional

import srsly
import torch

from .base import (
    MultiVectorQuery,
    SearchHit,
    StoredPoint,
    VectorStore,
    make_point_id,
)
from ..utils import _value_match

logger = logging.getLogger(__name__)


class LocalVectorStore(VectorStore):
    """In-process multi-vector store backed by .pt / .json.gz files.

    Scoring uses the full late-interaction MAX_SIM formula via
    processor.score(), which is injected at open() time so that the store
    remains independent of the ColPali processor class hierarchy.
    """

    backend_name: ClassVar[str] = "local"
    supports_multivector_native: ClassVar[bool] = True  # exact MAX_SIM

    def __init__(self) -> None:
        # Set by open()
        self._index_name: Optional[str] = None
        self._index_root: Optional[Path] = None
        self._processor: Any = None       # injected via set_processor()

        # In-memory state
        self._embeddings: List[torch.Tensor] = []           # shape (n_tokens, dim) each
        self._embed_id_to_doc_id: Dict[int, Dict[str, Any]] = {}

        # Reverse mapping: point_id → embed_id (built lazily on first use)
        self._point_id_to_embed_id: Dict[int, int] = {}

    # ------------------------------------------------------------------
    # Processor injection
    # ------------------------------------------------------------------

    def set_processor(self, processor: Any) -> None:
        """Inject the ColPali processor used for score()."""
        self._processor = processor

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def open(
        self,
        index_name: str,
        index_root: Path,
        *,
        create: bool,
        dim: Optional[int] = None,
    ) -> None:
        self._index_name = index_name
        self._index_root = Path(index_root)

    def close(self) -> None:
        pass  # nothing to release for the in-memory store

    # ------------------------------------------------------------------
    # Collection management
    # ------------------------------------------------------------------

    def collection_exists(self) -> bool:
        if self._index_name is None or self._index_root is None:
            return False
        embeddings_dir = self._index_root / self._index_name / "embeddings"
        return embeddings_dir.exists() and any(embeddings_dir.glob("*.pt"))

    def create_collection(self, dim: int) -> None:
        # No setup required — directories are created at export_sidecar() time.
        pass

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def upsert(self, points: List[StoredPoint]) -> None:
        for sp in points:
            if not self.point_exists(sp.point_id):
                embed_id = len(self._embeddings)
                self._embeddings.append(sp.vector.cpu())

                entry: Dict[str, Any] = {
                    "doc_id": int(sp.payload["doc_id"]),
                    "page_id": int(sp.payload["page_id"]),
                }
                chunk_id = sp.payload.get("chunk_id")
                if chunk_id is not None:
                    entry["chunk_id"] = int(chunk_id)

                self._embed_id_to_doc_id[embed_id] = entry
                self._point_id_to_embed_id[sp.point_id] = embed_id
            else:
                # Last-write-wins: update in place
                embed_id = self._point_id_to_embed_id[sp.point_id]
                self._embeddings[embed_id] = sp.vector.cpu()

    def point_exists(self, point_id: int) -> bool:
        return point_id in self._point_id_to_embed_id

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def search(
        self,
        query: MultiVectorQuery,
        k: int,
    ) -> List[SearchHit]:
        if not self._embeddings:
            return []

        if self._processor is None:
            raise RuntimeError(
                "LocalVectorStore.set_processor() must be called before search()."
            )

        filter_md = query.filter_metadata

        if filter_md:
            req_embeddings, req_embed_ids = self._filter_by_metadata(filter_md)
            if not req_embeddings:
                logger.warning(
                    "Metadata filter matched no documents — returning empty results."
                )
                return []
        else:
            req_embeddings = self._embeddings
            req_embed_ids = list(range(len(self._embeddings)))

        k = min(k, len(req_embeddings))
        qs = [query.vectors]
        scores = self._processor.score(qs, req_embeddings).cpu().numpy()
        top_local = scores.argsort(axis=1)[0][-k:][::-1].tolist()

        results = []
        for local_idx in top_local:
            embed_id = req_embed_ids[local_idx]
            doc_info = self._embed_id_to_doc_id[embed_id]
            pid = make_point_id(
                doc_info["doc_id"],
                doc_info["page_id"],
                doc_info.get("chunk_id"),
            )
            results.append(
                SearchHit(
                    point_id=pid,
                    score=float(scores[0][local_idx]),
                    payload=dict(doc_info),
                )
            )

        return results

    def fetch_vector(self, point_id: int) -> Optional[torch.Tensor]:
        embed_id = self._point_id_to_embed_id.get(point_id)
        if embed_id is None:
            return None
        return self._embeddings[embed_id].cpu()

    # ------------------------------------------------------------------
    # Metadata filtering helpers
    # ------------------------------------------------------------------

    def _filter_by_metadata(
        self,
        filter_md: Dict[str, Any],
    ) -> tuple[List[torch.Tensor], List[int]]:
        """Return (embeddings, embed_ids) matching the metadata filter."""
        # We need the full doc_id_to_metadata which is owned by ColPaliModel.
        # Inject it via set_doc_id_to_metadata().
        metadata_map = getattr(self, "_doc_id_to_metadata", {})

        from ..models_metadata import MetadataFilter
        f = (
            filter_md
            if isinstance(filter_md, MetadataFilter)
            else MetadataFilter(**filter_md)
        )

        matching_doc_ids = {
            int(did)
            for did, md in metadata_map.items()
            if _value_match(md, f)
        }

        req_embed_ids = [
            eid
            for eid, info in self._embed_id_to_doc_id.items()
            if int(info["doc_id"]) in matching_doc_ids
        ]
        req_embeddings = [self._embeddings[eid] for eid in req_embed_ids]
        return req_embeddings, req_embed_ids

    def set_doc_id_to_metadata(self, mapping: Dict[int, Any]) -> None:
        """Inject the doc_id→metadata mapping needed for filter_by_metadata."""
        self._doc_id_to_metadata = mapping

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def export_sidecar(self, index_path: Path) -> None:
        embeddings_dir = index_path / "embeddings"
        embeddings_dir.mkdir(exist_ok=True)

        chunk_size = 500
        for i in range(0, len(self._embeddings), chunk_size):
            chunk = self._embeddings[i : i + chunk_size]
            torch.save(chunk, embeddings_dir / f"embeddings_{i}.pt")

        srsly.write_gzip_json(
            index_path / "embed_id_to_doc_id.json.gz",
            self._embed_id_to_doc_id,
        )

    def load_sidecar(self, index_path: Path) -> None:
        embeddings_path = index_path / "embeddings"
        if not embeddings_path.exists():
            return

        embedding_files = sorted(
            embeddings_path.glob("embeddings_*.pt"),
            key=lambda x: int(x.stem.split("_")[1]),
        )
        self._embeddings = []
        for f in embedding_files:
            self._embeddings.extend(torch.load(f, map_location="cpu"))

        id_path = index_path / "embed_id_to_doc_id.json.gz"
        if id_path.exists():
            raw = srsly.read_gzip_json(id_path)
            self._embed_id_to_doc_id = {int(k): v for k, v in raw.items()}

        # Rebuild point_id → embed_id reverse map
        self._point_id_to_embed_id = {}
        for embed_id, info in self._embed_id_to_doc_id.items():
            pid = make_point_id(
                int(info["doc_id"]),
                int(info["page_id"]),
                info.get("chunk_id"),
            )
            self._point_id_to_embed_id[pid] = embed_id

    # ------------------------------------------------------------------
    # Accessors used by ColPaliModel (legacy compatibility)
    # ------------------------------------------------------------------

    @property
    def indexed_embeddings(self) -> List[torch.Tensor]:
        return self._embeddings

    @property
    def embed_id_to_doc_id(self) -> Dict[int, Dict[str, Any]]:
        return self._embed_id_to_doc_id
