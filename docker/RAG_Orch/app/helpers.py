import json
import math
import re
import uuid
from pathlib import Path

import fitz
from fastapi import UploadFile

from app.config import (
    MILVUS_COLLECTION_SUFFIX_PAGES,
    MILVUS_COLLECTION_SUFFIX_TOKENS,
    PAYLOAD_JSON_MAX_LENGTH,
)


def sanitize_identifier(value: str, fallback: str) -> str:
    # Normalize an arbitrary name into a safe identifier fragment for documents and collections.
    cleaned = re.sub(r"[^a-zA-Z0-9_-]+", "-", value.strip().lower()).strip("-")
    return cleaned or fallback


def default_collection_name(files: list[UploadFile]) -> str:
    # Build the default logical collection name from the uploaded file list.
    if len(files) == 1 and files[0].filename:
        return f"{Path(files[0].filename).stem}_colqwen35"
    return "corpus_colqwen35"


def extract_page_text(page: fitz.Page) -> str:
    # Extract and compact textual content from a PDF page for payload storage.
    text = page.get_text("text")
    return re.sub(r"\s+", " ", text).strip()


def page_identifier(document_id: str, page_number: int) -> str:
    # Generate a deterministic page identifier so Milvus page and token rows stay linked.
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{document_id}:{page_number}"))


def qdrant_point_id() -> str:
    # Generate a random point identifier for a Qdrant record.
    return str(uuid.uuid4())


def milvus_safe_collection_base_name(logical_name: str) -> str:
    # Convert a logical collection name into a Milvus-compatible collection prefix.
    cleaned = re.sub(r"[^a-zA-Z0-9_]+", "_", logical_name.strip().lower()).strip("_")
    cleaned = cleaned or "corpus_colqwen35"
    if cleaned[0].isdigit():
        cleaned = f"c_{cleaned}"
    return cleaned[:180]


def milvus_collection_names(logical_name: str) -> tuple[str, str]:
    # Derive the page and token collection names used by the Milvus late-interaction layout.
    base_name = milvus_safe_collection_base_name(logical_name)
    return (
        f"{base_name}{MILVUS_COLLECTION_SUFFIX_PAGES}",
        f"{base_name}{MILVUS_COLLECTION_SUFFIX_TOKENS}",
    )


def serialize_payload(payload: dict[str, object]) -> str:
    # Serialize payload data into a bounded JSON string accepted by the Milvus schema.
    serialized = json.dumps(payload, ensure_ascii=False)
    if len(serialized) > PAYLOAD_JSON_MAX_LENGTH:
        serialized = json.dumps(
            {
                **payload,
                "chunk_text": str(payload.get("chunk_text", ""))[:60000],
                "chunk_preview": str(payload.get("chunk_preview", ""))[:500],
                "_payload_truncated": True,
            },
            ensure_ascii=False,
        )
        if len(serialized) > PAYLOAD_JSON_MAX_LENGTH:
            serialized = serialized[:PAYLOAD_JSON_MAX_LENGTH]
    return serialized


def deserialize_payload(payload_json: str | None) -> dict[str, object]:
    # Decode a Milvus payload JSON string back into a Python dictionary.
    if not payload_json:
        return {}
    try:
        payload = json.loads(payload_json)
        if isinstance(payload, dict):
            return payload
    except json.JSONDecodeError:
        pass
    return {"payload_json": payload_json}


def mean_pool_vectors(vectors: list[list[float]]) -> list[float]:
    # Compute a normalized mean-pooled vector used for the Milvus page-level candidate search.
    if not vectors:
        raise ValueError("Cannot pool an empty vector list.")

    vector_size = len(vectors[0])
    sums = [0.0] * vector_size
    for vector in vectors:
        if len(vector) != vector_size:
            raise ValueError("Inconsistent vector sizes in embedding payload.")
        for index, value in enumerate(vector):
            sums[index] += float(value)

    pooled = [value / len(vectors) for value in sums]
    norm = math.sqrt(sum(value * value for value in pooled))
    if norm > 0:
        return [value / norm for value in pooled]
    return pooled


def milvus_quote_string(value: str) -> str:
    # Escape and quote a string value so it can be embedded safely in a Milvus filter expression.
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'
