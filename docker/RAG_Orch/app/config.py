import os
from dataclasses import dataclass
from functools import lru_cache


PAYLOAD_JSON_MAX_LENGTH = 65535
MILVUS_PRIMARY_KEY_MAX_LENGTH = 64
MILVUS_COLLECTION_SUFFIX_PAGES = "__pages"
MILVUS_COLLECTION_SUFFIX_TOKENS = "__tokens"
MILVUS_PAGE_VECTOR_FIELD = "page_vector"
MILVUS_TOKEN_VECTOR_FIELD = "token_vector"
MILVUS_TOKEN_GROUP_FIELD = "page_id"


@dataclass(frozen=True)
class Settings:
    embedding_service_url: str
    vector_db_backend: str
    qdrant_url: str
    milvus_url: str
    milvus_token: str | None
    default_pdf_dpi: int
    upsert_batch_size: int
    milvus_candidate_limit: int
    http_timeout_seconds: float


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    # Read and cache the runtime configuration exposed through environment variables.
    return Settings(
        embedding_service_url=os.getenv("EMBEDDING_SERVICE_URL", "http://localhost:8001"),
        vector_db_backend=os.getenv("VECTOR_DB_BACKEND", "milvus").strip().lower(),
        qdrant_url=os.getenv("QDRANT_URL", "http://localhost:6333"),
        milvus_url=os.getenv("MILVUS_URL", "http://localhost:19530"),
        milvus_token=os.getenv("MILVUS_TOKEN") or None,
        default_pdf_dpi=int(os.getenv("DEFAULT_PDF_DPI", "150")),
        upsert_batch_size=int(
            os.getenv(
                "VECTOR_DB_UPSERT_BATCH_SIZE",
                os.getenv("QDRANT_UPSERT_BATCH_SIZE", "16"),
            )
        ),
        milvus_candidate_limit=int(os.getenv("MILVUS_CANDIDATE_LIMIT", "64")),
        http_timeout_seconds=float(os.getenv("HTTP_TIMEOUT_SECONDS", "300")),
    )


def get_vector_db_backend() -> str:
    # Validate and return the configured vector database backend name.
    backend = get_settings().vector_db_backend
    if backend not in {"milvus", "qdrant"}:
        raise ValueError(
            f"Unsupported VECTOR_DB_BACKEND={backend!r}. Expected 'milvus' or 'qdrant'."
        )
    return backend
