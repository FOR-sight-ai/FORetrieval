from typing import Any

from pydantic import BaseModel, Field


class IndexingFileResult(BaseModel):
    filename: str
    document_id: str
    pages_indexed: int


class IndexingResponse(BaseModel):
    collection_name: str
    files: list[IndexingFileResult]
    total_pages_indexed: int
    points_upserted: int


class RetrievalRequest(BaseModel):
    collection_name: str
    query: str
    limit: int = Field(default=5, ge=1, le=100)


class RetrievalHit(BaseModel):
    point_id: str
    score: float
    payload: dict[str, Any] = Field(default_factory=dict)


class RetrievalResponse(BaseModel):
    collection_name: str
    query: str
    limit: int
    hits: list[RetrievalHit]


class HealthResponse(BaseModel):
    status: str
    embedding_api: str
    vector_db_backend: str
    vector_db: str


class IndexProgressResponse(BaseModel):
    progress_id: str
    status: str
    collection_name: str | None = None
    current_filename: str | None = None
    current_page: int = 0
    current_file_page_count: int = 0
    files_processed: int = 0
    total_files: int = 0
    pages_indexed: int = 0
    total_pages_estimate: int = 0
    points_upserted: int = 0
    started_at: float | None = None
    updated_at: float | None = None
    error: str | None = None
