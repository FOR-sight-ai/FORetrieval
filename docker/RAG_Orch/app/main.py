import httpx
from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from app.config import get_settings, get_vector_db_backend
from app.embeddings import call_embedding_api_for_query, check_embedding_api_health
from app.indexing import index_documents_service
from app.progress import get_index_progress
from app.schemas import (
    HealthResponse,
    IndexProgressResponse,
    IndexingResponse,
    RetrievalHit,
    RetrievalRequest,
    RetrievalResponse,
)
from app.vector_store import check_vector_db_health, search_points


app = FastAPI(title="RAG Orchestrator API", version="0.2.0")


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    # Report the health of the embedding service and the configured vector database backend.
    embedding_status = await check_embedding_api_health()
    vector_db_status = check_vector_db_health()
    status = "ok" if embedding_status == "ok" and vector_db_status == "ok" else "degraded"
    return HealthResponse(
        status=status,
        embedding_api=embedding_status,
        vector_db_backend=get_vector_db_backend(),
        vector_db=vector_db_status,
    )


@app.post("/v1/index", response_model=IndexingResponse)
async def index_documents(
    files: list[UploadFile] = File(...),
    collection_name: str | None = Form(default=None),
    machine_modele: str = Form(default="inconnu"),
    dpi: int = Form(default=get_settings().default_pdf_dpi),
    recreate_collection: bool = Form(default=False),
    progress_id: str | None = Form(default=None),
) -> IndexingResponse:
    # Index uploaded PDF documents into the currently selected vector backend.
    return await index_documents_service(
        files=files,
        collection_name=collection_name,
        machine_modele=machine_modele,
        dpi=dpi,
        recreate_collection=recreate_collection,
        progress_id=progress_id,
    )


@app.get("/v1/index/progress/{progress_id}", response_model=IndexProgressResponse)
async def get_index_progress_endpoint(progress_id: str) -> IndexProgressResponse:
    # Return the latest progress snapshot for a given indexing job.
    progress = get_index_progress(progress_id)
    if progress is None:
        raise HTTPException(status_code=404, detail="Unknown progress_id.")
    return progress


@app.post("/v1/retrieve", response_model=RetrievalResponse)
async def retrieve(payload: RetrievalRequest) -> RetrievalResponse:
    # Embed the query, search the backend, and normalize the results into the public API model.
    try:
        query_embedding = await call_embedding_api_for_query(payload.query)
        results = search_points(
            collection_name=payload.collection_name,
            query_vectors=query_embedding["vectors"],
            limit=payload.limit,
        )
    except httpx.HTTPStatusError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Embedding API error: {exc.response.text}",
        ) from exc
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return RetrievalResponse(
        collection_name=payload.collection_name,
        query=payload.query,
        limit=payload.limit,
        hits=[
            RetrievalHit(
                point_id=result.point_id,
                score=result.score,
                payload=result.payload,
            )
            for result in results
        ],
    )
