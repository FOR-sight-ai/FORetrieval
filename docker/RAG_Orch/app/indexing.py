from pathlib import Path
from typing import Any

import fitz
import httpx
from fastapi import HTTPException, UploadFile
from PIL import Image

from app.config import get_settings, get_vector_db_backend
from app.embeddings import call_embedding_api_for_page
from app.helpers import default_collection_name, extract_page_text, sanitize_identifier
from app.progress import set_index_progress
from app.schemas import IndexingFileResult, IndexingResponse
from app.vector_store import (
    PendingIndexBatch,
    add_index_record,
    ensure_collection,
    flush_index_batch,
    pending_page_count,
)


def build_page_payload(
    document_id: str,
    filename: str,
    page_number: int,
    machine_modele: str,
    dpi: int,
    embedding: dict[str, Any],
    page_text: str,
) -> dict[str, Any]:
    # Assemble the payload stored next to each indexed page.
    return {
        "document_id": document_id,
        "filename": filename,
        "page_number": page_number,
        "machine_modele": machine_modele,
        "dpi": dpi,
        "model_name": embedding["model_name"],
        "chunk_text": page_text,
        "chunk_preview": page_text[:500],
        **embedding.get("metadata", {}),
    }


def initialize_progress(
    progress_id: str | None,
    collection_name: str,
    total_files: int,
) -> None:
    # Seed the progress tracker with the initial state of a new indexing request.
    if progress_id:
        set_index_progress(
            progress_id,
            status="preparing",
            collection_name=collection_name,
            total_files=total_files,
            files_processed=0,
            pages_indexed=0,
            total_pages_estimate=0,
            points_upserted=0,
            error=None,
        )


def update_file_start_progress(
    progress_id: str | None,
    filename: str,
    file_page_count: int,
    total_pages_estimate: int,
) -> None:
    # Update the progress tracker when a new file starts indexing.
    if progress_id:
        set_index_progress(
            progress_id,
            status="running",
            current_filename=filename,
            current_page=0,
            current_file_page_count=file_page_count,
            total_pages_estimate=total_pages_estimate,
        )


def update_page_progress(
    progress_id: str | None,
    filename: str,
    page_number: int,
    file_page_count: int,
    file_index: int,
    total_pages_indexed: int,
    total_pages_estimate: int,
    points_upserted: int,
) -> None:
    # Update the progress tracker after one page has been prepared locally.
    if progress_id:
        set_index_progress(
            progress_id,
            status="running",
            current_filename=filename,
            current_page=page_number,
            current_file_page_count=file_page_count,
            files_processed=file_index - 1,
            pages_indexed=total_pages_indexed,
            total_pages_estimate=total_pages_estimate,
            points_upserted=points_upserted,
        )


def update_flush_progress(
    progress_id: str | None,
    filename: str,
    page_number: int,
    file_page_count: int,
    file_index: int,
    total_pages_indexed: int,
    total_pages_estimate: int,
    points_upserted: int,
) -> None:
    # Update the progress tracker after a backend flush has completed.
    if progress_id:
        set_index_progress(
            progress_id,
            status="running",
            current_filename=filename,
            current_page=page_number,
            current_file_page_count=file_page_count,
            files_processed=file_index - 1,
            pages_indexed=total_pages_indexed,
            total_pages_estimate=total_pages_estimate,
            points_upserted=points_upserted,
        )


def update_file_done_progress(
    progress_id: str | None,
    filename: str,
    file_page_count: int,
    file_index: int,
    total_pages_indexed: int,
    total_pages_estimate: int,
    points_upserted: int,
) -> None:
    # Update the progress tracker once a file has been fully processed.
    if progress_id:
        set_index_progress(
            progress_id,
            status="running",
            current_filename=filename,
            current_page=file_page_count,
            current_file_page_count=file_page_count,
            files_processed=file_index,
            pages_indexed=total_pages_indexed,
            total_pages_estimate=total_pages_estimate,
            points_upserted=points_upserted,
        )


def update_failure_progress(progress_id: str | None, error: str) -> None:
    # Mark the current indexing request as failed in the progress tracker.
    if progress_id:
        set_index_progress(progress_id, status="failed", error=error)


def update_completion_progress(
    progress_id: str | None,
    total_files: int,
    total_pages_indexed: int,
    total_pages_estimate: int,
    points_upserted: int,
) -> None:
    # Mark the current indexing request as completed in the progress tracker.
    if progress_id:
        set_index_progress(
            progress_id,
            status="completed",
            current_page=0,
            current_file_page_count=0,
            files_processed=total_files,
            pages_indexed=total_pages_indexed,
            total_pages_estimate=total_pages_estimate,
            points_upserted=points_upserted,
        )


async def index_documents_service(
    files: list[UploadFile],
    collection_name: str | None,
    machine_modele: str,
    dpi: int | None,
    recreate_collection: bool,
    progress_id: str | None,
) -> IndexingResponse:
    # Index the uploaded PDF files into the configured vector backend and return the API response.
    if not files:
        raise HTTPException(status_code=400, detail="At least one PDF file is required.")

    backend = get_vector_db_backend()
    effective_dpi = dpi or get_settings().default_pdf_dpi
    resolved_collection_name = collection_name or default_collection_name(files)
    file_results: list[IndexingFileResult] = []
    pending_batch = PendingIndexBatch()
    total_pages_indexed = 0
    points_upserted = 0
    recreate_pending = recreate_collection
    collection_ready = False
    total_pages_estimate = 0

    initialize_progress(progress_id, resolved_collection_name, len(files))

    for file_index, upload in enumerate(files, start=1):
        filename = upload.filename or f"document-{file_index}.pdf"
        if not filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail=f"{filename} is not a PDF file.")

        file_bytes = await upload.read()
        document_id = f"{file_index}-{sanitize_identifier(Path(filename).stem, 'document')}"

        try:
            document = fitz.open(stream=file_bytes, filetype="pdf")
        except Exception as exc:
            raise HTTPException(
                status_code=400,
                detail=f"Unable to open {filename} as a PDF: {exc}",
            ) from exc

        file_page_count = len(document)
        total_pages_estimate += file_page_count
        pages_indexed = 0
        update_file_start_progress(
            progress_id,
            filename,
            file_page_count,
            total_pages_estimate,
        )

        try:
            for page_offset in range(file_page_count):
                page_number = page_offset + 1
                page = document[page_offset]
                pix = page.get_pixmap(dpi=effective_dpi)
                image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                page_text = extract_page_text(page)

                embedding = await call_embedding_api_for_page(
                    image=image,
                    filename=f"{Path(filename).stem}-page-{page_number}.png",
                )
                vectors = embedding["vectors"]
                if not vectors:
                    raise HTTPException(
                        status_code=500,
                        detail=f"Embedding API returned no vectors for {filename} page {page_number}.",
                    )

                if not collection_ready:
                    ensure_collection(
                        collection_name=resolved_collection_name,
                        vector_size=len(vectors[0]),
                        recreate=recreate_pending,
                    )
                    collection_ready = True
                    recreate_pending = False

                payload = build_page_payload(
                    document_id=document_id,
                    filename=filename,
                    page_number=page_number,
                    machine_modele=machine_modele,
                    dpi=effective_dpi,
                    embedding=embedding,
                    page_text=page_text,
                )
                add_index_record(
                    batch=pending_batch,
                    backend=backend,
                    document_id=document_id,
                    page_number=page_number,
                    vectors=vectors,
                    payload=payload,
                )

                pages_indexed += 1
                total_pages_indexed += 1

                update_page_progress(
                    progress_id=progress_id,
                    filename=filename,
                    page_number=page_number,
                    file_page_count=file_page_count,
                    file_index=file_index,
                    total_pages_indexed=total_pages_indexed,
                    total_pages_estimate=total_pages_estimate,
                    points_upserted=points_upserted + pending_page_count(pending_batch, backend),
                )

                if pending_page_count(pending_batch, backend) >= get_settings().upsert_batch_size:
                    points_upserted += flush_index_batch(
                        resolved_collection_name,
                        pending_batch,
                        backend,
                    )
                    update_flush_progress(
                        progress_id=progress_id,
                        filename=filename,
                        page_number=page_number,
                        file_page_count=file_page_count,
                        file_index=file_index,
                        total_pages_indexed=total_pages_indexed,
                        total_pages_estimate=total_pages_estimate,
                        points_upserted=points_upserted,
                    )
        except httpx.HTTPStatusError as exc:
            error_message = f"Embedding API error while indexing {filename}: {exc.response.text}"
            update_failure_progress(progress_id, error_message)
            raise HTTPException(status_code=502, detail=error_message) from exc
        except Exception as exc:
            update_failure_progress(progress_id, str(exc))
            raise
        finally:
            document.close()

        file_results.append(
            IndexingFileResult(
                filename=filename,
                document_id=document_id,
                pages_indexed=pages_indexed,
            )
        )
        update_file_done_progress(
            progress_id=progress_id,
            filename=filename,
            file_page_count=file_page_count,
            file_index=file_index,
            total_pages_indexed=total_pages_indexed,
            total_pages_estimate=total_pages_estimate,
            points_upserted=points_upserted + pending_page_count(pending_batch, backend),
        )

    points_upserted += flush_index_batch(resolved_collection_name, pending_batch, backend)
    update_completion_progress(
        progress_id=progress_id,
        total_files=len(files),
        total_pages_indexed=total_pages_indexed,
        total_pages_estimate=total_pages_estimate,
        points_upserted=points_upserted,
    )

    return IndexingResponse(
        collection_name=resolved_collection_name,
        files=file_results,
        total_pages_indexed=total_pages_indexed,
        points_upserted=points_upserted,
    )
