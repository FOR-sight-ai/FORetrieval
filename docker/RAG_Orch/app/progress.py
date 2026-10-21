import threading
import time
from typing import Any

from app.schemas import IndexProgressResponse


INDEX_PROGRESS: dict[str, IndexProgressResponse] = {}
INDEX_PROGRESS_LOCK = threading.Lock()


def set_index_progress(progress_id: str, **updates: Any) -> None:
    # Merge new progress information into the in-memory tracker for an index job.
    now = time.time()
    with INDEX_PROGRESS_LOCK:
        current = INDEX_PROGRESS.get(progress_id)
        if current is None:
            current = IndexProgressResponse(
                progress_id=progress_id,
                status="pending",
                started_at=now,
                updated_at=now,
            )
        data = current.model_dump()
        data.update(updates)
        data["progress_id"] = progress_id
        data["updated_at"] = now
        if data.get("started_at") is None:
            data["started_at"] = now
        INDEX_PROGRESS[progress_id] = IndexProgressResponse(**data)


def get_index_progress(progress_id: str) -> IndexProgressResponse | None:
    # Return the latest known state for an index job, if it exists.
    with INDEX_PROGRESS_LOCK:
        return INDEX_PROGRESS.get(progress_id)
