"""Entry point for running the vector-DB server via uvicorn.

Usage:
    python -m foretrieval.vector_db_server.server_main

Or via the installed script (if declared in pyproject.toml):
    foretrieval-db-server

Environment variables:
    FOR_DB_HOST       Bind address (default: 0.0.0.0)
    FOR_DB_PORT       Bind port    (default: 18000)
    FOR_DB_DATA_DIR   Data root    (default: /data)
    FOR_DB_API_KEY    Bearer token (optional)
"""

from __future__ import annotations

import os
import uvicorn

from .server import app  # noqa: F401 — imported so uvicorn picks it up


def main() -> None:
    host = os.environ.get("FOR_DB_HOST", "0.0.0.0")
    port = int(os.environ.get("FOR_DB_PORT", "18000"))
    uvicorn.run(
        "foretrieval.vector_db_server.server:app",
        host=host,
        port=port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
