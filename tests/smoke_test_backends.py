"""
Smoke test: index toy_data/smartcockpit with all three backends using a remote embedding server,
then run one query and compare top-1 results.

Usage:
    uv run python FORetrieval/tests/smoke_test_backends.py
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

# Add FORetrieval to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "FORetrieval"))

from foretrieval import MultiModalRetrieverModel
from foretrieval.embedding_server import EmbeddingServerConfig

EMBEDDING_SERVER_URL = os.getenv("EMBEDDING_SERVER_URL", "http://localhost:18000")
MODEL_NAME = "athrael-soju/colqwen3.5-4.5B-v3"
DATA_DIR = Path(__file__).parent.parent.parent / "toy_data" / "smartcockpit"
QUERY = "What is the normal operating cabin altitude in cruise for the A320?"
TOP_K = 3

embedding_cfg = EmbeddingServerConfig(
    url=EMBEDDING_SERVER_URL,
    model_name=MODEL_NAME,
    auto_deploy=False,
    batch_size=4,
)

BACKENDS = ["local", "qdrant", "milvus"]
results_by_backend = {}

with tempfile.TemporaryDirectory() as tmp:
    for backend in BACKENDS:
        print(f"\n{'='*60}")
        print(f"Backend: {backend}")
        print(f"{'='*60}")

        rag = MultiModalRetrieverModel.from_pretrained(
            pretrained_model_name_or_path=MODEL_NAME,
            index_root=tmp,
            storage_backend=backend,
            embedding_server=embedding_cfg,
            device="cuda",
            verbose=1,
        )

        rag.index(
            input_path=str(DATA_DIR),
            index_name=f"smoke_{backend}",
            overwrite=True,
        )

        print(f"Query: {QUERY}")
        results = rag.search(QUERY, k=TOP_K, return_base64_results=False)
        results_by_backend[backend] = results

        for i, r in enumerate(results):
            print(f"  [{i+1}] doc_id={r.doc_id} page={r.page_num} score={r.score:.4f}")

# Compare top-1 file across backends
print("\n" + "="*60)
print("Comparison summary")
print("="*60)
print(f"Query: {QUERY}\n")
for backend, results in results_by_backend.items():
    if results:
        r = results[0]
        print(f"  {backend:6s}: doc_id={r.doc_id} page={r.page_num} score={r.score:.4f}")
    else:
        print(f"  {backend:6s}: NO RESULTS")

# All backends should return at least 1 result
for backend, results in results_by_backend.items():
    assert len(results) > 0, f"{backend} returned no results!"
print("\nAll backends returned results. Smoke test PASSED.")
