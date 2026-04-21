"""
Benchmark script for comparing embedding server solutions.

Usage (Sol A — vLLM):
    uv run python scripts/benchmark.py \
        --solution a \
        --server-url http://localhost:18000 \
        --model athrael-soju/colqwen3.5-4.5B-v3 \
        --data-dir ../toy_data/smartcockpit \
        --output benchmark_results/sol_a.json

Usage (Sol B — custom FastAPI):
    uv run python scripts/benchmark.py \
        --solution b \
        --server-url http://localhost:18001 \
        --model vidore/colqwen2-v1.0 \
        --data-dir ../toy_data/smartcockpit \
        --output benchmark_results/sol_b.json
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Queries — precise aviation questions whose answers are sentence/paragraph
# form, drawn from A319/A320 content in the smartcockpit PDFs.
# ---------------------------------------------------------------------------
QUERIES = [
    "What is the normal operating cabin altitude in cruise for the A320?",
    "How is the bleed air supply to the air conditioning system controlled on the A320?",
    "What happens to pressurization if both outflow valves fail in the open position?",
    "What is the purpose of the Ram Air inlet on the A320 air conditioning system?",
    "Describe the function of the Pack Flow Control Valve and its operating modes.",
]

QUERY_RUNS = 5  # repeat each query N times to get stable latency


# ---------------------------------------------------------------------------
# GPU VRAM sampler (non-blocking, best-effort)
# ---------------------------------------------------------------------------

def _sample_vram_mb() -> Optional[float]:
    """Return total used VRAM in MB across all GPUs, or None if unavailable."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
        values = [int(x.strip()) for x in out.decode().splitlines() if x.strip()]
        return float(sum(values)) if values else None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Sol B client (httpx-based, mirrors feature/client-server transport)
# ---------------------------------------------------------------------------

def _sol_b_health(server_url: str) -> bool:
    import httpx
    try:
        r = httpx.get(f"{server_url.rstrip('/')}/health", timeout=5)
        return r.status_code == 200
    except Exception:
        return False


def _sol_b_embed_images(server_url: str, images, model_name: str):
    """Call Sol B POST /v1/embed/images using torch binary transport."""
    import io
    import torch
    import httpx

    buf = io.BytesIO()
    # images: list of PIL.Image → convert to PNG bytes
    image_bytes_list = []
    for img in images:
        ibuf = io.BytesIO()
        img.save(ibuf, format="PNG")
        image_bytes_list.append(ibuf.getvalue())

    payload = {"model": model_name, "images": image_bytes_list}
    torch.save(payload, buf)
    buf.seek(0)

    with httpx.Client(timeout=600) as client:
        resp = client.post(
            f"{server_url.rstrip('/')}/v1/embed/images",
            content=buf.read(),
            headers={"Content-Type": "application/octet-stream"},
        )
        resp.raise_for_status()

    result_buf = io.BytesIO(resp.content)
    result = torch.load(result_buf, weights_only=False)
    return result["embeddings"]


def _sol_b_embed_query(server_url: str, query: str, model_name: str):
    """Call Sol B POST /v1/embed/queries using torch binary transport."""
    import io
    import torch
    import httpx

    payload = {"model": model_name, "queries": [query]}
    buf = io.BytesIO()
    torch.save(payload, buf)
    buf.seek(0)

    with httpx.Client(timeout=600) as client:
        resp = client.post(
            f"{server_url.rstrip('/')}/v1/embed/queries",
            content=buf.read(),
            headers={"Content-Type": "application/octet-stream"},
        )
        resp.raise_for_status()

    result_buf = io.BytesIO(resp.content)
    result = torch.load(result_buf, weights_only=False)
    return result["embeddings"]


# ---------------------------------------------------------------------------
# Main benchmark logic
# ---------------------------------------------------------------------------

def run_sol_a(server_url: str, model_name: str, data_dir: Path, index_root: Path) -> dict:
    """Benchmark Solution A: vLLM-backed MultiModalRetrieverModel."""
    from foretrieval import MultiModalRetrieverModel
    from foretrieval.embedding_server import EmbeddingServerConfig

    cfg = EmbeddingServerConfig(
        url=server_url,
        model_name=model_name,
        auto_deploy=False,
    )

    # Health check
    from foretrieval.embedding_server.client import EmbeddingServerClient
    client = EmbeddingServerClient(cfg)
    if not client.health_check():
        print(f"ERROR: Sol A server not healthy at {server_url}", file=sys.stderr)
        sys.exit(1)
    print(f"Sol A server healthy at {server_url}")

    # --- Indexing ---
    print("Indexing smartcockpit corpus (Sol A)...")
    pdf_files = list(data_dir.glob("*.pdf")) + list(data_dir.glob("*.PDF"))
    n_docs = len(pdf_files)
    vram_before = _sample_vram_mb()
    t0 = time.perf_counter()
    model = MultiModalRetrieverModel.from_pretrained(
        model_name,
        index_root=str(index_root),
        device="cpu",  # no local GPU needed — embeddings go to server
        verbose=1,
        embedding_server=cfg,
    )
    model.index(str(data_dir), index_name="smartcockpit_bench", overwrite=True)
    index_time = time.perf_counter() - t0
    vram_after_index = _sample_vram_mb()
    print(f"  Indexing done in {index_time:.1f}s")

    # --- Query latency ---
    query_results = []
    for q in QUERIES:
        latencies = []
        top_pages = None
        for run in range(QUERY_RUNS):
            t0 = time.perf_counter()
            results = model.search(q, k=3)
            latencies.append(time.perf_counter() - t0)
            if run == 0:
                top_pages = [
                    {"doc_id": r.doc_id, "page_num": r.page_num, "score": float(r.score) if r.score is not None else None}
                    for r in results
                ] if results else []
        mean_lat = sum(latencies) / len(latencies)
        std_lat = (sum((x - mean_lat) ** 2 for x in latencies) / len(latencies)) ** 0.5
        print(f"  Query '{q[:60]}...' → mean {mean_lat:.3f}s ± {std_lat:.3f}s")
        query_results.append({
            "query": q,
            "latency_mean_s": round(mean_lat, 4),
            "latency_std_s": round(std_lat, 4),
            "latency_all_s": [round(x, 4) for x in latencies],
            "top_3_pages": top_pages,
        })

    vram_peak = _sample_vram_mb()

    return {
        "solution": "A",
        "server_url": server_url,
        "model_name": model_name,
        "n_docs": n_docs,
        "indexing_time_s": round(index_time, 2),
        "vram_before_index_mb": vram_before,
        "vram_after_index_mb": vram_after_index,
        "vram_peak_mb": vram_peak,
        "queries": query_results,
    }


def run_sol_b(server_url: str, model_name: str, data_dir: Path) -> dict:
    """
    Benchmark Solution B: custom FastAPI server.

    Sol B has no MultiModalRetrieverModel integration so we call the HTTP
    endpoints directly and measure raw embedding throughput.  We convert
    PDFs to page images (same pipeline FORetrieval uses) then embed them.
    """
    try:
        import httpx  # noqa: F401
        import torch  # noqa: F401
    except ImportError as e:
        print(f"ERROR: missing dep for Sol B benchmark: {e}", file=sys.stderr)
        sys.exit(1)

    # Health check
    if not _sol_b_health(server_url):
        print(f"ERROR: Sol B server not healthy at {server_url}", file=sys.stderr)
        sys.exit(1)
    print(f"Sol B server healthy at {server_url}")

    # Warm-up: trigger model load before timing (Sol B loads lazily on first request)
    print("Warming up Sol B server (model load)...")
    from PIL import Image as PILImage
    _warmup_img = PILImage.new("RGB", (64, 64), color=(128, 128, 128))
    t_warmup = time.perf_counter()
    _sol_b_embed_images(server_url, [_warmup_img], model_name)
    print(f"  Warm-up done in {time.perf_counter() - t_warmup:.1f}s (model load included)")

    # Convert PDFs to images (reuse pdf2image, same as FORetrieval internals)
    from pdf2image import convert_from_path

    pdf_files = sorted((list(data_dir.glob("*.pdf")) + list(data_dir.glob("*.PDF"))))
    print(f"Converting {len(pdf_files)} PDFs to images...")
    all_images = []
    for pdf in pdf_files:
        pages = convert_from_path(str(pdf), dpi=150)
        all_images.extend(pages)
        print(f"  {pdf.name}: {len(pages)} pages")
    n_pages = len(all_images)
    n_docs = len(pdf_files)
    print(f"Total pages: {n_pages}")

    # --- Indexing (embedding all pages) ---
    print("Embedding all pages (Sol B)...")
    vram_before = _sample_vram_mb()
    BATCH = 4
    t0 = time.perf_counter()
    all_embeddings = []
    for i in range(0, n_pages, BATCH):
        batch = all_images[i : i + BATCH]
        emb = _sol_b_embed_images(server_url, batch, model_name)
        all_embeddings.append(emb)
        print(f"  Embedded pages {i+1}-{min(i+BATCH, n_pages)}/{n_pages}")
    index_time = time.perf_counter() - t0
    vram_after_index = _sample_vram_mb()
    print(f"  Indexing done in {index_time:.1f}s  ({n_pages / index_time:.2f} pages/s)")

    # --- Query latency ---
    query_results = []
    for q in QUERIES:
        latencies = []
        for run in range(QUERY_RUNS):
            t0 = time.perf_counter()
            _sol_b_embed_query(server_url, q, model_name)
            latencies.append(time.perf_counter() - t0)
        mean_lat = sum(latencies) / len(latencies)
        std_lat = (sum((x - mean_lat) ** 2 for x in latencies) / len(latencies)) ** 0.5
        print(f"  Query '{q[:60]}...' → mean {mean_lat:.3f}s ± {std_lat:.3f}s")
        query_results.append({
            "query": q,
            "latency_mean_s": round(mean_lat, 4),
            "latency_std_s": round(std_lat, 4),
            "latency_all_s": [round(x, 4) for x in latencies],
            "top_3_pages": None,  # no retrieval integration in Sol B direct mode
        })

    vram_peak = _sample_vram_mb()

    return {
        "solution": "B",
        "server_url": server_url,
        "model_name": model_name,
        "n_docs": n_docs,
        "n_pages": n_pages,
        "indexing_time_s": round(index_time, 2),
        "pages_per_sec": round(n_pages / index_time, 2),
        "vram_before_index_mb": vram_before,
        "vram_after_index_mb": vram_after_index,
        "vram_peak_mb": vram_peak,
        "queries": query_results,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Embedding server benchmark")
    parser.add_argument("--solution", choices=["a", "b"], required=True,
                        help="Which solution to benchmark: 'a' (vLLM) or 'b' (custom FastAPI)")
    parser.add_argument("--server-url", required=True,
                        help="Base URL of the embedding server (e.g. http://localhost:18000)")
    parser.add_argument("--model", required=True,
                        help="Model name served by the server")
    parser.add_argument("--data-dir", default="../toy_data/smartcockpit",
                        help="Path to directory containing smartcockpit PDFs")
    parser.add_argument("--output", default=None,
                        help="Path to write JSON results (default: benchmark_results/sol_{a,b}.json)")
    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    if not data_dir.exists():
        print(f"ERROR: data-dir not found: {data_dir}", file=sys.stderr)
        sys.exit(1)

    out_path = Path(args.output) if args.output else Path(f"benchmark_results/sol_{args.solution}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"=== Benchmark Solution {'A (vLLM)' if args.solution == 'a' else 'B (custom FastAPI)'} ===")
    print(f"Server : {args.server_url}")
    print(f"Model  : {args.model}")
    print(f"Data   : {data_dir}")
    print()

    if args.solution == "a":
        index_root = Path(tempfile.mkdtemp(prefix="foretrieval_bench_"))
        try:
            results = run_sol_a(args.server_url, args.model, data_dir, index_root)
        finally:
            shutil.rmtree(index_root, ignore_errors=True)
    else:
        results = run_sol_b(args.server_url, args.model, data_dir)

    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults written to {out_path}")

    # Print summary
    print("\n--- Summary ---")
    print(f"Indexing time : {results['indexing_time_s']}s")
    if "pages_per_sec" in results:
        print(f"Pages/sec     : {results['pages_per_sec']}")
    if results.get("vram_peak_mb"):
        print(f"Peak VRAM     : {results['vram_peak_mb']} MB")
    q_means = [q["latency_mean_s"] for q in results["queries"]]
    overall_mean = sum(q_means) / len(q_means)
    print(f"Query latency : {overall_mean:.3f}s avg over {len(QUERIES)} queries x {QUERY_RUNS} runs")


if __name__ == "__main__":
    main()
