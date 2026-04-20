from __future__ import annotations

import asyncio
import gc
import io
from typing import Dict, Optional, Tuple

from fastapi import FastAPI, HTTPException, Request, Response
from PIL import Image

from ..client.transport import dumps_payload, loads_payload
from ..retriever import MultiModalRetrieverModel


def create_app(
    model_name: str,
    device: str = "cuda",
    hf_token: Optional[str] = None,
    verbose: int = 1,
    load_in_4bit: bool = False,
    load_in_8bit: bool = False,
    bnb_4bit_quant_type: str = "nf4",
    bnb_4bit_compute_dtype: str = "float16",
    flash_attention_mode: str = "auto",
    max_inflight: int = 1,
    single_model_cache: bool = False,
) -> FastAPI:
    app = FastAPI(title="FORetrieval Embedding Server")
    inflight_semaphore = asyncio.Semaphore(max_inflight)
    cache_lock = asyncio.Lock()

    retriever_cache: Dict[str, MultiModalRetrieverModel] = {}
    retriever_in_use: Dict[int, int] = {}
    pending_releases: Dict[int, MultiModalRetrieverModel] = {}

    def _release_retriever(retriever: MultiModalRetrieverModel) -> None:
        # Best effort release: drop strong refs, force GC, then flush CUDA allocator cache.
        del retriever
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                if hasattr(torch.cuda, "ipc_collect"):
                    torch.cuda.ipc_collect()
        except Exception:
            # Cache eviction should not fail request processing.
            pass

    def _schedule_or_release_locked(retriever: MultiModalRetrieverModel) -> None:
        rid = id(retriever)
        if retriever_in_use.get(rid, 0) > 0:
            pending_releases[rid] = retriever
            return
        _release_retriever(retriever)

    async def _acquire_retriever(requested_model: Optional[str]) -> Tuple[MultiModalRetrieverModel, int]:
        target_model = requested_model or model_name
        async with cache_lock:
            cached = retriever_cache.get(target_model)
            if cached is None:
                if single_model_cache and retriever_cache:
                    old_retrievers = list(retriever_cache.values())
                    retriever_cache.clear()
                    for old in old_retrievers:
                        _schedule_or_release_locked(old)

                cached = MultiModalRetrieverModel.from_pretrained(
                    target_model,
                    device=device,
                    verbose=verbose,
                    hf_token=hf_token,
                    embedding_mode="local",
                    load_in_4bit=load_in_4bit,
                    load_in_8bit=load_in_8bit,
                    bnb_4bit_quant_type=bnb_4bit_quant_type,
                    bnb_4bit_compute_dtype=bnb_4bit_compute_dtype,
                    flash_attention_mode=flash_attention_mode,
                )
                retriever_cache[target_model] = cached

            rid = id(cached)
            retriever_in_use[rid] = retriever_in_use.get(rid, 0) + 1
            return cached, rid

    async def _release_retriever_use(rid: int) -> None:
        to_release = None
        async with cache_lock:
            current = retriever_in_use.get(rid, 0)
            if current <= 1:
                retriever_in_use.pop(rid, None)
                to_release = pending_releases.pop(rid, None)
            else:
                retriever_in_use[rid] = current - 1
        if to_release is not None:
            _release_retriever(to_release)

    @app.get("/health")
    def health():
        return {
            "status": "ok",
            "default_model": model_name,
            "loaded_models": list(retriever_cache.keys()),
            "single_model_cache": single_model_cache,
            "in_use_retrievers": len(retriever_in_use),
            "pending_evictions": len(pending_releases),
            "device": str(device),
            "max_inflight": max_inflight,
        }

    @app.post("/v1/embed/images")
    async def embed_images(request: Request):
        try:
            payload = loads_payload(await request.body())
            async with inflight_semaphore:
                retriever, rid = await _acquire_retriever(payload.get("model"))
                images = [Image.open(io.BytesIO(b)).convert("RGB") for b in payload["images"]]
                try:
                    embeddings = retriever.model.encode_image(images)
                finally:
                    await _release_retriever_use(rid)
            return Response(
                content=dumps_payload({"embeddings": embeddings.cpu()}),
                media_type="application/octet-stream",
            )
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Invalid image payload: {exc}") from exc

    @app.post("/v1/embed/queries")
    async def embed_queries(request: Request):
        try:
            payload = loads_payload(await request.body())
            async with inflight_semaphore:
                retriever, rid = await _acquire_retriever(payload.get("model"))
                queries = payload["queries"]
                try:
                    embeddings = retriever.model.encode_query(queries)
                finally:
                    await _release_retriever_use(rid)
            return Response(
                content=dumps_payload({"embeddings": embeddings.cpu()}),
                media_type="application/octet-stream",
            )
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Invalid query payload: {exc}") from exc

    return app
