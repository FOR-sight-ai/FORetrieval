from __future__ import annotations

import asyncio
import io
from typing import Dict, Optional

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
    max_inflight: int = 1,
) -> FastAPI:
    app = FastAPI(title="FORetrieval Embedding Server")
    inflight_semaphore = asyncio.Semaphore(max_inflight)

    retriever_cache: Dict[str, MultiModalRetrieverModel] = {}

    def get_retriever(requested_model: Optional[str]) -> MultiModalRetrieverModel:
        target_model = requested_model or model_name
        if target_model not in retriever_cache:
            retriever_cache[target_model] = MultiModalRetrieverModel.from_pretrained(
                target_model,
                device=device,
                verbose=verbose,
                hf_token=hf_token,
                embedding_mode="local",
                load_in_4bit=load_in_4bit,
                load_in_8bit=load_in_8bit,
                bnb_4bit_quant_type=bnb_4bit_quant_type,
                bnb_4bit_compute_dtype=bnb_4bit_compute_dtype,
            )
        return retriever_cache[target_model]

    @app.get("/health")
    def health():
        return {
            "status": "ok",
            "default_model": model_name,
            "loaded_models": list(retriever_cache.keys()),
            "device": str(device),
            "max_inflight": max_inflight,
        }

    @app.post("/v1/embed/images")
    async def embed_images(request: Request):
        try:
            payload = loads_payload(await request.body())
            async with inflight_semaphore:
                retriever = get_retriever(payload.get("model"))
                images = [Image.open(io.BytesIO(b)).convert("RGB") for b in payload["images"]]
                embeddings = retriever.model.encode_image(images)
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
                retriever = get_retriever(payload.get("model"))
                queries = payload["queries"]
                embeddings = retriever.model.encode_query(queries)
            return Response(
                content=dumps_payload({"embeddings": embeddings.cpu()}),
                media_type="application/octet-stream",
            )
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Invalid query payload: {exc}") from exc

    return app
