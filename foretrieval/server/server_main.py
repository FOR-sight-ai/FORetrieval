from __future__ import annotations

import os

from .embedding_server import create_app


def _as_bool(name: str, default: bool = False) -> bool:
    val = os.environ.get(name)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "on"}


def _as_positive_int(name: str, default: int) -> int:
    val = os.environ.get(name)
    if val is None:
        return default
    try:
        parsed = int(val)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {val!r}.") from exc
    if parsed < 1:
        raise ValueError(f"{name} must be >= 1, got {parsed}.")
    return parsed


def _as_optional_str(name: str) -> str | None:
    val = os.environ.get(name)
    if val is None:
        return None
    stripped = val.strip()
    return stripped if stripped else None


MODEL_NAME = os.environ.get("FOR_EMBED_MODEL", "vidore/colqwen2-v1.0")
DEVICE = os.environ.get("FOR_EMBED_DEVICE", "cuda")
VERBOSE = int(os.environ.get("FOR_EMBED_VERBOSE", "1"))
HF_TOKEN = _as_optional_str("HF_TOKEN")
LOAD_IN_4BIT = _as_bool("FOR_EMBED_LOAD_IN_4BIT", False)
LOAD_IN_8BIT = _as_bool("FOR_EMBED_LOAD_IN_8BIT", False)
BNB_4BIT_QUANT_TYPE = os.environ.get("FOR_EMBED_BNB_4BIT_QUANT_TYPE", "nf4")
BNB_4BIT_COMPUTE_DTYPE = os.environ.get("FOR_EMBED_BNB_4BIT_COMPUTE_DTYPE", "float16")
MAX_INFLIGHT = _as_positive_int("FOR_SERVER_MAX_INFLIGHT", 1)

app = create_app(
    model_name=MODEL_NAME,
    device=DEVICE,
    verbose=VERBOSE,
    hf_token=HF_TOKEN,
    load_in_4bit=LOAD_IN_4BIT,
    load_in_8bit=LOAD_IN_8BIT,
    bnb_4bit_quant_type=BNB_4BIT_QUANT_TYPE,
    bnb_4bit_compute_dtype=BNB_4BIT_COMPUTE_DTYPE,
    max_inflight=MAX_INFLIGHT,
)
