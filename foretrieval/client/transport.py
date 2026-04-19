from __future__ import annotations

import io
from typing import Any

import torch


def dumps_payload(payload: Any) -> bytes:
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    return buffer.getvalue()


def loads_payload(data: bytes) -> Any:
    buffer = io.BytesIO(data)
    return torch.load(buffer, map_location="cpu")
