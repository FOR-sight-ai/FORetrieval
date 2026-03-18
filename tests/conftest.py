"""
Shared fixtures for the FORetrieval test suite.
"""

from __future__ import annotations

import os
import urllib.error
import urllib.request
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

TESTS_DIR = Path(__file__).parent
DATA_DIR = TESTS_DIR / "data"
SAMPLE_DATA_DIR = Path(__file__).parent.parent / "sample_data"

# ---------------------------------------------------------------------------
# AI backend selection — same priority order as FORag
# ---------------------------------------------------------------------------

_BACKEND_MAP = {
    "openrouter": ("OPENROUTER_API_KEY", "mistralai/mistral-small-3.2-24b-instruct"),
    "openai": ("OPENAI_API_KEY", "gpt-4o-mini"),
    "mistral": ("MISTRAL_API_KEY", "mistral-small-latest"),
}


def _all_available_ai_backends() -> list[dict]:
    """Return an ai_cfg dict for every available backend.

    For cloud backends the credential is an API key.  For Ollama, the
    ``base_url`` key points to the daemon's OpenAI-compatible endpoint.

    Priority order: OpenRouter → OpenAI → Mistral → Ollama.
    This mirrors the backend selection used in FORag's integration suite.

    Ollama model defaults to ``mistral-small-latest`` (a text-only model
    suitable for metadata generation).  Override with the ``OLLAMA_MODEL``
    environment variable.
    """
    available = []

    for provider, (env_var, model) in _BACKEND_MAP.items():
        key = os.environ.get(env_var)
        if key:
            available.append({"provider": provider, "name": model, "api_key": key})

    host = os.environ.get("OLLAMA_HOST")
    if host:
        try:
            urllib.request.urlopen(f"{host}/api/tags", timeout=3)
            # Default to a text-only model — not a vision model — since
            # metadata generation uses text prompts only.
            model = os.environ.get("OLLAMA_MODEL", "mistral-small-latest")
            available.append(
                {"provider": "ollama", "name": model, "base_url": f"{host}/v1"}
            )
        except (urllib.error.URLError, OSError):
            pass  # daemon unreachable — skip silently

    return available


@pytest.fixture(
    scope="module",
    params=_all_available_ai_backends(),
    ids=lambda cfg: cfg["provider"],
)
def ai_cfg(request) -> dict:
    """Module-scoped fixture parametrized over every available AI backend.

    Each test module that depends on this fixture will be run once per
    available backend.  If no backend is available the entire module is
    skipped.

    Set one of the following environment variables to enable AI tests:

        OPENROUTER_API_KEY  (preferred — fast, high rate limits)
        OPENAI_API_KEY
        MISTRAL_API_KEY
        OLLAMA_HOST         (+ optionally OLLAMA_MODEL, default: mistral-small-latest)
    """
    if not _all_available_ai_backends():
        pytest.skip(
            "No API key or Ollama daemon found. Set OPENROUTER_API_KEY, "
            "OPENAI_API_KEY, MISTRAL_API_KEY, or OLLAMA_HOST."
        )
    return request.param
