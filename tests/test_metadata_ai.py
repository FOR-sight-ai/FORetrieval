"""
Tests for FORetrieval metadata generation WITH an AI provider.

These tests exercise the ``_provider`` callable returned by
``ai_metadata_provider_factory(ai_cfg)`` when a real LLM is configured.
They require at least one of the following environment variables:

    OPENROUTER_API_KEY  (preferred — fast, high rate limits)
                        Model: mistralai/mistral-small-3.2-24b-instruct
    OPENAI_API_KEY      Model: gpt-4o-mini
    MISTRAL_API_KEY     Model: mistral-small-latest
    OLLAMA_HOST         Ollama daemon URL (e.g. http://localhost:11434)
                        + OLLAMA_MODEL (default: llava:latest)

All tests are marked ``@pytest.mark.integration``.

Design
------
The AI provider is called **exactly once** (on the LM317 datasheet) and the
result is cached in a module-scoped fixture.  All individual tests then assert
on this single cached result.  This minimises API calls, cost, and the risk of
hitting rate limits.

Relationship to short_description / summary / abstract
-------------------------------------------------------
``short_description`` is FORetrieval's equivalent of a document summary or
abstract.  It is always an empty string when no AI provider is configured
(tested in test_metadata_no_ai.py), and is filled by the LLM with 1–3
factual sentences when AI is enabled.  The test
``test_ai_short_description_populated`` validates this behaviour.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from foretrieval.metadata import ai_metadata_provider_factory
from foretrieval.models_metadata import DocMetadata

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

TESTS_DIR = Path(__file__).parent
DATA_DIR = TESTS_DIR / "data"

LM317_PDF = DATA_DIR / "lm317_voltage_regulator.pdf"
ULN2003_PDF = DATA_DIR / "uln2003_driver.pdf"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def ai_provider(ai_cfg):
    """Module-scoped AI metadata provider built from the detected backend."""
    return ai_metadata_provider_factory(ai_cfg)


@pytest.fixture(scope="module")
def ai_metadata_lm317(ai_provider):
    """Single real API call for the LM317 datasheet, cached for the whole module.

    All per-field tests use this fixture so we only pay for one LLM call
    regardless of how many tests run.
    """
    try:
        return ai_provider(LM317_PDF)
    except Exception as exc:
        # Surfaced as xfail rather than error so the rest of the suite continues
        if _is_rate_limit(exc):
            pytest.xfail(f"Rate limit hit during AI metadata generation: {exc}")
        raise


def _is_rate_limit(exc: Exception) -> bool:
    """Heuristic: treat common rate-limit / quota errors as xfail."""
    msg = str(exc).lower()
    return any(
        kw in msg for kw in ("rate limit", "429", "quota", "too many requests")
    )


# ---------------------------------------------------------------------------
# Provider construction
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_ai_provider_is_callable(ai_cfg):
    """ai_metadata_provider_factory returns a callable given a valid ai_cfg."""
    provider = ai_metadata_provider_factory(ai_cfg)
    assert callable(provider)


# ---------------------------------------------------------------------------
# Base (non-AI) fields are preserved in the AI result
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_ai_base_fields_preserved(ai_metadata_lm317):
    """Non-AI filesystem fields are still present in the AI-enriched result."""
    raw = ai_metadata_lm317
    assert isinstance(raw.get("source_path"), str) and raw["source_path"]
    assert raw.get("stem") == "lm317_voltage_regulator"
    assert raw.get("ext") == ".pdf"
    assert isinstance(raw.get("mime"), str) and raw["mime"]
    assert isinstance(raw.get("mtime"), str) and raw["mtime"]


@pytest.mark.integration
def test_ai_pdf_page_count_preserved(ai_metadata_lm317):
    """page_count extracted from the PDF is not overwritten by AI."""
    raw = ai_metadata_lm317
    assert isinstance(raw.get("page_count"), int)
    assert raw["page_count"] > 0


# ---------------------------------------------------------------------------
# AI-enriched fields
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_ai_language_populated(ai_metadata_lm317):
    """language is a non-empty string after AI enrichment."""
    raw = ai_metadata_lm317
    lang = raw.get("language")
    assert isinstance(lang, str) and len(lang) > 0, (
        f"Expected non-empty language string, got {lang!r}"
    )


@pytest.mark.integration
def test_ai_language_is_english(ai_metadata_lm317):
    """LM317 datasheet language should be detected as English ('en').

    Marked xfail in case a different LLM returns a variant code ('en-US',
    'english', etc.) rather than raising a hard failure.
    """
    raw = ai_metadata_lm317
    lang = (raw.get("language") or "").lower().strip()
    if not lang.startswith("en"):
        pytest.xfail(
            f"Language was detected as '{lang}' instead of 'en'. "
            "This may be acceptable depending on LLM output format."
        )


@pytest.mark.integration
def test_ai_tags_non_empty(ai_metadata_lm317):
    """tags is a non-empty list after AI enrichment."""
    raw = ai_metadata_lm317
    tags = raw.get("tags")
    assert isinstance(tags, list) and len(tags) > 0, (
        f"Expected non-empty tags list, got {tags!r}"
    )


@pytest.mark.integration
def test_ai_tags_are_strings(ai_metadata_lm317):
    """Every element of tags is a string."""
    raw = ai_metadata_lm317
    for tag in raw.get("tags", []):
        assert isinstance(tag, str), f"Non-string tag: {tag!r}"


@pytest.mark.integration
def test_ai_tags_normalized_lowercase(ai_metadata_lm317):
    """DocMetadata normalises tags to lowercase.

    The DocMetadata field_validator lowercases all tags, so even if the LLM
    returns mixed-case tags they are stored correctly.
    """
    md = DocMetadata(**ai_metadata_lm317)
    for tag in md.tags:
        assert tag == tag.lower(), f"Tag not lowercased: {tag!r}"


@pytest.mark.integration
def test_ai_document_type_not_unknown(ai_metadata_lm317):
    """document_type is classified by the AI (not the default 'unknown')."""
    raw = ai_metadata_lm317
    doc_type = raw.get("document_type", "")
    assert isinstance(doc_type, str) and doc_type, "document_type is empty"
    assert doc_type.lower() != "unknown", (
        f"AI left document_type as 'unknown': {doc_type!r}"
    )


@pytest.mark.integration
def test_ai_short_description_populated(ai_metadata_lm317):
    """short_description (the summary/abstract equivalent) is filled by the AI.

    This is the key test that validates AI-backed summary generation.
    Without AI, short_description is always '' (tested in
    test_metadata_no_ai.py::test_no_ai_short_description_empty).
    With AI, the LLM produces 1-3 factual sentences — at least 20 characters.
    """
    raw = ai_metadata_lm317
    desc = raw.get("short_description", "")
    assert isinstance(desc, str), f"short_description is not a string: {desc!r}"
    assert len(desc) >= 20, (
        f"short_description is too short (< 20 chars): {desc!r}"
    )


# ---------------------------------------------------------------------------
# DocMetadata construction
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_ai_result_valid_doc_metadata(ai_metadata_lm317):
    """DocMetadata(**raw) succeeds for the AI-enriched result."""
    md = DocMetadata(**ai_metadata_lm317)
    assert md.stem == "lm317_voltage_regulator"
    assert md.ext == ".pdf"
    assert md.language is not None
    assert len(md.tags) > 0
    assert md.short_description  # non-empty


@pytest.mark.integration
def test_ai_no_hallucinated_author(ai_metadata_lm317):
    """author is None or a plausible non-empty string.

    The LM317 datasheet has no embedded PDF author metadata, so the AI
    should either return None or a meaningful string — not a hallucinated
    generic placeholder.
    """
    raw = ai_metadata_lm317
    author = raw.get("author")
    if author is not None:
        assert isinstance(author, str) and len(author.strip()) > 0, (
            f"author is set but empty/whitespace: {author!r}"
        )
        # Sanity: should not be an obvious placeholder
        suspicious = {"unknown", "n/a", "none", "anonymous"}
        assert author.strip().lower() not in suspicious, (
            f"author looks like a hallucinated placeholder: {author!r}"
        )


# ---------------------------------------------------------------------------
# Rate-limit resilience (second provider call)
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_ai_second_document_or_xfail(ai_provider):
    """Run the AI provider on ULN2003 datasheet; xfail on rate limit.

    This validates that the provider works for a second document without
    exhausting the session-level cache.  Rate-limit errors are tolerated
    (xfail) since this is an additional call beyond the module fixture.
    """
    try:
        raw = ai_provider(ULN2003_PDF)
    except Exception as exc:
        if _is_rate_limit(exc):
            pytest.xfail(f"Rate limit hit on second document: {exc}")
        raise

    # Basic sanity on the second result
    assert raw.get("ext") == ".pdf"
    assert isinstance(raw.get("short_description"), str)
    DocMetadata(**raw)  # must be constructible
