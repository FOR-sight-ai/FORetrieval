"""
Tests for FORetrieval metadata generation WITHOUT an AI provider.

These tests exercise the ``_no_ai_provider`` callable returned by
``ai_metadata_provider_factory(None)``.  No API key, no GPU, and no network
access are required — every test runs against real files on disk.

Metadata fields and their expected values without AI
-----------------------------------------------------
+--------------------+----------------------------------+
| Field              | Expected value (no AI)           |
+====================+==================================+
| source_path        | str (absolute path)              |
| stem               | str (filename without extension) |
| ext                | str (starts with '.', lowercase) |
| mime               | str                              |
| mtime              | ISO-8601 UTC string              |
| page_count         | int > 0  (PDFs only)             |
| image_width        | int > 0  (images only)           |
| image_height       | int > 0  (images only)           |
| author             | str or None (from PDF metadata)  |
| title              | str or None (from PDF metadata)  |
| language           | None  ← not detected without AI  |
| tags               | []    ← not generated without AI |
| document_type      | "unknown"                        |
| short_description  | ""    ← the "summary/abstract"   |
|                    |    equivalent; empty without AI  |
+--------------------+----------------------------------+

Note: ``short_description`` is the field that corresponds to a document
summary or abstract.  It is empty when no AI provider is configured, and
is filled by the LLM when an AI provider is used (see test_metadata_ai.py).
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from foretrieval.metadata import ai_metadata_provider_factory
from foretrieval.models_metadata import DocMetadata, build_metadata_list_for_dir

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

TESTS_DIR = Path(__file__).parent
DATA_DIR = TESTS_DIR / "data"
SAMPLE_DATA_DIR = TESTS_DIR.parent / "sample_data"

LM317_PDF = DATA_DIR / "lm317_voltage_regulator.pdf"
ULN2003_PDF = DATA_DIR / "uln2003_driver.pdf"
SAMPLE_PNG = SAMPLE_DATA_DIR / "sample_png.png"
SAMPLE_JPG = SAMPLE_DATA_DIR / "sample_jpg.jpg"
SAMPLE_TXT = SAMPLE_DATA_DIR / "sample_txt.txt"
SAMPLE_DOCX = SAMPLE_DATA_DIR / "sample_docx.docx"
SAMPLE_XLSX = SAMPLE_DATA_DIR / "sample_xlsx.xlsx"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def provider():
    """Module-scoped no-AI metadata provider."""
    return ai_metadata_provider_factory(None)


# ---------------------------------------------------------------------------
# Base field tests
# ---------------------------------------------------------------------------


def test_no_ai_pdf_base_fields(provider):
    """All base filesystem fields are populated for a PDF."""
    raw = provider(LM317_PDF)

    assert isinstance(raw["source_path"], str)
    assert raw["source_path"].endswith("lm317_voltage_regulator.pdf")
    assert raw["stem"] == "lm317_voltage_regulator"
    assert isinstance(raw["mime"], str) and raw["mime"]
    assert isinstance(raw["mtime"], str) and raw["mtime"]


def test_no_ai_ext_normalized_pdf(provider):
    """ext is '.pdf' (lowercase, starts with dot) for a PDF file."""
    raw = provider(LM317_PDF)
    assert raw["ext"] == ".pdf"


def test_no_ai_ext_normalized_image(provider):
    """ext starts with '.' and is lowercase for image files."""
    for path in [SAMPLE_PNG, SAMPLE_JPG]:
        raw = provider(path)
        assert raw["ext"].startswith("."), f"ext missing leading dot for {path.name}"
        assert raw["ext"] == raw["ext"].lower(), f"ext not lowercase for {path.name}"


def test_no_ai_mtime_is_valid_iso_utc(provider):
    """mtime is a parseable ISO-8601 string with UTC timezone info."""
    raw = provider(LM317_PDF)
    mtime_str = raw["mtime"]
    # Parse via datetime — should not raise
    dt = datetime.fromisoformat(mtime_str.replace("Z", "+00:00"))
    assert dt.tzinfo is not None, "mtime must carry timezone info"


# ---------------------------------------------------------------------------
# PDF-specific tests
# ---------------------------------------------------------------------------


def test_no_ai_pdf_page_count_lm317(provider):
    """page_count is a positive integer for the LM317 datasheet."""
    raw = provider(LM317_PDF)
    assert isinstance(raw["page_count"], int)
    assert raw["page_count"] > 0


def test_no_ai_pdf_page_count_uln2003(provider):
    """page_count is a positive integer for the ULN2003 datasheet."""
    raw = provider(ULN2003_PDF)
    assert isinstance(raw["page_count"], int)
    assert raw["page_count"] > 0


def test_no_ai_non_pdf_page_count_is_none(provider):
    """page_count is None for non-PDF files."""
    for path in [SAMPLE_PNG, SAMPLE_TXT, SAMPLE_DOCX]:
        raw = provider(path)
        assert raw["page_count"] is None, (
            f"Expected page_count=None for {path.name}, got {raw['page_count']}"
        )


# ---------------------------------------------------------------------------
# Image dimension tests
# ---------------------------------------------------------------------------


def test_no_ai_image_dims_png(provider):
    """image_width and image_height are positive ints for a PNG."""
    raw = provider(SAMPLE_PNG)
    assert isinstance(raw["image_width"], int) and raw["image_width"] > 0
    assert isinstance(raw["image_height"], int) and raw["image_height"] > 0


def test_no_ai_image_dims_jpg(provider):
    """image_width and image_height are positive ints for a JPEG."""
    raw = provider(SAMPLE_JPG)
    assert isinstance(raw["image_width"], int) and raw["image_width"] > 0
    assert isinstance(raw["image_height"], int) and raw["image_height"] > 0


def test_no_ai_non_image_dims_are_none(provider):
    """image_width and image_height are None for non-image files."""
    for path in [LM317_PDF, SAMPLE_TXT, SAMPLE_DOCX]:
        raw = provider(path)
        assert raw["image_width"] is None, f"Expected None for {path.name}"
        assert raw["image_height"] is None, f"Expected None for {path.name}"


# ---------------------------------------------------------------------------
# AI field absence tests (the critical no-AI assertions)
# ---------------------------------------------------------------------------


def test_no_ai_language_is_none(provider):
    """language is None without AI — language detection requires the LLM."""
    for path in [LM317_PDF, ULN2003_PDF]:
        raw = provider(path)
        assert raw["language"] is None, (
            f"Expected language=None without AI for {path.name}"
        )


def test_no_ai_tags_empty(provider):
    """tags is an empty list without AI — tagging requires the LLM."""
    for path in [LM317_PDF, ULN2003_PDF]:
        raw = provider(path)
        assert raw["tags"] == [], (
            f"Expected tags=[] without AI for {path.name}"
        )


def test_no_ai_document_type_unknown(provider):
    """document_type is 'unknown' without AI — classification requires the LLM."""
    for path in [LM317_PDF, ULN2003_PDF]:
        raw = provider(path)
        assert raw["document_type"] == "unknown", (
            f"Expected document_type='unknown' without AI for {path.name}"
        )


def test_no_ai_short_description_empty(provider):
    """short_description (the summary/abstract equivalent) is empty without AI.

    ``short_description`` is FORetrieval's equivalent of a document
    summary or abstract.  Without an AI provider it is always an empty
    string; the LLM fills it with 1-3 factual sentences when AI is
    enabled (see test_metadata_ai.py::test_ai_short_description_populated).
    """
    for path in [LM317_PDF, ULN2003_PDF]:
        raw = provider(path)
        assert raw["short_description"] == "", (
            f"Expected short_description='' without AI for {path.name}"
        )


# ---------------------------------------------------------------------------
# DocMetadata construction tests
# ---------------------------------------------------------------------------


def test_no_ai_result_valid_doc_metadata_lm317(provider):
    """DocMetadata(**raw) succeeds for the LM317 datasheet."""
    raw = provider(LM317_PDF)
    md = DocMetadata(**raw)
    assert md.stem == "lm317_voltage_regulator"
    assert md.ext == ".pdf"


def test_no_ai_result_valid_doc_metadata_uln2003(provider):
    """DocMetadata(**raw) succeeds for the ULN2003 datasheet."""
    raw = provider(ULN2003_PDF)
    md = DocMetadata(**raw)
    assert md.ext == ".pdf"


def test_no_ai_doc_metadata_tags_normalized(provider):
    """DocMetadata normalises tags to lowercase stripped strings."""
    raw = provider(LM317_PDF)
    raw["tags"] = ["  Python ", "RAG", "FOO"]
    md = DocMetadata(**raw)
    assert md.tags == ["python", "rag", "foo"]


def test_no_ai_doc_metadata_ext_normalized(provider):
    """DocMetadata normalises ext: adds leading dot, lowercases."""
    raw = provider(LM317_PDF)
    raw["ext"] = "PDF"  # deliberate unnormalized value
    md = DocMetadata(**raw)
    assert md.ext == ".pdf"


# ---------------------------------------------------------------------------
# Multiple format smoke tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "path",
    [
        SAMPLE_TXT,
        SAMPLE_DOCX,
        SAMPLE_XLSX,
        SAMPLE_PNG,
        SAMPLE_JPG,
    ],
    ids=["txt", "docx", "xlsx", "png", "jpg"],
)
def test_no_ai_multiple_formats_no_crash(provider, path):
    """provider() returns a valid dict for all common file types without crashing."""
    raw = provider(path)
    assert isinstance(raw, dict)
    # Must always contain these base keys
    for key in ("source_path", "stem", "ext", "mime", "mtime"):
        assert key in raw, f"Missing key '{key}' for {path.name}"
    # Must be constructible as DocMetadata
    DocMetadata(**raw)


# ---------------------------------------------------------------------------
# build_metadata_list_for_dir tests
# ---------------------------------------------------------------------------


def test_build_metadata_list_for_dir_length(provider):
    """Result list has the same length as the recursive file enumeration used by index()."""
    items = sorted(
        (p for p in DATA_DIR.rglob("*") if p.is_file()),
        key=lambda p: p.relative_to(DATA_DIR),
    )
    md_list = build_metadata_list_for_dir(DATA_DIR, provider)
    assert len(md_list) == len(items)


def test_build_metadata_list_for_dir_all_doc_metadata(provider):
    """Every entry in the result list is a DocMetadata instance (no None placeholders)."""
    md_list = build_metadata_list_for_dir(DATA_DIR, provider)
    for md in md_list:
        # All entries are files (directories are excluded); provider always
        # returns metadata for PDF files in DATA_DIR.
        assert isinstance(md, DocMetadata), (
            f"Expected DocMetadata for all entries, got {type(md)}"
        )


def test_build_metadata_list_for_dir_order_stable(provider):
    """Two consecutive calls return stems in the same order (sort is deterministic)."""
    stems_first = [
        md.stem for md in build_metadata_list_for_dir(DATA_DIR, provider)
        if md is not None
    ]
    stems_second = [
        md.stem for md in build_metadata_list_for_dir(DATA_DIR, provider)
        if md is not None
    ]
    assert stems_first == stems_second, (
        "build_metadata_list_for_dir returned different ordering on consecutive calls"
    )


def test_build_metadata_list_for_dir_stems_match_sorted_filenames(provider):
    """Stems in the result list match the stems of recursively-sorted filenames.

    Validates that the list is aligned with ColPaliModel.index() which iterates
    files via rglob("*") sorted by relative path.
    """
    sorted_files = sorted(
        (p for p in DATA_DIR.rglob("*") if p.is_file()),
        key=lambda p: p.relative_to(DATA_DIR),
    )
    md_list = [
        md for md in build_metadata_list_for_dir(DATA_DIR, provider)
        if md is not None
    ]
    assert len(md_list) == len(sorted_files)
    for expected_path, md in zip(sorted_files, md_list):
        assert md.stem == expected_path.stem, (
            f"Stem mismatch: expected '{expected_path.stem}', got '{md.stem}'"
        )


def test_build_metadata_list_for_dir_recursive_alignment(provider, tmp_path):
    """metadata list is aligned with index()'s rglob enumeration for nested dirs.

    This is a regression test for the bug where build_metadata_list_for_dir used
    non-recursive iterdir() while index() used recursive rglob(), causing a count
    mismatch and ValueError when add_metadata=True on a multi-level dataset.
    """
    # Create a nested directory structure:
    #   tmp_path/
    #     a.pdf  (top-level file)
    #     sub/
    #       b.pdf
    #       deep/
    #         c.pdf
    (tmp_path / "a.pdf").write_bytes(b"%PDF-1.4 fake")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.pdf").write_bytes(b"%PDF-1.4 fake")
    (tmp_path / "sub" / "deep").mkdir()
    (tmp_path / "sub" / "deep" / "c.pdf").write_bytes(b"%PDF-1.4 fake")

    def simple_provider(path):
        return {"stem": path.stem, "ext": path.suffix}

    md_list = build_metadata_list_for_dir(tmp_path, simple_provider)

    # index() would enumerate: a.pdf, sub/b.pdf, sub/deep/c.pdf (sorted by relative path)
    expected_files = sorted(
        (p for p in tmp_path.rglob("*") if p.is_file()),
        key=lambda p: p.relative_to(tmp_path),
    )
    assert len(md_list) == len(expected_files) == 3, (
        f"Expected 3 entries, got {len(md_list)}"
    )
    for i, (expected_path, md) in enumerate(zip(expected_files, md_list)):
        assert md is not None, f"Entry {i} ({expected_path.name}) should not be None"
        assert md.stem == expected_path.stem, (
            f"Entry {i}: stem mismatch — expected '{expected_path.stem}', got '{md.stem}'"
        )


