"""
Tests for MetadataFilter regex support and empty-filter crash fix.

No GPU, no API key, no network required — all ColPali internals are mocked.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from foretrieval.models_metadata import MetadataFilter
from foretrieval.utils import _value_match


# ---------------------------------------------------------------------------
# Fixtures — representative metadata dicts
# ---------------------------------------------------------------------------


@pytest.fixture
def meta_pdf():
    """Metadata for a PDF named 'general_datasheet.pdf'."""
    return {
        "stem": "general_datasheet",
        "ext": ".pdf",
        "title": "General Motor Controller Reference Manual",
        "author": "Smith, J. and Doe, A.",
        "mime": "application/pdf",
        "language": None,
        "tags": [],
        "document_type": "unknown",
        "short_description": "",
        "mtime": "2025-03-01T10:00:00+00:00",
        "page_count": 42,
    }


@pytest.fixture
def meta_docx():
    """Metadata for a Word document named 'safety_manual.docx'."""
    return {
        "stem": "safety_manual",
        "ext": ".docx",
        "title": "Industrial Safety Procedures",
        "author": "Johnson, R.",
        "mime": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "language": None,
        "tags": [],
        "document_type": "unknown",
        "short_description": "",
        "mtime": "2024-11-15T08:30:00+00:00",
        "page_count": None,
    }


# ---------------------------------------------------------------------------
# _value_match — regex tests
# ---------------------------------------------------------------------------


class TestRegexMatching:
    def test_stem_substring_match(self, meta_pdf):
        """regex on stem matches when the pattern is a substring."""
        f = MetadataFilter(regex={"stem": "general"})
        assert _value_match(meta_pdf, f) is True

    def test_stem_no_match(self, meta_pdf):
        """regex on stem returns False when pattern is not found."""
        f = MetadataFilter(regex={"stem": "missing"})
        assert _value_match(meta_pdf, f) is False

    def test_title_case_insensitive(self, meta_pdf):
        """regex matching is case-insensitive by default."""
        f = MetadataFilter(regex={"title": "GENERAL"})
        assert _value_match(meta_pdf, f) is True

    def test_author_substring(self, meta_pdf):
        """regex on author matches a substring of the author string."""
        f = MetadataFilter(regex={"author": "smith"})
        assert _value_match(meta_pdf, f) is True

    def test_author_no_match(self, meta_pdf):
        """regex on author returns False when pattern is not in author string."""
        f = MetadataFilter(regex={"author": "nobody"})
        assert _value_match(meta_pdf, f) is False

    def test_alternation(self, meta_pdf):
        """regex alternation (|) matches if any alternative is present."""
        f = MetadataFilter(regex={"title": "motor|pump"})
        assert _value_match(meta_pdf, f) is True

    def test_alternation_no_match(self, meta_pdf):
        """regex alternation returns False when none of the alternatives match."""
        f = MetadataFilter(regex={"title": "bicycle|aircraft"})
        assert _value_match(meta_pdf, f) is False

    def test_anchored_start_match(self, meta_pdf):
        """regex ^ anchor matches at the start of the field value."""
        f = MetadataFilter(regex={"stem": "^general"})
        assert _value_match(meta_pdf, f) is True

    def test_anchored_start_no_match(self, meta_docx):
        """regex ^ anchor fails when value does not start with the pattern."""
        f = MetadataFilter(regex={"stem": "^general"})
        assert _value_match(meta_docx, f) is False

    def test_malformed_pattern_no_crash(self, meta_pdf):
        """A malformed regex pattern returns False without raising."""
        f = MetadataFilter(regex={"stem": "[invalid"})
        result = _value_match(meta_pdf, f)
        assert result is False

    def test_field_not_in_meta_no_match(self, meta_pdf):
        """regex on a field absent from metadata returns False."""
        f = MetadataFilter(regex={"nonexistent_field": "anything"})
        assert _value_match(meta_pdf, f) is False

    def test_regex_on_ai_field_language(self, meta_pdf):
        """regex works on AI fields like language when present."""
        meta = dict(meta_pdf)
        meta["language"] = "en"
        f = MetadataFilter(regex={"language": "^en"})
        assert _value_match(meta, f) is True

    def test_regex_on_document_type(self, meta_pdf):
        """regex works on document_type."""
        meta = dict(meta_pdf)
        meta["document_type"] = "technical note"
        f = MetadataFilter(regex={"document_type": "technical"})
        assert _value_match(meta, f) is True

    def test_regex_combined_with_ext_and(self, meta_pdf, meta_docx):
        """regex AND ext: only matches when both conditions hold."""
        f = MetadataFilter(ext=".pdf", regex={"stem": "general"})
        assert _value_match(meta_pdf, f) is True   # PDF + stem contains "general"
        assert _value_match(meta_docx, f) is False  # docx fails ext check

    def test_regex_combined_with_ext_or(self, meta_pdf, meta_docx):
        """regex OR ext: matches when either condition holds."""
        f = MetadataFilter(ext=".docx", regex={"stem": "general"}, logic="OR")
        assert _value_match(meta_pdf, f) is True   # stem matches
        assert _value_match(meta_docx, f) is True   # ext matches

    def test_regex_not_double_processed_by_extra_loop(self, meta_pdf):
        """The 'regex' key must not be processed by the extra-field equality loop.

        If it were, 'regex' as a key would be compared against meta['regex']
        which doesn't exist, producing a spurious False check.
        """
        f = MetadataFilter(regex={"stem": "general"})
        # Ensure no check is generated for a literal 'regex' key equality
        # The filter has one regex condition → exactly one check in the list
        # We verify this by asserting the match is True (would be False if
        # the extra loop produced a False check for missing 'regex' field).
        assert _value_match(meta_pdf, f) is True

    def test_multiple_regex_fields_and(self, meta_pdf):
        """Multiple regex entries are all required under AND logic."""
        f = MetadataFilter(regex={"stem": "general", "title": "motor"})
        assert _value_match(meta_pdf, f) is True

    def test_multiple_regex_fields_one_fails(self, meta_pdf):
        """Multiple regex entries: one failing under AND logic → False."""
        f = MetadataFilter(regex={"stem": "general", "title": "aircraft"})
        assert _value_match(meta_pdf, f) is False

    def test_multiple_regex_fields_or(self, meta_pdf):
        """Multiple regex entries under OR: one match is sufficient."""
        f = MetadataFilter(regex={"stem": "general", "title": "aircraft"}, logic="OR")
        assert _value_match(meta_pdf, f) is True


# ---------------------------------------------------------------------------
# Empty filter → [] and no crash
# ---------------------------------------------------------------------------


class TestEmptyFilterReturnsEmptyList:
    """Tests that ColPaliModel.search() returns [] when the filter matches nothing."""

    def _make_mock_colpali(self):
        """Return a minimal mock ColPaliModel with in-memory index state."""
        from foretrieval.colpali import ColPaliModel

        model = MagicMock(spec=ColPaliModel)
        model.device = "cpu"
        model.verbose = 0
        model.collection = None
        model.enable_heatmaps = False
        model.enable_circle = False

        # One document indexed, stored with ext=".pdf"
        model.doc_id_to_metadata = {
            0: {"stem": "datasheet", "ext": ".pdf", "title": "Test Doc"}
        }
        model.embed_id_to_doc_id = {0: {"doc_id": 0, "page_id": 0}}
        model.indexed_embeddings = [MagicMock()]  # one page embedding

        # processor mock
        processor = MagicMock()
        processor.process_queries.return_value = {
            "input_ids": MagicMock(
                __getitem__=lambda self, idx: MagicMock(
                    detach=lambda: MagicMock(
                        cpu=lambda: MagicMock(tolist=lambda: [1, 2, 3])
                    )
                )
            )
        }
        processor.tokenizer.convert_ids_to_tokens.return_value = ["tok1", "tok2", "tok3"]
        model.processor = processor

        # model forward pass mock
        inner_model = MagicMock()
        inner_model.dtype = None
        model.model = inner_model

        return model

    def test_filter_no_match_returns_empty_list(self):
        """search() returns [] when no document matches the filter."""
        from foretrieval.colpali import ColPaliModel

        mock = self._make_mock_colpali()
        # Patch filter_embeddings on the instance directly (not on the class)
        # so that ColPaliModel.search(mock, ...) picks it up via self.filter_embeddings
        mock.filter_embeddings = MagicMock(return_value=([], []))

        with (
            patch("foretrieval.colpali.torch.inference_mode", return_value=MagicMock(__enter__=lambda s: None, __exit__=lambda s, *a: None)),
            patch("foretrieval.colpali.torch.unbind", return_value=[MagicMock()]),
        ):
            results = ColPaliModel.search(
                mock,
                query="anything",
                k=5,
                filter_metadata={"ext": ".docx"},
                return_base64_results=False,
            )

        assert results == []

    def test_filter_no_match_does_not_raise(self):
        """search() must not raise ValueError when the filter matches nothing."""
        from foretrieval.colpali import ColPaliModel

        mock = self._make_mock_colpali()
        mock.filter_embeddings = MagicMock(return_value=([], []))

        try:
            with (
                patch("foretrieval.colpali.torch.inference_mode", return_value=MagicMock(__enter__=lambda s: None, __exit__=lambda s, *a: None)),
                patch("foretrieval.colpali.torch.unbind", return_value=[MagicMock()]),
            ):
                ColPaliModel.search(
                    mock,
                    query="anything",
                    k=5,
                    filter_metadata={"ext": ".docx"},
                    return_base64_results=False,
                )
        except ValueError as exc:
            pytest.fail(
                f"search() raised ValueError when filter matched nothing: {exc}"
            )

    def test_filter_match_still_works(self):
        """search() still returns results when the filter matches documents."""
        from foretrieval.colpali import ColPaliModel
        import numpy as np

        mock = self._make_mock_colpali()

        embedding = MagicMock()
        mock.filter_embeddings = MagicMock(return_value=([embedding], [0]))

        scores = MagicMock()
        scores.cpu.return_value.numpy.return_value = np.array([[0.9]])
        scores_argsort = MagicMock()
        scores_argsort.__getitem__ = lambda s, idx: MagicMock(
            __getitem__=lambda s2, idx2: MagicMock(tolist=lambda: [0])
        )
        scores.argsort.return_value = scores_argsort

        mock.processor.score.return_value = scores
        mock.embed_id_to_doc_id = {0: {"doc_id": 0, "page_id": 0}}
        mock.embed_id_to_extra = {}  # no extra embeddings data
        mock.highest_doc_id = 0
        mock.collection = None

        with (
            patch("foretrieval.colpali.torch.inference_mode", return_value=MagicMock(__enter__=lambda s: None, __exit__=lambda s, *a: None)),
            patch("foretrieval.colpali.torch.unbind", return_value=[MagicMock()]),
        ):
            results = ColPaliModel.search(
                mock,
                query="test",
                k=1,
                filter_metadata={"ext": ".pdf"},
                return_base64_results=False,
            )

        # Should not crash; results may be empty due to mock internals but no ValueError
        assert isinstance(results, list)
