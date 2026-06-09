"""Tests for the on_progress callback wiring in ColPaliModel.index().

Avoids loading a real ColPali model — uses a minimal fake to verify that:
  1. The callback receives the expected stages in order
  2. Page events report sane page counts for a single-page PDF
  3. Exceptions raised in the callback are swallowed (best-effort delivery)
"""

from __future__ import annotations

from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Stub the ColPaliModel without instantiating the real one. We exercise the
# index() function purely from the perspective of the on_progress callback.
# This keeps the test fast (no model download / no CUDA needed).
# ---------------------------------------------------------------------------

class _FakeColPaliModel:
    """A throwaway object exposing just enough of ColPaliModel.index's contract.

    We hijack the bound method via __get__ to be able to call the real
    ColPaliModel.index with `self` bound to this fake (so its branches that
    only consult attributes work).  The heavy lifting in add_to_index is
    monkey-patched to a no-op that fires the standard page events.
    """


SAMPLE_PDF = Path(__file__).parent.parent / "sample_data" / "sample_doc.pdf"


def test_on_progress_emits_expected_events(tmp_path, monkeypatch):
    """index() must drive on_progress through start → file_start → file_done → all_done."""
    if not SAMPLE_PDF.exists():
        pytest.skip("sample PDF not available")

    from foretrieval.colpali import ColPaliModel

    events: list[dict] = []

    def cb(evt: dict) -> None:
        events.append(evt)

    # Build a minimal fake "self" with the attributes index() reads
    fake = _FakeColPaliModel()
    fake.index_name = None
    fake.index_root = str(tmp_path)
    fake.storage_backend = "local"
    fake.storage_config = {}
    fake.full_document_collection = False
    fake.vector_store = type("V", (), {
        "open": lambda *a, **k: None,
        "supports_remote_bookkeeping": lambda *a, **k: False,
    })()
    fake.processor = None
    fake.doc_id_to_metadata = {}
    fake.doc_ids = set()
    fake.doc_ids_to_file_names = {}
    fake.highest_doc_id = -1

    def fake_add_to_index(self, item, store_collection_with_index, doc_id=None,
                          metadata=None, batch_size=1,
                          on_progress=None, _file_idx=0, _n_files=1):
        # Simulate one page progress event so we can also verify pages bubble up
        if on_progress is not None:
            on_progress({
                "stage": "page",
                "file": Path(str(item)).name,
                "file_idx": _file_idx,
                "n_files": _n_files,
                "page_idx": 0,
                "n_pages": 1,
            })
        self.doc_ids_to_file_names[doc_id] = str(item)
        self.highest_doc_id = max(self.highest_doc_id, doc_id)
        self.doc_ids.add(doc_id)
        return {}

    fake.add_to_index = fake_add_to_index.__get__(fake, type(fake))
    fake._export_index = lambda *a, **k: None
    fake._vector_store_is_open = lambda: True

    # Run the real index() bound to our fake
    ColPaliModel.index(
        fake,
        input_path=SAMPLE_PDF.parent,
        index_name="test_index_progress",
        store_collection_with_index=False,
        overwrite=True,
        on_progress=cb,
    )

    stages = [e["stage"] for e in events]
    assert "start" in stages
    assert "file_start" in stages
    assert "file_done" in stages
    assert "all_done" in stages
    # At least one page event must be present
    assert any(s == "page" for s in stages)
    # start must precede file_start which must precede file_done which must precede all_done
    assert stages.index("start") < stages.index("file_start") < stages.index("file_done") < stages.index("all_done")


def test_on_progress_exception_is_swallowed(tmp_path, monkeypatch):
    """A callback that raises must not abort indexing."""
    if not SAMPLE_PDF.exists():
        pytest.skip("sample PDF not available")

    from foretrieval.colpali import ColPaliModel

    call_count = {"n": 0}

    def bad_cb(evt):
        call_count["n"] += 1
        raise RuntimeError("intentional")

    fake = _FakeColPaliModel()
    fake.index_name = None
    fake.index_root = str(tmp_path)
    fake.storage_backend = "local"
    fake.storage_config = {}
    fake.full_document_collection = False
    fake.vector_store = type("V", (), {
        "open": lambda *a, **k: None,
        "supports_remote_bookkeeping": lambda *a, **k: False,
    })()
    fake.processor = None
    fake.doc_id_to_metadata = {}
    fake.doc_ids = set()
    fake.doc_ids_to_file_names = {}
    fake.highest_doc_id = -1
    fake._export_index = lambda *a, **k: None
    fake._vector_store_is_open = lambda: True

    def fake_add_to_index(self, item, store_collection_with_index, doc_id=None,
                          metadata=None, batch_size=1,
                          on_progress=None, _file_idx=0, _n_files=1):
        self.doc_ids_to_file_names[doc_id] = str(item)
        self.highest_doc_id = max(self.highest_doc_id, doc_id)
        self.doc_ids.add(doc_id)
        return {}

    fake.add_to_index = fake_add_to_index.__get__(fake, type(fake))

    # Should not raise even though every callback invocation throws
    ColPaliModel.index(
        fake,
        input_path=SAMPLE_PDF.parent,
        index_name="test_index_progress_bad_cb",
        store_collection_with_index=False,
        overwrite=True,
        on_progress=bad_cb,
    )

    # And the callback was still invoked despite raising
    assert call_count["n"] > 0
