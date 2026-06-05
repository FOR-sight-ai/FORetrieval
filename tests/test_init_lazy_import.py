"""Tests for foretrieval.__init__ — specifically the lazy-import guarantee.

The vector-DB server is started by Python importing
``foretrieval.vector_db_server.server_main`` which triggers the
top-level ``foretrieval/__init__.py``.  If that __init__ eagerly
imports ``colpali.py``, it pulls in ``colpali_engine``, ``transformers``
and ``torch._dynamo``, which crashes on the CPU-only Docker image with:

    AssertionError: Artifact of type=precompile already registered
                    in mega-cache artifact factory

The fix is a lazy ``__getattr__``-based __init__ so the heavy ML stack
is only loaded when the caller explicitly requests it.
"""

from __future__ import annotations

import importlib
import sys


def test_foretrieval_init_does_not_import_colpali():
    """Importing ``foretrieval`` must NOT pull in colpali as a side-effect.

    This is the regression guard for the vector-DB server startup crash:
    the server only needs ``foretrieval.vector_db_server.*`` and must be
    able to start without torch / colpali_engine being importable.
    """
    # Remove any already-cached foretrieval / colpali modules so the
    # test is independent of import order within the test session.
    to_remove = [k for k in sys.modules if k.startswith("foretrieval") or "colpali" in k]
    for k in to_remove:
        sys.modules.pop(k, None)

    colpali_before = {k for k in sys.modules if "colpali" in k}

    import foretrieval  # noqa: F401

    colpali_after = {k for k in sys.modules if "colpali" in k}

    new_colpali = colpali_after - colpali_before
    assert not new_colpali, (
        f"Importing 'foretrieval' pulled in colpali modules: {new_colpali}. "
        "The __init__.py must use lazy imports so the vector-DB server "
        "can start without torch/colpali_engine being available."
    )


def test_foretrieval_multimodal_retriever_model_accessible():
    """MultiModalRetrieverModel must still be reachable via the package."""
    # Re-import fresh
    to_remove = [k for k in sys.modules if k.startswith("foretrieval")]
    for k in to_remove:
        sys.modules.pop(k, None)

    import foretrieval
    cls = foretrieval.MultiModalRetrieverModel
    assert cls is not None
    # Confirm it is the real class (not a stub)
    from foretrieval.retriever import MultiModalRetrieverModel as Direct
    assert cls is Direct


def test_foretrieval_ai_metadata_provider_accessible():
    """ai_metadata_provider_factory must still be reachable via the package."""
    to_remove = [k for k in sys.modules if k.startswith("foretrieval")]
    for k in to_remove:
        sys.modules.pop(k, None)

    import foretrieval
    fn = foretrieval.ai_metadata_provider_factory
    assert callable(fn)


def test_foretrieval_unknown_attr_raises():
    """Accessing a non-existent attribute must raise AttributeError, not hang."""
    to_remove = [k for k in sys.modules if k.startswith("foretrieval")]
    for k in to_remove:
        sys.modules.pop(k, None)

    import foretrieval
    import pytest
    with pytest.raises(AttributeError, match="foretrieval"):
        _ = foretrieval.this_does_not_exist
