"""FORetrieval public API.

Imports are lazy (PEP 562) so that sub-packages such as the vector-DB
server — which only need ``foretrieval.vector_db_server.*`` — can be
imported without pulling in ColPali, transformers, or PyTorch.  Any
code that does ``from foretrieval import MultiModalRetrieverModel`` or
``import foretrieval; foretrieval.MultiModalRetrieverModel`` continues
to work identically.
"""

from __future__ import annotations

__all__ = ["MultiModalRetrieverModel", "ai_metadata_provider_factory"]


def __getattr__(name: str):
    if name == "MultiModalRetrieverModel":
        from .retriever import MultiModalRetrieverModel
        return MultiModalRetrieverModel
    if name == "ai_metadata_provider_factory":
        from .metadata import ai_metadata_provider_factory
        return ai_metadata_provider_factory
    raise AttributeError(f"module 'foretrieval' has no attribute {name!r}")
