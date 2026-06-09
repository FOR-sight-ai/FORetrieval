_all__ = []

try:
    from foretrieval.integrations._langchain import FORetrievalLangChain  # noqa: F401

    _all__.append("FORetrievalLangChainRetriever")
except ImportError:
    pass
