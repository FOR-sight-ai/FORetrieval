_all__ = []

try:
    from foretrieval.integrations._langchain import FORetrievalLangChain

    _all__.append("FORetrievalLangChainRetriever")
except ImportError:
    pass
