from typing import Any, Dict, List
from ..agentic_retrieval import AgenticRetriever
from FORetrieval import MultiModalRetrieverModel, Result
from .common import get_litellm_model


class OllamaAgenticRetriever(AgenticRetriever):
    def __init__(
        self,
        model: str,
        retriever: MultiModalRetrieverModel,
        image_analyzer_params: Dict[str, Any],
        max_steps: int,
        url: str,
        **kwargs,
    ):
        litellm_model = get_litellm_model("ollama", model, url=url, **kwargs)
        super().__init__(
            litellm_model, retriever, max_steps, top_k=kwargs.pop("top_k"), **kwargs
        )

    def __call__(
        self, query: str, digest: List[str] | None = None, **kwargs
    ) -> List[Result] | None:
        return super().__call__(query, digest, **kwargs)
