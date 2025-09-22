from typing import Any, Dict, List
from ..agentic_retrieval import AgenticRetriever
from FORetrieval import MultiModalRetrieverModel, Result
from .common import get_litellm_model


class MistralAgenticRetriever(AgenticRetriever):
    def __init__(
        self,
        model: str,
        retriever: MultiModalRetrieverModel,
        image_analyzer_params: Dict[str, Any],
        max_steps: int,
        api_key: str,
        **kwargs,
    ):
        litellm_model = get_litellm_model("mistral", model, api_key=api_key, **kwargs)
        top_k = kwargs.pop("top_k")
        super().__init__(
            litellm_model,
            retriever,
            max_steps,
            top_k,
            image_analyzer_params,
            **kwargs,
        )

    def __call__(
        self, query: str, digest: List[str] | None = None, **kwargs
    ) -> List[Result] | None:
        results = super().__call__(query, digest, **kwargs)
        return results
