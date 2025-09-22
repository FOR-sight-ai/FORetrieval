import abc
from typing import Any, Dict, List, Optional
from FORetrieval import Result
from ..agents.digest import Digest


class Generator(abc.ABC):
    backend: str
    model_name: str
    digest: Digest
    raw_context: List[Result]
    context: List[str]

    system_prompt_no_digest: str = (
        "**Generate Response to User Query**\n"
        "**Step 1: Parse Context Information**\n"
        "Extract and utilize relevant knowledge from the provided images as context to answer the user's query.\n"
        "**Step 2: Analyze User Query**\n"
        "Carefully read and comprehend the user's query, pinpointing the key concepts, entities, and intent behind the question.\n"
        "**Step 3: Determine Response**\n"
        "If the answer to the user's query can be directly inferred from the context information, provide a concise and accurate response in the same language as the user's query.\n"
        "**Step 4: Consider the images' relevance when determining the response. The list of images is ordered by relevance to the user's query (most relevant first).**\n"
        "The score is a measure of the relevance of the context to the user's query. The higher the score, the more relevant the context is to the user's query.**\n"
        "**Step 5: Handle Uncertainty**\n"
        "If the answer is not clear, ask the user for clarification to ensure an accurate response.\n"
        "**Step 6: Avoid Context Attribution**\n"
        "When formulating your response, do not indicate that the information was derived from the context.\n"
        "**Step 7: Respond in User's Language**\n"
        "Maintain consistency by ensuring the response is in the same language as the user's query.\n"
        "**Step 8: Provide Response**\n"
        "Generate a clear, concise, and informative response to the user's query, adhering to the guidelines outlined above.\n"
    )

    system_prompt_with_digest: str = (
        "**Generate Response to User Query**\n"
        "**Step 0: Check if the user is referring to previous queries/answers (called 'digest'). Relevant elements might be in the digest.**\n"
        "**Step 1: Parse Context Information**\n"
        "Extract and utilize relevant knowledge from the provided images as context to answer the user's query.\n"
        "**Step 2: Analyze User Query**\n"
        "Carefully read and comprehend the user's query, pinpointing the key concepts, entities, and intent behind the question.\n"
        "**Step 3: Use Digest**\n"
        "Use the digest to provide additional context to the user's query. It contains a summary of the conversation so far along with the results of the previous queries.\n"
        "**Step 4: Determine Response**\n"
        "If the answer to the user's query can be directly inferred from the context information, provide a concise and accurate response in the same language as the user's query.\n"
        "**Step 5: Consider the images' relevance when determining the response. The list of images is ordered by relevance to the user's query (most relevant first).**\n"
        "The score is a measure of the relevance of the context to the user's query. The higher the score, the more relevant the context is to the user's query.**\n"
        "**Step 6: Handle Uncertainty**\n"
        "If the answer is not clear, ask the user for clarification to ensure an accurate response.\n"
        "**Step 7: Avoid Context Attribution**\n"
        "When formulating your response, do not indicate that the information was derived from the context.\n"
        "**Step 8: Respond in User's Language**\n"
        "Maintain consistency by ensuring the response is in the same language as the user's query.\n"
        "**Step 9: Provide Response**\n"
        "Generate a clear, concise, and informative response to the user's query, adhering to the guidelines outlined above.\n"
    )

    system_prompt: str = system_prompt_no_digest

    @abc.abstractmethod
    def __init__(self, backend: str, model_name: str, digest: Digest, **kwargs):
        self.backend = backend
        self.model_name = model_name
        self.digest = digest
        pass

    def set_retrieval_context(self, raw_context: List[Result]):
        self.raw_context = raw_context
        images_scores = [(result.base64, result.score) for result in self.raw_context]
        # sort images by decreasing score
        images_scores.sort(key=lambda x: x[1], reverse=True)
        self.context = [img_score[0] for img_score in images_scores if img_score[0]]

    @abc.abstractmethod
    def _check_model(self, model_name: str):
        pass

    @abc.abstractmethod
    def generate(self, prompt: str) -> str:
        pass

    def __call__(
        self, prompt: str, raw_context: List[Result], **kwargs: Dict[str, Any]
    ) -> str:
        self.set_retrieval_context(raw_context)
        if len(self.digest) > 0:
            self.system_prompt = self.system_prompt_with_digest
        else:
            self.system_prompt = self.system_prompt_no_digest
        return self.generate(prompt)

    @classmethod
    def get_generator(
        cls, backend: str, model_name: str, api_key: Optional[str] = None, **kwargs
    ) -> "Generator":
        if backend == "ollama":
            from .backends.ollama import OllamaGenerator

            return OllamaGenerator(model_name, **kwargs)
        elif backend == "mistral":
            from .backends.mistral import MistralGenerator

            return MistralGenerator(model_name, api_key, **kwargs)
        else:
            raise ValueError(f"Unsupported backend: {backend}")
