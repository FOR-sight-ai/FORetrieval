from typing import Any, Dict
from FORetrieval import MultiModalRetrieverModel
from smolagents import CodeAgent, AgentMemory
from .backends.common import get_litellm_model
from .image_analysis import ImageAnalyzer
from .digest import Digest
from .agentic_retrieval import AgenticRetriever


def check_response(response: Any, memory: AgentMemory) -> bool:
    result = True
    try:
        assert isinstance(response, str)
    except AssertionError:
        result = False
        raise TypeError("Response must be a string")
    finally:
        return result


task_no_digest = (
    "**Generate Response to User Query**\n"
    "**Step 1: Analyze User Query**\n"
    "Carefully read and comprehend the user's query, pinpointing the key concepts, entities, and intent behind the question.\n"
    "**Step 2: Determine if extra context is required**\n"
    "If and only if deemed necessary, task the retrieval agent to retrieve additional context to answer the user's query. If not, skip to step 7.\n"
    "**Step 3: Parse retrieved context information**\n"
    "The context is a list of results containing images and their relevance scores.\n"
    "Extract and utilize relevant knowledge from the images as context to answer the user's query. Use the image analyzer for image analysis.\n"
    "**Step 4: Determine Response**\n"
    "If the answer to the user's query can be directly inferred from the context information, provide a concise and accurate response in the same language as the user's query.\n"
    "**Step 5: Consider the images' relevance when determining the response.**\n"
    "The score is a measure of the relevance of the context to the user's query. The higher the score (max 100), the more relevant the context is to the user's query.**\n"
    "**Step 6: Avoid Context Attribution**\n"
    "When formulating your response, do not indicate that the information was derived from the context.\n"
    "**Step 7: Handle Uncertainty**\n"
    "If the answer is not clear, ask the user for clarification to ensure an accurate response.\n"
    "**Step 8: Respond in User's Language**\n"
    "Maintain consistency by ensuring the response is in the same language as the user's query.\n"
    "**Step 9: Provide Response**\n"
    "Generate a clear, concise, and informative response to the user's query, adhering to the guidelines outlined above.\n"
    "Your final answer needs to be a string."
)

task_with_digest = (
    "**Generate Response to User Query**\n"
    "**Step 1: Analyze User Query**\n"
    "Carefully read and comprehend the user's query, pinpointing the key concepts, entities, and intent behind the question.\n"
    "**Step 2: Use Digest**\n"
    "Use the digest to provide additional context to the user's query. It contains a summary of the conversation so far along with the results of the previous queries.\n"
    "**Step 3: Determine if extra context is required**\n"
    "If and only if deemed necessary, task the retrieval agent to retrieve additional context to answer the user's query. If not, skip to step 7.\n"
    "**Step 4: Parse retrieved context information**\n"
    "The context is a list of results containing images and their relevance scores.\n"
    "Extract and utilize relevant knowledge from the images as context to answer the user's query. Use the image analyzer for image analysis.\n"
    "**Step 5: Determine Response**\n"
    "If the answer to the user's query can be directly inferred from the context information, provide a concise and accurate response in the same language as the user's query.\n"
    "**Step 6: Consider the images' relevance when determining the response.**\n"
    "The score is a measure of the relevance of the context to the user's query. The higher the score (max 100), the more relevant the context is to the user's query.**\n"
    "**Step 7: Handle Uncertainty**\n"
    "If the answer is not clear, ask the user for clarification to ensure an accurate response.\n"
    "**Step 8: Avoid Context Attribution**\n"
    "When formulating your response, do not indicate that the information was derived from the context.\n"
    "**Step 9: Respond in User's Language**\n"
    "Maintain consistency by ensuring the response is in the same language as the user's query.\n"
    "**Step 10: Provide Response**\n"
    "Generate a clear, concise, and informative response to the user's query, adhering to the guidelines outlined above.\n"
    "Your final answer needs to be a string."
)


class AgenticGenerator:
    max_steps: int
    generation_agent: CodeAgent
    retrieval_agent: AgenticRetriever
    image_analyzer: ImageAnalyzer
    digest: Digest
    task: str = task_no_digest

    def __init__(
        self,
        backend: str,
        model_name: str,
        max_steps: int,
        retrieval_agent: AgenticRetriever,
        image_analyzer_params: Dict[str, Any],
        retriever: MultiModalRetrieverModel,
        digest: Digest,
        **kwargs,
    ):
        self.max_steps = max_steps
        self.task = kwargs.get("task", task_no_digest)
        self.retrieval_agent = retrieval_agent
        self.digest = digest

        model = get_litellm_model(backend, model_name, **kwargs)
        self.image_analyzer = ImageAnalyzer.get_image_analyzer(
            image_analyzer_params, retriever
        )
        self.generation_agent = CodeAgent(
            tools=[self.image_analyzer],
            model=model,
            add_base_tools=False,
            managed_agents=[self.retrieval_agent.agent],
            planning_interval=5,
            additional_authorized_imports=["numpy", "pandas"],
            final_answer_checks=[check_response],
        )

    def generate(self, query: str) -> str | Any:
        if len(self.digest) > 0:
            digest_context = self.digest.get_context(
                query
            )  # TODO: consider adding index from retriever
            self.task = task_with_digest
        else:
            digest_context = None
        additional_args = (
            {"query": query, "digest": digest_context}
            if digest_context
            else {"query": query}
        )
        result = self.generation_agent.run(
            self.task,
            max_steps=self.max_steps,
            additional_args=additional_args,
        )
        return result

    def __call__(self, query: str) -> str:
        return self.generate(query)
