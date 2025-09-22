import abc
import json
import logging
from typing import Any, Dict, List
from FORetrieval import MultiModalRetrieverModel, Result
from smolagents import LiteLLMModel, Tool, ToolCallingAgent
from .image_analysis import ImageAnalyzer

logger = logging.getLogger(__name__)


class RetrieverTool(Tool):
    name = "retriever"
    description = (
        "Uses an image-based retriever to retrieve pages of documents that could be most relevant to answer your query."
        "These pages are images of the pages of the documents. The retriever will return the top-k most relevant results as an array of objects."
        "Each object will have a 'doc_id' field, a 'page_num' field, a 'score' field, and an optional 'base64' field containing the image of the page in base64 format."
    )
    inputs = {
        "query": {
            "type": "string",
            "description": "The query to perform. This should be semantically close to your target documents."
            "Use the affirmative form rather than a question.",
        },
        "top_k": {
            "type": "integer",
            "description": "The number of results to return.",
            "default": "10",
        },
    }
    output_type = "array"

    def __init__(self, base_retriever: MultiModalRetrieverModel, **kwargs):
        self.base_retriever = base_retriever
        super().__init__(**kwargs)

    def forward(self, query: str, top_k: int) -> List[Result]:
        """Execute the retrieval based on the provided query. You can select the number of results to return."""
        logger.info(f" *** Retrieving {top_k} results for query: {query}")
        results = self.base_retriever.search(query, top_k, return_base64_results=False)

        return results


class AgenticRetriever(abc.ABC):
    model: LiteLLMModel
    agent: ToolCallingAgent
    retriever: MultiModalRetrieverModel
    retriever_tool: RetrieverTool
    image_analyzer: ImageAnalyzer
    image_analyzer_params: Dict[str, Any]
    max_steps: int
    top_k: int
    task: str = (
        "You are an agentic retriever. You do the following steps:"
        "* Rephrase and split the query if necessary into multiple, simpler queries, focusing on different entities or aspects of the query.\n"
        "* Retrieve the most relevant documents for the given query. Documents are in the form of images.\n"
        "* Analyze the images using the image analyzer tool to understand their content and use this information to look for additional relevant documents if necessary.\n"
        "* You can select the number of intermediate results to return by passing the top_k parameter to the retriever tool"
        "  but do not exceed the specified top_k limit in your final answer.\n"
        "* If you are really unsure about the query or not sure about the results, keep looking by modifying the query and using the retriever again.\n"
        "* Finally, return the results in the form of a list of objects with the following fields: doc_id, page_num, score."
        "  Score is the relevance score of the document according to your final analysis, 0: not relevant, 100: extremely relevant."
        "  Returned documents should all have different scores, even if the difference is small. Sorting is mandatory."
    )

    def __init__(
        self,
        model: LiteLLMModel,
        retriever: MultiModalRetrieverModel,
        max_steps: int,
        top_k: int,
        image_analyzer_params: Dict[str, Any],
        **kwargs,
    ):
        self.model = model
        self.retriever = retriever
        self.top_k = top_k
        self.task = kwargs.get("task", self.task)
        self.retriever_tool = RetrieverTool(retriever, **kwargs)
        self.image_analyzer_params = image_analyzer_params
        self.image_analyzer = ImageAnalyzer.get_image_analyzer(
            self.image_analyzer_params, self.retriever
        )
        self.max_steps = max_steps

        available_tools: List[Tool] = [self.retriever_tool, self.image_analyzer]

        self.agent = ToolCallingAgent(
            tools=available_tools,
            model=model,
            add_base_tools=False,
            # final_answer_checks=[lambda answer, _: self.check_agent_answer(answer)],
            name="AgenticRetriever",
            description=self.task,
        )

    @abc.abstractmethod
    def __call__(
        self, query: str, digest: List[str] | None = None, **kwargs
    ) -> List[Result] | None:
        results: Any = self.agent.run(
            self.task,
            max_steps=self.max_steps,
            additional_args={"query": query, "top_k": self.top_k},
            **kwargs,
        )
        if isinstance(results, str):
            results = json.loads(
                results.replace("'", '"')
            )  # avoid issues with single quotes and json
        results = [Result(**result) for result in results]
        return results

    def check_agent_answer(self, answer: Any) -> bool:
        real_answer = (
            json.loads(answer.replace("'", '"')) if isinstance(answer, str) else answer
        )
        result = True
        if not isinstance(real_answer, list):
            result = False
            raise TypeError("The agent's answer is not a list")
        if not all(isinstance(item, dict) for item in real_answer):
            result = False
            raise TypeError("The agent's answer is not a list of dictionaries")
        if not all(isinstance(item["doc_id"], int) for item in real_answer) or not all(
            isinstance(item["page_num"], int) for item in real_answer
        ):
            result = False
            raise TypeError("The agent's answer has non-integer doc_id or page_num")
        if len(real_answer) > self.top_k:
            result = False
            raise ValueError(f"The agent's answer has more than {self.top_k} items")
        return result

    @classmethod
    def get_agentic_retriever(
        cls,
        backend: str,
        model_name: str,
        image_analyzer_params: Dict[str, Any],
        **kwargs,
    ) -> "AgenticRetriever":
        if backend == "mistral":
            from .backends.mistral import MistralAgenticRetriever

            return MistralAgenticRetriever(
                model_name, image_analyzer_params=image_analyzer_params, **kwargs
            )
        elif backend == "ollama":
            from .backends.ollama import OllamaAgenticRetriever

            return OllamaAgenticRetriever(
                model_name, image_analyzer_params=image_analyzer_params, **kwargs
            )
        else:
            raise ValueError(f"Unsupported backend: {backend}")
