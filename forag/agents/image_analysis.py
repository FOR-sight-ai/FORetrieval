from typing import Any, Dict, List
from smolagents import LiteLLMModel, Tool, ChatMessage, MessageRole
from FORetrieval import Result, MultiModalRetrieverModel
from .backends.common import get_litellm_model, rate_limit_handler


class ImageAnalyzer(Tool):
    name = "image_analyzer"
    description = "Analyze an image and return a short description of its content."
    inputs = {
        "result": {
            "type": "object",
            "description": "The result of the retrieval (a single object, not an array of objects, as this tool is meant to be used on a single image at a time. If you have multiple images, you should call this tool multiple times), containing the doc_id and page_num of the image to be analyzed.",
        }
    }

    output_type = "string"

    def __init__(self, model: LiteLLMModel, rag: MultiModalRetrieverModel, **kwargs):
        self.model = model
        self.rag = rag
        # assert litellm.supports_vision(model=self.model.model_id), (
        #     f"The model {self.model.model_id} does not support vision."
        # ) #TODO: Fix this
        super().__init__(**kwargs)

    def forward(self, result: Dict[str, Any] | List[Dict[str, Any]]) -> str:
        """Analyze the image and return a description of its content."""
        if isinstance(result, list):
            result = result[0]
        image_data = self.rag.fetch(Result(**result)).base64
        system_prompt = (
            "You are an image analysis assistant. You will be given an image and you will describe its content in detail."
            "Your description will comply with the following template (DO NOT EXCEED THE NUMBER OF MAX WORDS):\n"
            "Doc_id: <doc_id> Page_num: <page_num> \n"
            "Description: <150 words max description.>"
        )
        system_msg = ChatMessage(role=MessageRole.SYSTEM, content=system_prompt)
        user_prompt = f"Describe the content of the image in detail while respecting the template and the word limit. Doc_id: {result['doc_id']} Page_num: {result['page_num']}"
        user_msg = ChatMessage(
            role=MessageRole.USER,
            content=[
                {"type": "text", "text": user_prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{image_data}"},
                },
            ],
        )
        with rate_limit_handler(wait_seconds=10):
            response = str(self.model([system_msg, user_msg]).content)

        # keep 200 words only no matter what
        response = " ".join(response.split()[:200])

        return response

    @classmethod
    def get_image_analyzer(
        cls, params: Dict[str, Any], rag: MultiModalRetrieverModel, **kwargs
    ) -> "ImageAnalyzer":
        model = get_litellm_model(
            backend=params["provider"], model_name=params["name"], **kwargs
        )
        return ImageAnalyzer(model, rag, **kwargs)
