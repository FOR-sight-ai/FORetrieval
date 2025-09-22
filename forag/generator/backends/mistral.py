import os
from time import sleep
from typing import Optional
from ..generator import Generator

try:
    from mistralai import Mistral, SDKError
    from mistralai.models import UserMessage, SystemMessage
except ImportError:
    raise ImportError(
        "Please install mistral dependencies with `pip install forag[mistral]`"
    )


class MistralGenerator(Generator):
    def __init__(self, model_name: str, api_key: Optional[str] = None, **kwargs):
        super().__init__("mistral", model_name, **kwargs)
        if not api_key:
            api_key = os.environ["MISTRAL_API_KEY"]
        self.client = Mistral(api_key=api_key)
        self._check_model(self.model_name)
        self.max_attempts = kwargs.pop("max_attempts", 5)
        if "system_prompt" in kwargs:
            self.system_prompt = kwargs.pop("system_prompt")

    def _check_model(self, model_name: str):
        available_models = self.client.models.list().data

        # Check if the model is available
        try:
            assert len(available_models) > 0
            assert any(model.id == model_name for model in available_models)
        except AssertionError:
            if len(available_models) == 0:
                raise ValueError("No models are available.")
            else:
                available_model_ids = [model.id for model in available_models]
                raise ValueError(
                    f"Model {self.model_name} is not available. Available models are: {available_model_ids}"
                )

        # Get the capabilities of the model from the available models
        model_capabilities = [
            model.capabilities for model in available_models if model.id == model_name
        ][0]
        try:
            assert model_capabilities.vision
        except AssertionError:
            raise ValueError(f"Model {self.model_name} does not support vision.")

    def generate(self, prompt: str) -> str:
        system_message = SystemMessage(content=self.system_prompt)
        user_message = UserMessage(
            content=[
                {"type": "text", "text": prompt},
                *[
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image_data}"},
                    }
                    for image_data in self.context
                ],
            ]
        )
        all_messages = [system_message, user_message]
        if len(self.digest) > 0:
            digest_message = UserMessage(content=self.digest.get_context(prompt))
            print(f"Digest:\n{digest_message}\n")
            all_messages = [system_message, digest_message, user_message]
        attempts = 0
        while attempts < self.max_attempts:
            try:
                chat_response = self.client.chat.complete(
                    model=self.model_name,
                    messages=all_messages,
                )
                break
            except SDKError:
                sleep(10)
                attempts += 1

        return chat_response.choices[0].message.content
