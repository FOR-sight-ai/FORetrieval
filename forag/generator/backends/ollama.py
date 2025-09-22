from ..generator import Generator

try:
    import ollama
except ImportError:
    raise ImportError(
        "Please install ollama dependencies with `pip install forag[ollama]`"
    )


class OllamaGenerator(Generator):
    url_base: str

    def __init__(self, model_name: str, **kwargs):
        super().__init__("ollama", model_name, **kwargs)
        self.url_base = kwargs.get("url_base", "http://localhost:11434")
        self.client = ollama.Client(self.url_base)

        self._check_model(self.model_name)

        if "system_prompt" in kwargs:
            self.system_prompt = kwargs.pop("system_prompt")

    def _check_model(self, model_name: str):
        available_models = self.client.list()["models"]

        # Check if the model is available
        try:
            assert len(available_models) > 0
            assert any(model.model == model_name for model in available_models)
        except AssertionError:
            if len(available_models) == 0:
                raise ValueError("No models are available.")
            else:
                raise ValueError(
                    f"Model {self.model_name} is not available. Available models are: {available_models}"
                )

        # Check if the model supports vision
        model_info = self.client.show(model_name)
        try:
            assert "vision" in model_info["capabilities"]
        except AssertionError:
            raise ValueError(f"Model {self.model_name} does not support vision.")

    def generate(self, prompt: str) -> str:
        if len(self.digest) > 0:
            full_prompt = self.digest.get_context(prompt) + "\n\n" + prompt
        else:
            full_prompt = prompt
        response = self.client.generate(
            model=self.model_name,
            prompt=full_prompt,
            system=self.system_prompt,
            images=self.context,
        )
        return response["response"]
