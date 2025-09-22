import os
import time
from contextlib import contextmanager
from smolagents import LiteLLMModel
from litellm.exceptions import RateLimitError
import logging

logger = logging.getLogger(__name__)


def get_litellm_model(backend: str, model_name: str, **kwargs) -> LiteLLMModel:
    if backend == "mistral":
        litellm_model_name = f"mistral/{model_name}"
        api_key = kwargs.pop("api_key", None)
        if api_key is None or api_key == "":
            api_key = os.environ.get("MISTRAL_API_KEY", "")
        requests_per_minute = kwargs.pop("requests_per_minute", None)
        return LiteLLMModel(
            model_id=litellm_model_name,
            api_key=api_key,
            requests_per_minute=requests_per_minute,
        )
    elif backend == "ollama":
        litellm_model_name = f"ollama_chat/{model_name}"
        url = kwargs.pop("url", None)
        num_ctx = kwargs.pop("num_ctx", None)
        return LiteLLMModel(
            model_id=litellm_model_name,
            api_base=url,
            num_ctx=num_ctx,
        )
    else:
        raise ValueError(f"Unknown backend: {backend}")


@contextmanager
def rate_limit_handler(
    *,
    max_attempts: int = 5,
    wait_seconds: int = 5,
):
    """
    Context manager that retries the block on a specified exception.

    Parameters
    ----------
    max_attempts
        The maximum number of times to attempt the block.
    wait_seconds
        Seconds to wait between each attempt.
    """
    attempts = 0
    while attempts < max_attempts:
        try:
            # Yield control to the context block.
            yield
            # If block finished cleanly, we are done.
            return
        except RateLimitError as exc:
            logger.warning("Throttling due to rate limitations")
            attempts += 1
            if attempts >= max_attempts:
                # After the last attempt we re‑raise the exception.
                logger.error("Max attempts reached for rate limit handling")
                raise exc
            time.sleep(wait_seconds)
