import logging
import os
import threading
import time
from typing import Optional

from langchain_openai import ChatOpenAI
from openai import APIConnectionError, APIError, RateLimitError


logger = logging.getLogger(__name__)


class LlmProvider:

    def create_chat_model(self, model_name, temperature=0, timeout=None, max_tokens=None):
        raise NotImplementedError("Child class must create the chat model")


class OpenAiProvider(LlmProvider):

    def create_chat_model(self, model_name, temperature=0, timeout=None, max_tokens=None) -> ChatOpenAI:
        return ChatOpenAI(
            model=model_name,
            api_key=os.getenv("OPENAI_API_KEY"),
            temperature=temperature,
            timeout=timeout,
            max_tokens=max_tokens,
        )


class LlmProviderFactory:

    def __init__(self):
        self.providers = {
            "openai": OpenAiProvider,
        }

    def get_provider_name(self) -> str:
        return (os.getenv("LLM_PROVIDER") or "openai").strip().lower()

    def create_provider(self, provider_name: Optional[str] = None) -> LlmProvider:
        provider_name = provider_name or self.get_provider_name()

        if provider_name not in self.providers:
            raise ValueError(f"Unknown LLM provider: {provider_name}")

        provider_class = self.providers[provider_name]
        return provider_class()


class ModelChain:

    def __init__(self, env_name: str, default_models: str):
        self.env_name = env_name
        self.default_models = default_models

    def get_model_names(self) -> list[str]:
        raw_models = os.getenv(self.env_name, self.default_models)
        model_names = []

        for model_name in raw_models.split(","):
            model_name = model_name.strip()

            if model_name and model_name not in model_names:
                model_names.append(model_name)

        return model_names


class RetryChecker:

    def __init__(self):
        self.retry_words = [
            "429", "toomanyrequests", "ratelimit", "overloaded",
            "serviceunavailable", "internalservererror", "timeout",
            "timedout", "connection",
        ]

    def is_retryable(self, error: BaseException) -> bool:
        if isinstance(error, (RateLimitError, APIConnectionError)):
            return True

        if isinstance(error, APIError):
            status_code = getattr(error, "status_code", None)

            if status_code == 429 or (isinstance(status_code, int) and status_code >= 500):
                return True

        message = str(error).lower().replace(" ", "").replace("_", "").replace("-", "")

        for word in self.retry_words:
            if word in message:
                return True

        return False


class LlmClient:

    def __init__(
        self,
        provider: Optional[LlmProvider] = None,
        model_env_name: str = "LLM_MODEL_CHAIN",
        default_models: str = "gpt-4o-mini,gpt-4o",
        retries_per_model: int = 2,
        backoff_seconds: float = 2.0,
        timeout: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ):
        if retries_per_model < 1:
            raise ValueError("retries_per_model must be at least 1")

        if backoff_seconds < 0:
            raise ValueError("backoff_seconds must be non negative")

        self.provider = provider or LlmProviderFactory().create_provider()
        self.model_names = ModelChain(model_env_name, default_models).get_model_names()
        self.retry_checker = RetryChecker()
        self.retries_per_model = retries_per_model
        self.backoff_seconds = backoff_seconds
        self.timeout = timeout
        self.max_tokens = max_tokens

        if not self.model_names:
            raise RuntimeError(f"No models configured in {model_env_name}")

    def run(self, llm_model_name: Optional[str], messages: list, output_model):
        last_error = None
        model_names = [llm_model_name] if llm_model_name else self.model_names

        for model_index, model_name in enumerate(model_names):
            chat_model = self.provider.create_chat_model(
                model_name,
                timeout=self.timeout,
                max_tokens=self.max_tokens,
            )
            structured_model = chat_model.with_structured_output(output_model)

            for attempt in range(1, self.retries_per_model + 1):
                try:
                    return structured_model.invoke(messages)
                except Exception as error:
                    last_error = error

                    if not self.retry_checker.is_retryable(error):
                        raise

                    if attempt < self.retries_per_model:
                        wait_seconds = self.backoff_seconds * (2 ** (attempt - 1))
                        logger.warning("LLM failed on %s (attempt %d/%d), retrying in %.1fs: %s", model_name, attempt, self.retries_per_model, wait_seconds, error)
                        time.sleep(wait_seconds)

            if model_index < len(model_names) - 1:
                logger.warning("Used all retries on %s, trying next model", model_name)

        raise last_error


class VisionLimiter:

    def __init__(self, max_calls: Optional[int] = None):
        if max_calls is None:
            max_calls = self.read_max_calls_from_env()

        self.semaphore = threading.Semaphore(max(1, max_calls))

    def read_max_calls_from_env(self) -> int:
        try:
            return int(os.getenv("LLM_VISION_MAX_CONCURRENT", "3"))
        except ValueError:
            return 3

    def acquire(self):
        self.semaphore.acquire()

    def release(self):
        self.semaphore.release()


def create_openai_chat_model(model_name, temperature=0, timeout=None, max_tokens=None) -> ChatOpenAI:
    return OpenAiProvider().create_chat_model(model_name, temperature, timeout, max_tokens)


def ask_openai(prompt: str, model_name: str = "gpt-4o-mini") -> str:
    chat_model = create_openai_chat_model(model_name)
    return chat_model.invoke(prompt).content
