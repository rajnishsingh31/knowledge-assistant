from abc import ABC, abstractmethod

from ollama import Client

from knowledge_assistant.models import Prompt


DEFAULT_OLLAMA_HOST = "http://localhost:11434"
DEFAULT_OLLAMA_MODEL = "qwen3:1.7b"


class LLMProvider(ABC):
    """Provider-neutral contract for text generation."""

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the provider identifier."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the configured model identifier."""

    @abstractmethod
    def generate(
        self,
        prompt: Prompt,
    ) -> str:
        """Generate text from system and user prompts."""


class OllamaProvider(LLMProvider):
    """Generate answers using a locally running Ollama server."""

    def __init__(
        self,
        model_name: str = DEFAULT_OLLAMA_MODEL,
        host: str = DEFAULT_OLLAMA_HOST,
    ) -> None:
        self._model_name = model_name
        self._client = Client(host=host)

    @property
    def provider_name(self) -> str:
        return "ollama"

    @property
    def model_name(self) -> str:
        return self._model_name

    def generate(
        self,
        prompt: Prompt,
    ) -> str:
        response = self._client.chat(
            model=self.model_name,
            messages=[
                {
                    "role": "system",
                    "content": prompt.system,
                },
                {
                    "role": "user",
                    "content": prompt.user,
                },
            ],
            options={
                "temperature": 0,
            },
        )

        content = response.message.content.strip()

        if not content:
            raise RuntimeError("The Ollama model returned an empty response")

        return content