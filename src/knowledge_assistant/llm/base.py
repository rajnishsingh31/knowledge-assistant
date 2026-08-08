from abc import ABC, abstractmethod
from knowledge_assistant.llm.models import Prompt

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