from abc import ABC, abstractmethod

from knowledge_assistant.agent.models import (
    AgentToolCall,
    AgentToolResult,
)
from knowledge_assistant.agent.synthesis_prompts import (
    build_synthesis_prompt,
)
from knowledge_assistant.llm import LLMProvider


class AgentResponseSynthesizer(ABC):
    """Convert a tool observation into a final answer."""

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the synthesizer provider name."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the synthesizer model name."""

    @abstractmethod
    def synthesize(
        self,
        query: str,
        tool_call: AgentToolCall,
        tool_result: AgentToolResult,
    ) -> str:
        """Produce the final user-facing answer."""


class LLMAgentResponseSynthesizer(
    AgentResponseSynthesizer
):
    """Use an LLM to synthesize a tool observation."""

    def __init__(
        self,
        llm_provider: LLMProvider,
    ) -> None:
        self._llm_provider = llm_provider

    @property
    def provider_name(self) -> str:
        return self._llm_provider.provider_name

    @property
    def model_name(self) -> str:
        return self._llm_provider.model_name

    def synthesize(
        self,
        query: str,
        tool_call: AgentToolCall,
        tool_result: AgentToolResult,
    ) -> str:
        prompt = build_synthesis_prompt(
            query=query,
            tool_call=tool_call,
            tool_result=tool_result,
        )

        content = self._llm_provider.generate(prompt)
        normalized_content = content.strip()

        if not normalized_content:
            raise RuntimeError(
                "Agent synthesizer returned an empty response"
            )

        return normalized_content