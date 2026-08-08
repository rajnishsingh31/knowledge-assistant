from abc import ABC, abstractmethod

from knowledge_assistant.agent.models import (
    AgentToolCall,
    AgentToolResult,
    AgentStep,
)
from knowledge_assistant.llm.synthesis_prompts import (
    build_synthesis_prompt,
)
from knowledge_assistant.llm import LLMProvider
from knowledge_assistant.agent.citations import (
    append_citations_if_missing,
    deduplicate_citations,
)


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
        steps: tuple[AgentStep, ...],
    ) -> str:
         """Produce a final answer from accumulated observations."""


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
        steps: tuple[AgentStep, ...],
    ) -> str:
        if not steps:
            raise ValueError(
                "At least one agent step is required for synthesis"
            )

        citations = deduplicate_citations(
            tuple(
                citation
                for step in steps
                for citation
                in step.tool_result.observation.citations
            )
        )

        prompt = build_synthesis_prompt(
            query=query,
            steps=steps,
        )
      

        content = self._llm_provider.generate(prompt)
        normalized_content = content.strip()

        if not normalized_content:
            raise RuntimeError(
                "Agent synthesizer returned an empty response"
            )

        return append_citations_if_missing(
            content=normalized_content,
            citations=citations,
        )