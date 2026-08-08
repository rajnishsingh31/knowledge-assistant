from abc import ABC, abstractmethod
import logging

from knowledge_assistant.agent.models import (
    AgentContext,
    PlannerDecision,
)
from knowledge_assistant.llm.planner_prompts import (
    build_planner_prompt,
)
from knowledge_assistant.agent.tools.specifications import (
    ToolSpecification,
)
from knowledge_assistant.agent.validation import (
    parse_planner_decision,
)
from knowledge_assistant.llm import LLMProvider


logger = logging.getLogger(__name__)


class AgentPlanner(ABC):
    @property
    @abstractmethod
    def provider_name(self) -> str:
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        ...

    @abstractmethod
    def plan(
        self,
        context: AgentContext,
        specifications: tuple[ToolSpecification, ...],
    ) -> PlannerDecision:
        """Choose the next action."""


class LLMAgentPlanner(AgentPlanner):
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

    def plan(
        self,
        context: AgentContext,
        specifications: tuple[ToolSpecification, ...],
    ) -> PlannerDecision:
        prompt = build_planner_prompt(
            context=context,
            specifications=specifications,
        )

        content = self._llm_provider.generate(prompt)

        logger.debug(
            "planner_raw_response=%r",
            content,
        )

        return parse_planner_decision(
            content=content,
            specifications=specifications,
        )