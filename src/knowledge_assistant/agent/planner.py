from abc import ABC, abstractmethod

from knowledge_assistant.agent.models import (
    PlannerDecision,
)
from knowledge_assistant.agent.prompts import (
    build_planner_prompt,
)
from knowledge_assistant.agent.tools.specifications import (
    ToolSpecification,
)
from knowledge_assistant.agent.validation import (
    parse_planner_decision,
)
from knowledge_assistant.llm import LLMProvider

import logging

logger = logging.getLogger(__name__)


class AgentPlanner(ABC):
    """Choose the next action for an agent request."""

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the planner provider name."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the planner model name."""

    @abstractmethod
    def plan(
        self,
        query: str,
        specifications: tuple[
            ToolSpecification,
            ...
        ],
    ) -> PlannerDecision:
        """Return one validated planner decision."""


class LLMAgentPlanner(AgentPlanner):
    """Use an LLM to select a tool or final answer."""
    
    @property
    def provider_name(self) -> str:
        return self._llm_provider.provider_name


    @property
    def model_name(self) -> str:
        return self._llm_provider.model_name


    def __init__(
        self,
        llm_provider: LLMProvider,
    ) -> None:
        self._llm_provider = llm_provider

    def plan(
        self,
        query: str,
        specifications: tuple[
            ToolSpecification,
            ...
        ],
    ) -> PlannerDecision:
        normalized_query = query.strip()

        if not normalized_query:
            raise ValueError(
                "Agent query cannot be empty"
            )

        if not specifications:
            raise ValueError(
                "Planner requires at least one tool"
            )

        prompt = build_planner_prompt(
            query=normalized_query,
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