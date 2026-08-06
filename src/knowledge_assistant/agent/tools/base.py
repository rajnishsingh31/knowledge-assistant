from abc import ABC, abstractmethod
from typing import Any

from knowledge_assistant.agent.models import AgentToolResult
from knowledge_assistant.agent.tools.specifications import (
    ToolSpecification,
)


class AgentTool(ABC):
    @property
    @abstractmethod
    def specification(self) -> ToolSpecification:
        """Describe the tool and its accepted parameters."""

    @abstractmethod
    def execute(
        self,
        arguments: dict[str, Any],
    ) -> AgentToolResult:
        """Execute the tool."""