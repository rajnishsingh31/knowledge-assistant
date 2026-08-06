from typing import Any

from knowledge_assistant.agent.models import (
    AgentToolName,
    AgentToolResult,
)
from knowledge_assistant.agent.tools.base import AgentTool
from knowledge_assistant.agent.tools.specifications import (
    ToolSpecification,
)


class AgentToolRegistry:
    def __init__(
        self,
        tools: list[AgentTool],
    ) -> None:
        if not tools:
            raise ValueError(
                "At least one agent tool is required"
            )

        self._tools = {
            tool.specification.name: tool
            for tool in tools
        }

    @property
    def specifications(
        self,
    ) -> tuple[ToolSpecification, ...]:
        return tuple(
            tool.specification
            for tool in self._tools.values()
        )

    def execute(
        self,
        tool_name: AgentToolName,
        arguments: dict[str, Any],
    ) -> AgentToolResult:
        tool = self._tools.get(tool_name)

        if tool is None:
            raise ValueError(
                f"Unknown agent tool: {tool_name}"
            )

        return tool.execute(arguments)