from typing import Any

from knowledge_assistant.agent.models import AgentToolResult
from knowledge_assistant.agent.registry import AgentToolRegistry
from knowledge_assistant.agent.tools.base import AgentTool
from knowledge_assistant.agent.tools.specifications import (
    ToolSpecification,
)


class StubTool(AgentTool):
    @property
    def specification(self) -> ToolSpecification:
        return ToolSpecification(
            name="get_index_stats",
            description="Test tool",
            parameters=(),
        )

    def execute(
        self,
        arguments: dict[str, Any],
    ) -> AgentToolResult:
        return AgentToolResult(
            tool_name=self.specification.name,
            content="done",
        )


def test_registry_exposes_specifications() -> None:
    registry = AgentToolRegistry(
        tools=[StubTool()]
    )

    assert len(registry.specifications) == 1
    assert (
        registry.specifications[0].name
        == "get_index_stats"
    )


def test_registry_executes_registered_tool() -> None:
    registry = AgentToolRegistry(
        tools=[StubTool()]
    )

    result = registry.execute(
        tool_name="get_index_stats",
        arguments={},
    )

    assert result.content == "done"