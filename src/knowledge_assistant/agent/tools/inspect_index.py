import json
from typing import Any

from knowledge_assistant.agent.models import AgentToolResult
from knowledge_assistant.agent.tools.base import AgentTool
from knowledge_assistant.agent.tools.specifications import (
    ToolParameter,
    ToolSpecification,
)
from knowledge_assistant.application import (
    KnowledgeAssistantApplication,
)


class InspectIndexTool(AgentTool):
    def __init__(
        self,
        application: KnowledgeAssistantApplication,
    ) -> None:
        self._application = application

    @property
    def specification(self) -> ToolSpecification:
        return ToolSpecification(
            name="inspect_index",
            description=(
                "Inspect raw indexed chunk records for diagnostics."
            ),
            parameters=(
                ToolParameter(
                    name="limit",
                    description="Maximum records",
                    type_name="integer",
                    required=False,
                ),
            ),
        )

    def execute(
        self,
        arguments: dict[str, Any],
    ) -> AgentToolResult:
        limit = int(arguments.get("limit", 5))

        if limit <= 0:
            raise ValueError(
                "inspect_index limit must be greater than zero"
            )

        records = self._application.inspect(limit=limit)

        return AgentToolResult(
            tool_name=self.specification.name,
            content=json.dumps(
                records,
                indent=2,
                default=str,
            ),
        )