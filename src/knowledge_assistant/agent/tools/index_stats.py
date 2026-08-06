import json
from typing import Any

from knowledge_assistant.agent.models import AgentToolResult
from knowledge_assistant.agent.tools.base import AgentTool
from knowledge_assistant.agent.tools.specifications import (
    ToolSpecification,
)
from knowledge_assistant.application import (
    KnowledgeAssistantApplication,
)


class GetIndexStatsTool(AgentTool):
    def __init__(
        self,
        application: KnowledgeAssistantApplication,
    ) -> None:
        self._application = application

    @property
    def specification(self) -> ToolSpecification:
        return ToolSpecification(
            name="get_index_stats",
            description=(
                "Return indexed document, chunk, model, and dimension "
                "statistics."
            ),
            parameters=(),
        )

    def execute(
        self,
        arguments: dict[str, Any],
    ) -> AgentToolResult:
        stats = self._application.stats()

        payload = {
            "table_name": stats.table_name,
            "document_count": stats.document_count,
            "chunk_count": stats.chunk_count,
            "embedding_models": list(stats.embedding_models),
            "dimensions": list(stats.dimensions),
        }

        return AgentToolResult(
            tool_name=self.specification.name,
            content=json.dumps(payload, indent=2),
        )