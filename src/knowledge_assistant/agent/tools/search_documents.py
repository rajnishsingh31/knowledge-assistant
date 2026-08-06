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


class SearchDocumentsTool(AgentTool):
    def __init__(
        self,
        application: KnowledgeAssistantApplication,
    ) -> None:
        self._application = application

    @property
    def specification(self) -> ToolSpecification:
        return ToolSpecification(
            name="search_documents",
            description=(
                "Search indexed documents for relevant evidence."
            ),
            parameters=(
                ToolParameter(
                    name="query",
                    description="Search query",
                    type_name="string",
                    required=True,
                ),
                ToolParameter(
                    name="limit",
                    description="Maximum results",
                    type_name="integer",
                    required=False,
                ),
            ),
        )

    def execute(
        self,
        arguments: dict[str, Any],
    ) -> AgentToolResult:
        query = str(arguments.get("query", "")).strip()

        if not query:
            raise ValueError(
                "search_documents requires a non-empty query"
            )

        limit = int(arguments.get("limit", 5))

        if limit <= 0:
            raise ValueError(
                "search_documents limit must be greater than zero"
            )

        results = self._application.search(
            query=query,
            limit=limit,
        )

        payload = [
            {
                "rank": index,
                "source": result.chunk.source_path.name,
                "start_line": result.chunk.start_line,
                "end_line": result.chunk.end_line,
                "content": result.chunk.content,
                "retrieval_method": result.retrieval_method,
                "score": result.score,
            }
            for index, result in enumerate(
                results,
                start=1,
            )
        ]

        return AgentToolResult(
            tool_name=self.specification.name,
            content=json.dumps(payload, indent=2),
        )