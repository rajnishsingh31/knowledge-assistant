from abc import ABC, abstractmethod
import json
from typing import Any

from knowledge_assistant.application import (
    KnowledgeAssistantApplication,
)
from knowledge_assistant.models import (
    AgentToolName,
    AgentToolResult,
)


class AgentTool(ABC):
    """A controlled capability available to the agent."""

    @property
    @abstractmethod
    def name(self) -> AgentToolName:
        """Return the tool name."""

    @property
    @abstractmethod
    def description(self) -> str:
        """Describe when the tool should be used."""

    @abstractmethod
    def execute(
        self,
        arguments: dict[str, Any],
    ) -> AgentToolResult:
        """Execute the tool with validated arguments."""

    class SearchDocumentsTool(AgentTool):
    """Search indexed document chunks."""

    def __init__(
        self,
        application: KnowledgeAssistantApplication,
    ) -> None:
        self._application = application

    @property
    def name(self) -> AgentToolName:
        return "search_documents"

    @property
    def description(self) -> str:
        return (
            "Search indexed documents for relevant evidence. "
            "Use this when the user asks about information that may "
            "exist in the indexed document collection."
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

        limit_value = arguments.get("limit", 5)
        limit = int(limit_value)

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
            tool_name=self.name,
            content=json.dumps(
                payload,
                indent=2,
            ),
        )

class SearchDocumentsTool(AgentTool):
    """Search indexed document chunks."""

    def __init__(
        self,
        application: KnowledgeAssistantApplication,
    ) -> None:
        self._application = application

    @property
    def name(self) -> AgentToolName:
        return "search_documents"

    @property
    def description(self) -> str:
        return (
            "Search indexed documents for relevant evidence. "
            "Use this when the user asks about information that may "
            "exist in the indexed document collection."
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

        limit_value = arguments.get("limit", 5)
        limit = int(limit_value)

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
            tool_name=self.name,
            content=json.dumps(
                payload,
                indent=2,
            ),
        )

class GetIndexStatsTool(AgentTool):
    """Return index statistics."""

    def __init__(
        self,
        application: KnowledgeAssistantApplication,
    ) -> None:
        self._application = application

    @property
    def name(self) -> AgentToolName:
        return "get_index_stats"

    @property
    def description(self) -> str:
        return (
            "Return index statistics such as document count, "
            "chunk count, embedding models, and dimensions."
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
            "embedding_models": list(
                stats.embedding_models
            ),
            "dimensions": list(stats.dimensions),
        }

        return AgentToolResult(
            tool_name=self.name,
            content=json.dumps(
                payload,
                indent=2,
            ),
        )

class AnswerFromDocumentsTool(AgentTool):
    """Generate a grounded answer from indexed documents."""

    def __init__(
        self,
        application: KnowledgeAssistantApplication,
    ) -> None:
        self._application = application

    @property
    def name(self) -> AgentToolName:
        return "answer_from_documents"

    @property
    def description(self) -> str:
        return (
            "Generate a final grounded answer using indexed documents. "
            "Use this when the user asks a direct knowledge question."
        )

    def execute(
        self,
        arguments: dict[str, Any],
    ) -> AgentToolResult:
        query = str(arguments.get("query", "")).strip()

        if not query:
            raise ValueError(
                "answer_from_documents requires a non-empty query"
            )

        limit_value = arguments.get("limit")
        limit = (
            int(limit_value)
            if limit_value is not None
            else None
        )

        answer = self._application.ask(
            query=query,
            limit=limit,
        )

        payload = {
            "answer": answer.content,
            "provider_name": answer.provider_name,
            "model_name": answer.model_name,
            "sources": [
                {
                    "source": source.chunk.source_path.name,
                    "start_line": source.chunk.start_line,
                    "end_line": source.chunk.end_line,
                }
                for source in answer.sources
            ],
        }

        return AgentToolResult(
            tool_name=self.name,
            content=json.dumps(
                payload,
                indent=2,
            ),
        )

class AgentToolRegistry:
    """Look up and execute registered agent tools."""

    def __init__(
        self,
        tools: list[AgentTool],
    ) -> None:
        if not tools:
            raise ValueError(
                "At least one agent tool is required"
            )

        self._tools = {
            tool.name: tool
            for tool in tools
        }

    @property
    def tools(self) -> tuple[AgentTool, ...]:
        return tuple(self._tools.values())

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