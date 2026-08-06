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


class AnswerFromDocumentsTool(AgentTool):
    def __init__(
        self,
        application: KnowledgeAssistantApplication,
    ) -> None:
        self._application = application

    @property
    def specification(self) -> ToolSpecification:
        return ToolSpecification(
            name="answer_from_documents",
            description=(
                "Generate a grounded answer using indexed documents."
            ),
            parameters=(
                ToolParameter(
                    name="query",
                    description="Question to answer",
                    type_name="string",
                    required=True,
                ),
                ToolParameter(
                    name="limit",
                    description="Maximum source chunks",
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
                "answer_from_documents requires a non-empty query"
            )

        raw_limit = arguments.get("limit")
        limit = int(raw_limit) if raw_limit is not None else None

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
            tool_name=self.specification.name,
            content=json.dumps(payload, indent=2),
        )