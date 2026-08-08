import json
from typing import Any

from knowledge_assistant.agent.citations import (
    deduplicate_citations,
)
from knowledge_assistant.agent.models import (
    AgentCitation,
    AgentObservation,
    AgentToolResult,
)
from knowledge_assistant.agent.tools.base import AgentTool
from knowledge_assistant.agent.tools.specifications import (
    ToolParameter,
    ToolSpecification,
)
from knowledge_assistant.application import (
    KnowledgeAssistantApplication,
)


class AnswerFromDocumentsTool(AgentTool):
    """Retrieve high-quality evidence for answering a question."""

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
                "Retrieve and rerank the strongest document evidence "
                "for answering a direct knowledge question. This tool "
                "returns evidence, not a generated answer."
            ),
            parameters=(
                ToolParameter(
                    name="query",
                    description="Question requiring document evidence",
                    type_name="string",
                    required=True,
                ),
                ToolParameter(
                    name="limit",
                    description="Maximum evidence chunks",
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
        limit = (
            int(raw_limit)
            if raw_limit is not None
            else None
        )

        if limit is not None and limit <= 0:
            raise ValueError(
                "answer_from_documents limit must be greater than zero"
            )

        context = self._application.retrieve_answer_context(
            query=query,
            limit=limit,
        )

        evidence = [
            {
                "rank": rank,
                "source_name": result.chunk.source_path.name,
                "start_line": result.chunk.start_line,
                "end_line": result.chunk.end_line,
                "content": result.chunk.content,
                "retrieval_method": result.retrieval_method,
                "score": result.score,
                "reranker_score": result.reranker_score,
            }
            for rank, result in enumerate(
                context.results,
                start=1,
            )
        ]

        citations = deduplicate_citations(
            tuple(
                AgentCitation(
                    source_name=result.chunk.source_path.name,
                    start_line=result.chunk.start_line,
                    end_line=result.chunk.end_line,
                )
                for result in context.results
            )
        )

        return AgentToolResult(
            tool_name=self.specification.name,
            observation=AgentObservation(
                content=json.dumps(
                    evidence,
                    indent=2,
                ),
                citations=citations,
                metadata={
                    "query": query,
                    "evidence_count": len(evidence),
                    "evidence_type": "reranked_document_chunks",
                },
            ),
        )