from pathlib import Path

from knowledge_assistant.agent.tools.answer_documents import (
    AnswerFromDocumentsTool,
)
from knowledge_assistant.models import (
    Chunk,
    RetrievedContext,
    SearchResult,
)


def create_search_result() -> SearchResult:
    return SearchResult(
        chunk=Chunk(
            chunk_id="chunk-1",
            document_id="document-1",
            source_path=Path("cloud-security.docx"),
            content=(
                "Least privilege grants only the permissions "
                "required for assigned tasks."
            ),
            start_line=7,
            end_line=14,
            document_hash="document-hash-1",
            chunk_hash="chunk-hash-1",
        ),
        retrieval_method="hybrid+reranked",
        score=4.2,
        reranker_score=4.2,
    )


class StubApplication:
    def retrieve_answer_context(
        self,
        query: str,
        limit: int | None = None,
        retrieval_filter: object | None = None,
    ) -> RetrievedContext:
        assert query == "What is least privilege?"
        assert limit == 3

        return RetrievedContext(
            query=query,
            results=(create_search_result(),),
        )

def test_answer_tool_returns_evidence_without_generation() -> None:
    tool = AnswerFromDocumentsTool(
        application=StubApplication(),  # type: ignore[arg-type]
    )

    result = tool.execute(
        {
            "query": "What is least privilege?",
            "limit": 3,
        }
    )

    observation = result.observation

    assert result.tool_name == "answer_from_documents"
    assert (
        observation.metadata["evidence_type"]
        == "reranked_document_chunks"
    )
    assert observation.metadata["evidence_count"] == 1
    assert len(observation.citations) == 1
    assert (
        observation.citations[0].source_name
        == "cloud-security.docx"
    )
    assert (
        "Least privilege grants only the permissions"
        in observation.content
    )