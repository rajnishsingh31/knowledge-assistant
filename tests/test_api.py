from pathlib import Path

from fastapi.testclient import TestClient

from knowledge_assistant.api.app import app
from knowledge_assistant.models import (
    Chunk,
    GeneratedAnswer,
    IndexStats,
    SearchResult,
)


def create_search_result() -> SearchResult:
    return SearchResult(
        chunk=Chunk(
            chunk_id="chunk-1",
            document_id="document-1",
            source_path=Path("cloud-security.docx"),
            content="Least privilege grants only required permissions.",
            start_line=1,
            end_line=2,
            document_hash="document-hash-1",
            chunk_hash="chunk-hash-1",
        ),
        retrieval_method="hybrid+reranked",
        score=4.2,
        reranker_score=4.2,
    )


class StubApplication:
    def stats(self) -> IndexStats:
        return IndexStats(
            table_name="test-table",
            chunk_count=1,
            document_count=1,
            embedding_models=("test-model",),
            dimensions=(384,),
        )

    def search(
        self,
        query: str,
        limit: int | None = None,
        strategy_name: str | None = None,
        retrieval_filter: object | None = None,
    ) -> list[SearchResult]:
        return [create_search_result()]

    def ask(
        self,
        query: str,
        limit: int | None = None,
        retrieval_filter: object | None = None,
    ) -> GeneratedAnswer:
        result = create_search_result()

        return GeneratedAnswer(
            content="Least privilege limits permissions.",
            provider_name="stub",
            model_name="stub-model",
            sources=(result,),
        )


def test_health() -> None:
    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"