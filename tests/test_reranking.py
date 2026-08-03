from pathlib import Path

import pytest

from knowledge_assistant.models import Chunk, SearchResult
from knowledge_assistant.reranking import IdentityReranker


def create_result(
    chunk_id: str,
    score: float,
) -> SearchResult:
    return SearchResult(
        chunk=Chunk(
            chunk_id=chunk_id,
            document_id="document-1",
            source_path=Path(f"{chunk_id}.md"),
            content=f"Content for {chunk_id}",
            start_line=1,
            end_line=2,
            document_hash="document-hash-1",
            chunk_hash=f"chunk-hash-{chunk_id}",
        ),
        retrieval_method="hybrid",
        score=score,
    )


def test_identity_reranker_preserves_order() -> None:
    results = [
        create_result("first", 0.9),
        create_result("second", 0.8),
        create_result("third", 0.7),
    ]

    reranked = IdentityReranker().rerank(
        query="test query",
        results=results,
        limit=2,
    )

    assert [result.chunk.chunk_id for result in reranked] == [
        "first",
        "second",
    ]


def test_identity_reranker_rejects_invalid_limit() -> None:
    with pytest.raises(
        ValueError,
        match="Limit must be greater than zero",
    ):
        IdentityReranker().rerank(
            query="test",
            results=[],
            limit=0,
        )