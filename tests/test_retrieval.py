from pathlib import Path

from knowledge_assistant.models import Chunk, SearchResult
from knowledge_assistant.retrieval import (
    HybridRetrievalStrategy,
    RetrievalStrategy,
)


def create_result(
    chunk_id: str,
    method: str,
) -> SearchResult:
    return SearchResult(
        chunk=Chunk(
            chunk_id=chunk_id,
            document_id=f"document-{chunk_id}",
            source_path=Path(f"{chunk_id}.md"),
            content=f"Content for {chunk_id}",
            start_line=1,
            end_line=2,
            document_hash="document-hash-1",
            chunk_hash=f"chunk-hash-{chunk_id}",
        ),
        retrieval_method=method,
        score=1.0,
        vector_distance=0.5 if method == "vector" else None,
        bm25_score=2.0 if method == "bm25" else None,
    )


class StubRetrievalStrategy(RetrievalStrategy):
    def __init__(self, results: list[SearchResult]) -> None:
        self._results = results

    def search(
        self,
        query: str,
        limit: int,
    ) -> list[SearchResult]:
        return self._results[:limit]


def test_hybrid_strategy_rewards_chunks_found_by_both_methods() -> None:
    vector_strategy = StubRetrievalStrategy(
        [
            create_result("shared", "vector"),
            create_result("vector-only", "vector"),
        ]
    )

    bm25_strategy = StubRetrievalStrategy(
        [
            create_result("bm25-only", "bm25"),
            create_result("shared", "bm25"),
        ]
    )

    hybrid = HybridRetrievalStrategy(
        vector_strategy=vector_strategy,
        bm25_strategy=bm25_strategy,
        candidate_limit=10,
        rrf_k=60,
    )

    results = hybrid.search(
        query="test query",
        limit=3,
    )

    assert results[0].chunk.chunk_id == "shared"
    assert results[0].retrieval_method == "hybrid"
    assert results[0].vector_distance is not None
    assert results[0].bm25_score is not None