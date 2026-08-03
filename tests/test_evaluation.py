from pathlib import Path

from knowledge_assistant.evaluation import RetrievalEvaluator
from knowledge_assistant.models import (
    Chunk,
    EvaluationCase,
    RetrievalFilter,
    SearchResult,
)
from knowledge_assistant.retrieval import RetrievalStrategy, Retriever
import pytest


def create_result(filename: str) -> SearchResult:
    return SearchResult(
        chunk=Chunk(
            chunk_id=f"chunk-{filename}",
            document_id=f"document-{filename}",
            source_path=Path(filename),
            content="content",
            start_line=1,
            end_line=1,
            document_hash="document-hash-1",
            chunk_hash=f"chunk-hash-1",
        ),
        retrieval_method="vector",
        score=1.0,
    )


class QueryBasedStrategy(RetrievalStrategy):
    def search(
        self,
        query: str,
        limit: int,
        retrieval_filter: RetrievalFilter | None = None,
    ) -> list[SearchResult]:
        results_by_query = {
            "first": [
                create_result("expected.md"),
                create_result("other.md"),
            ],
            "second": [
                create_result("other.md"),
                create_result("expected.md"),
            ],
            "third": [
                create_result("other.md"),
            ],
        }

        return results_by_query[query][:limit]


def test_evaluator_calculates_top_1_and_top_k_accuracy() -> None:
    cases = [
        EvaluationCase(
            case_id="case-1",
            query="first",
            expected_documents=("expected.md",),
        ),
        EvaluationCase(
            case_id="case-2",
            query="second",
            expected_documents=("expected.md",),
        ),
        EvaluationCase(
            case_id="case-3",
            query="third",
            expected_documents=("expected.md",),
        ),
    ]

    evaluator = RetrievalEvaluator(
        retriever=Retriever(
            strategy=QueryBasedStrategy(),
        ),
        strategy_name="test",
    )

    summary = evaluator.evaluate(
        cases=cases,
        top_k=2,
    )

    assert summary.case_count == 3
    assert summary.top_1_hits == 1
    assert summary.top_k_hits == 2
    assert summary.top_1_accuracy == pytest.approx(1 / 3)
    assert summary.top_k_accuracy == pytest.approx(2 / 3)