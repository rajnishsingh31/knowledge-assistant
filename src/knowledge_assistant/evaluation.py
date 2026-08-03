import json
from pathlib import Path
from typing import Any

from knowledge_assistant.models import (
    EvaluationCase,
    EvaluationCaseResult,
    RetrievalEvaluationSummary,
)
from knowledge_assistant.retrieval import Retriever
from knowledge_assistant.reranking import Reranker


def load_evaluation_cases(
    dataset_path: Path,
) -> list[EvaluationCase]:
    """Load retrieval evaluation cases from JSON."""

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Evaluation dataset not found: {dataset_path}"
        )

    if not dataset_path.is_file():
        raise ValueError(
            f"Evaluation dataset path is not a file: {dataset_path}"
        )

    raw_data: Any = json.loads(
        dataset_path.read_text(encoding="utf-8")
    )

    if not isinstance(raw_data, list):
        raise ValueError(
            "Evaluation dataset must contain a JSON array"
        )

    cases: list[EvaluationCase] = []

    for item in raw_data:
        if not isinstance(item, dict):
            raise ValueError(
                "Each evaluation case must be a JSON object"
            )

        case_id = str(item["case_id"]).strip()
        query = str(item["query"]).strip()

        expected_documents_value = item["expected_documents"]

        if not isinstance(expected_documents_value, list):
            raise ValueError(
                "'expected_documents' must be a JSON array"
            )

        expected_documents = tuple(
            str(document).strip()
            for document in expected_documents_value
            if str(document).strip()
        )

        if not case_id:
            raise ValueError("Evaluation case ID cannot be empty")

        if not query:
            raise ValueError(
                f"Query cannot be empty for case: {case_id}"
            )

        if not expected_documents:
            raise ValueError(
                f"Expected documents cannot be empty: {case_id}"
            )

        cases.append(
            EvaluationCase(
                case_id=case_id,
                query=query,
                expected_documents=expected_documents,
            )
        )

    if not cases:
        raise ValueError("Evaluation dataset contains no cases")

    return cases


class RetrievalEvaluator:
    """Evaluate one retrieval strategy against known cases."""

    def __init__(
        self,
        retriever: Retriever,
        strategy_name: str,
        reranker: Reranker | None = None,
        candidate_limit: int | None = None,
    ) -> None:
        self._retriever = retriever
        self._strategy_name = strategy_name
        self._reranker = reranker
        self._candidate_limit = candidate_limit


    def evaluate(
        self,
        cases: list[EvaluationCase],
        top_k: int,
    ) -> RetrievalEvaluationSummary:
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")

        case_results: list[EvaluationCaseResult] = []

        retrieval_limit = (
            self._candidate_limit
            if self._reranker and self._candidate_limit is not None
            else top_k
        )

        for case in cases:

            search_results = self._retriever.search(
                query=case.query,
                limit=retrieval_limit,
            )

            if self._reranker is not None:
                search_results = self._reranker.rerank(
                    query=case.query,
                    results=search_results,
                    limit=top_k,
            )

            retrieved_documents = tuple(
                dict.fromkeys(
                    result.chunk.source_path.name
                    for result in search_results
                )
            )

            top_1_hit = bool(
                retrieved_documents
                and retrieved_documents[0]
                in case.expected_documents
            )

            top_k_hit = any(
                document in case.expected_documents
                for document in retrieved_documents
            )

            case_results.append(
                EvaluationCaseResult(
                    case_id=case.case_id,
                    query=case.query,
                    expected_documents=case.expected_documents,
                    retrieved_documents=retrieved_documents,
                    top_1_hit=top_1_hit,
                    top_k_hit=top_k_hit,
                )
            )

        top_1_hits = sum(
            result.top_1_hit
            for result in case_results
        )

        top_k_hits = sum(
            result.top_k_hit
            for result in case_results
        )

        case_count = len(case_results)

        return RetrievalEvaluationSummary(
            strategy_name=self._strategy_name,
            case_count=case_count,
            top_1_hits=top_1_hits,
            top_k_hits=top_k_hits,
            top_1_accuracy=top_1_hits / case_count,
            top_k_accuracy=top_k_hits / case_count,
            results=tuple(case_results),
        )