from abc import ABC, abstractmethod

from sentence_transformers import CrossEncoder

from knowledge_assistant.models import SearchResult


DEFAULT_RERANKER_MODEL = (
    "cross-encoder/ms-marco-MiniLM-L6-v2"
)


class Reranker(ABC):
    """Rank retrieved search results by relevance."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the configured reranker model."""

    @abstractmethod
    def rerank(
        self,
        query: str,
        results: list[SearchResult],
        limit: int,
    ) -> list[SearchResult]:
        """Return the highest-ranked results."""


class IdentityReranker(Reranker):
    """Return retrieved results without changing their order."""

    @property
    def model_name(self) -> str:
        return "identity"

    def rerank(
        self,
        query: str,
        results: list[SearchResult],
        limit: int,
    ) -> list[SearchResult]:
        if limit <= 0:
            raise ValueError("Limit must be greater than zero")

        return results[:limit]


class CrossEncoderReranker(Reranker):
    """Rerank candidate chunks using a local cross-encoder."""

    def __init__(
        self,
        model_name: str = DEFAULT_RERANKER_MODEL,
    ) -> None:
        self._model_name = model_name
        self._model = CrossEncoder(model_name)

    @property
    def model_name(self) -> str:
        return self._model_name

    def rerank(
        self,
        query: str,
        results: list[SearchResult],
        limit: int,
    ) -> list[SearchResult]:
        normalized_query = query.strip()

        if not normalized_query:
            raise ValueError("Query cannot be empty")

        if limit <= 0:
            raise ValueError("Limit must be greater than zero")

        if not results:
            return []

        sentence_pairs = [
            (normalized_query, result.chunk.content)
            for result in results
        ]

        predicted_scores = self._model.predict(
            sentence_pairs,
            show_progress_bar=False,
        )

        reranked_results = [
            SearchResult(
                chunk=result.chunk,
                retrieval_method=(
                    f"{result.retrieval_method}+reranked"
                ),
                score=float(reranker_score),
                vector_distance=result.vector_distance,
                bm25_score=result.bm25_score,
                reranker_score=float(reranker_score),
            )
            for result, reranker_score in zip(
                results,
                predicted_scores,
                strict=True,
            )
        ]

        reranked_results.sort(
            key=lambda result: result.score,
            reverse=True,
        )

        return reranked_results[:limit]