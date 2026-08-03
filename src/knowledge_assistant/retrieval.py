import logging
from time import perf_counter
from abc import ABC, abstractmethod
from dataclasses import dataclass

from knowledge_assistant.embeddings import EmbeddingProvider
from knowledge_assistant.models import SearchResult
from knowledge_assistant.vector_store import LanceDBVectorStore
from knowledge_assistant.models import Chunk

logger = logging.getLogger(__name__)

class RetrievalStrategy(ABC):
    """Contract for retrieving relevant chunks."""

    @abstractmethod
    def search(
        self,
        query: str,
        limit: int,
    ) -> list[SearchResult]:
        """Retrieve relevant chunks."""

class VectorRetrievalStrategy(RetrievalStrategy):
    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        vector_store: LanceDBVectorStore,
    ) -> None:
        self._embedding_provider = embedding_provider
        self._vector_store = vector_store

    def search(
        self,
        query: str,
        limit: int,
    ) -> list[SearchResult]:
        query_vector = self._embedding_provider.embed_query(query)

        return self._vector_store.search_vector(
            query_vector=query_vector,
            limit=limit,
        )

class BM25RetrievalStrategy(RetrievalStrategy):
    def __init__(
        self,
        vector_store: LanceDBVectorStore,
    ) -> None:
        self._vector_store = vector_store

    def search(
        self,
        query: str,
        limit: int,
    ) -> list[SearchResult]:
        return self._vector_store.search_text(
            query=query,
            limit=limit,
        )

@dataclass
class _FusionEntry:
    chunk: Chunk
    fusion_score: float = 0.0
    vector_distance: float | None = None
    bm25_score: float | None = None

class HybridRetrievalStrategy(RetrievalStrategy):
    """Combine vector and BM25 rankings using RRF."""

    def __init__(
        self,
        vector_strategy: RetrievalStrategy,
        bm25_strategy: RetrievalStrategy,
        candidate_limit: int = 10,
        rrf_k: int = 60,
    ) -> None:
        self._vector_strategy = vector_strategy
        self._bm25_strategy = bm25_strategy
        self._candidate_limit = candidate_limit
        self._rrf_k = rrf_k

    def search(
        self,
        query: str,
        limit: int,
    ) -> list[SearchResult]:

        started = perf_counter()
        
        vector_results = self._vector_strategy.search(
            query=query,
            limit=self._candidate_limit,
        )

        bm25_results = self._bm25_strategy.search(
            query=query,
            limit=self._candidate_limit,
        )

        logger.debug(
            "hybrid_candidates query=%r vector=%d bm25=%d",
            query,
            len(vector_results),
            len(bm25_results),
        )

        entries: dict[str, _FusionEntry] = {}

        for rank, result in enumerate(vector_results, start=1):
            entry = entries.setdefault(
                result.chunk.chunk_id,
                _FusionEntry(chunk=result.chunk),
            )

            # RRF contribution = 1 / (k + rank)
            entry.fusion_score += 1 / (self._rrf_k + rank)
            entry.vector_distance = result.vector_distance

        for rank, result in enumerate(bm25_results, start=1):
            entry = entries.setdefault(
                result.chunk.chunk_id,
                _FusionEntry(chunk=result.chunk),
            )

            entry.fusion_score += 1 / (self._rrf_k + rank)
            entry.bm25_score = result.bm25_score

        ranked_entries = sorted(
            entries.values(),
            key=lambda entry: entry.fusion_score,
            reverse=True,
        )

        duration_ms = (perf_counter() - started) * 1000

        logger.debug(
            "hybrid_fusion_completed unique_candidates=%d "
            "returned=%d duration_ms=%.2f",
            len(entries),
            min(limit, len(ranked_entries)),
            duration_ms,
        )

        return [
            SearchResult(
                chunk=entry.chunk,
                retrieval_method="hybrid",
                score=entry.fusion_score,
                vector_distance=entry.vector_distance,
                bm25_score=entry.bm25_score,
            )
            for entry in ranked_entries[:limit]
        ]

class Retriever:
    """Application-facing retrieval service."""

    def __init__(
        self,
        strategy: RetrievalStrategy,
    ) -> None:
        self._strategy = strategy

    def search(
        self,
        query: str,
        limit: int = 3,
    ) -> list[SearchResult]:
        normalized_query = query.strip()

        if not normalized_query:
            raise ValueError("Query cannot be empty")

        if limit <= 0:
            raise ValueError("Limit must be greater than zero")

        return self._strategy.search(
            query=normalized_query,
            limit=limit,
        )