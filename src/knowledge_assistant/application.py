from dataclasses import dataclass
from pathlib import Path

from knowledge_assistant.answering import AnswerService
from knowledge_assistant.chunking import chunk_document
from knowledge_assistant.config import Settings
from knowledge_assistant.document_loader import (
    load_document,
    load_documents,
)
from knowledge_assistant.embeddings import EmbeddingProvider
from knowledge_assistant.models import (
    GeneratedAnswer,
    GenerationTrace,
    IndexStats,
    SearchResult,
)
from knowledge_assistant.retrieval import (
    Retriever, 
    RetrievalStrategy
    )
from knowledge_assistant.vector_store import LanceDBVectorStore
from knowledge_assistant.evaluation import (
    RetrievalEvaluator,
    load_evaluation_cases,
)
from knowledge_assistant.models import (
    RetrievalEvaluationSummary,
)


@dataclass(frozen=True)
class IngestionResult:
    """Summary of a completed ingestion operation."""

    document_count: int
    chunk_count: int
    embedding_count: int
    embedding_model: str
    table_name: str


class KnowledgeAssistantApplication:
    """Application facade for knowledge-assistant use cases."""

    @property
    def default_retrieval_limit(self) -> int:
        return self._settings.retrieval.default_limit


    def __init__(
        self,
        settings: Settings,
        embedding_provider: EmbeddingProvider,
        vector_store: LanceDBVectorStore,
        retrieval_strategies: dict[str, RetrievalStrategy],
        answer_service: AnswerService,
    ) -> None:
        self._settings = settings
        self._embedding_provider = embedding_provider
        self._vector_store = vector_store
        self._retrieval_strategies = retrieval_strategies
        self._answer_service = answer_service

    def ingest(
        self,
        source_path: Path | None = None,
    ) -> IngestionResult:
        """Load, chunk, embed, and index documents."""

        path = source_path or self._settings.documents.path

        if path.is_file():
            documents = [load_document(path)]
        else:
            documents = load_documents(path)

        if not documents:
            raise ValueError(
                f"No supported documents found at: {path}"
            )

        chunks = [
            chunk
            for document in documents
            for chunk in chunk_document(
                document=document,
                max_lines=(
                    self._settings.documents.max_chunk_lines
                ),
                overlap_lines=(
                    self._settings.documents.overlap_lines
                ),
            )
        ]

        embeddings = self._embedding_provider.embed_chunks(chunks)

        self._vector_store.replace(
            chunks=chunks,
            embeddings=embeddings,
        )

        return IngestionResult(
            document_count=len(documents),
            chunk_count=len(chunks),
            embedding_count=len(embeddings),
            embedding_model=self._embedding_provider.model_name,
            table_name=self._settings.vector_store.table_name,
        )

    def _create_retriever(
        self,
        strategy_name: str | None = None,
    ) -> Retriever:
        selected_strategy = (
            strategy_name or self._settings.retrieval.strategy
        )

        strategy = self._retrieval_strategies.get(selected_strategy)

        if strategy is None:
            raise ValueError(
                f"Unsupported retrieval strategy: {selected_strategy}"
            )

        return Retriever(strategy=strategy)

    def create_retriever(
        self,
        strategy_name: str | None = None,
    ) -> Retriever:
        """Create a retriever using a configured strategy."""

        return self._create_retriever(strategy_name)

    def search(
        self,
        query: str,
        limit: int | None = None,
        strategy_name: str | None = None,
    ) -> list[SearchResult]:
        """Search indexed document chunks."""

        effective_limit = (
            limit or self._settings.retrieval.default_limit
        )

        retriever = self._create_retriever(strategy_name)

        return retriever.search(
            query=query,
            limit=effective_limit,
        )

    def ask(
        self,
        query: str,
        limit: int | None = None,
    ) -> GeneratedAnswer:
        """Generate a grounded answer."""

        effective_limit = (
            limit or self._settings.retrieval.default_limit
        )

        return self._answer_service.answer(
            query=query,
            retrieval_limit=effective_limit,
        )

    def explain(
        self,
        query: str,
        limit: int | None = None,
    ) -> GenerationTrace:
        """Return the complete retrieval and generation trace."""

        effective_limit = (
            limit or self._settings.retrieval.default_limit
        )

        return self._answer_service.generate_trace(
            query=query,
            retrieval_limit=effective_limit,
        )

    def stats(self) -> IndexStats:
        """Return vector-index statistics."""

        return self._vector_store.stats()

    def inspect(
        self,
        limit: int = 10,
    ) -> list[dict[str, object]]:
        """Return stored vector records for inspection."""

        return self._vector_store.inspect(limit=limit)

    def evaluate_retrieval(
        self,
        strategy_name: str,
        dataset_path: Path | None = None,
        top_k: int | None = None,
    ) -> RetrievalEvaluationSummary:
        """Evaluate one retrieval strategy."""

        effective_dataset_path = (
            dataset_path
            or self._settings.evaluation.dataset_path
        )

        effective_top_k = (
            top_k
            or self._settings.evaluation.top_k
        )

        cases = load_evaluation_cases(
            effective_dataset_path
        )

        retriever = self.create_retriever(strategy_name)

        evaluator = RetrievalEvaluator(
            retriever=retriever,
            strategy_name=strategy_name,
        )

        return evaluator.evaluate(
            cases=cases,
            top_k=effective_top_k,
        )

    @property
    def embedding_model_name(self) -> str:
        return self._embedding_provider.model_name