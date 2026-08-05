import logging
from dataclasses import dataclass
from pathlib import Path

from knowledge_assistant.answering import AnswerService
from knowledge_assistant.chunking import chunk_document
from knowledge_assistant.config import Settings
from knowledge_assistant.document_loader import (
    DocumentService,
)
from knowledge_assistant.embeddings import EmbeddingProvider
from knowledge_assistant.models import (
    GeneratedAnswer,
    GenerationTrace,
    IndexStats,
    RetrievalFilter,
    SearchResult,
    IngestionTimings,
    StartupTimings,
    Embedding,
    Chunk,
)
from knowledge_assistant.reranking import Reranker
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
from time import perf_counter

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class IncrementalIngestionResult:
    """Summary of incremental document ingestion."""

    discovered_document_count: int
    added_document_count: int
    updated_document_count: int
    deleted_document_count: int
    unchanged_document_count: int
    reused_embedding_count: int
    embedded_chunk_count: int
    embedding_model: str
    table_name: str
    timings: IngestionTimings


class KnowledgeAssistantApplication:
    """Application facade for knowledge-assistant use cases."""

    @property
    def default_retrieval_limit(self) -> int:
        return self._settings.retrieval.default_limit

    def __init__(
        self,
        settings: Settings,
        document_service: DocumentService,
        embedding_provider: EmbeddingProvider,
        vector_store: LanceDBVectorStore,
        retrieval_strategies: dict[str, RetrievalStrategy],
        reranker: Reranker,
        answer_service: AnswerService,
         startup_timings: StartupTimings | None = None,
    ) -> None:
        self._settings = settings
        self._document_service = document_service
        self._embedding_provider = embedding_provider
        self._vector_store = vector_store
        self._retrieval_strategies = retrieval_strategies
        self._reranker = reranker
        self._answer_service = answer_service
        self._startup_timings = startup_timings

    
    def record_startup_timings(
        self,
        timings: StartupTimings,
    ) -> None:
        self._startup_timings = timings

    @property
    def startup_timings(self) -> StartupTimings | None:
        return self._startup_timings

    def ingest(
        self,
        source_path: Path | None = None,
    ) -> IncrementalIngestionResult:
        """Incrementally synchronize documents with the vector index."""

        total_started = perf_counter()
        path = source_path or self._settings.documents.path

        loading_started = perf_counter()

        if path.is_file():
            documents = [
                self._document_service.load_document(path)
            ]
            synchronize_deletions = False
        else:
            documents = self._document_service.load_documents(path)
            synchronize_deletions = True

        document_loading_ms = (
            perf_counter() - loading_started
        ) * 1000

        if not documents and not synchronize_deletions:
            raise ValueError(
                f"No supported documents found at: {path}"
            )

        indexed_hashes = (
            self._vector_store.get_document_hashes()
        )

        discovered_documents = {
            document.document_id: document
            for document in documents
        }

        added_documents = [
            document
            for document in documents
            if document.document_id not in indexed_hashes
        ]

        updated_documents = [
            document
            for document in documents
            if (
                document.document_id in indexed_hashes
                and indexed_hashes[document.document_id]
                != document.content_hash
            )
        ]

        unchanged_documents = [
            document
            for document in documents
            if (
                document.document_id in indexed_hashes
                and indexed_hashes[document.document_id]
                == document.content_hash
            )
        ]

        deleted_document_ids: set[str] = set()

        updated_document_ids = {
            document.document_id
            for document in updated_documents
        }

        
        if synchronize_deletions:
            deleted_document_ids = (
                set(indexed_hashes)
                - set(discovered_documents)
            )

        changed_document_ids = {
            document.document_id
            for document in updated_documents
        }

        document_ids_to_delete = (
            changed_document_ids
            | deleted_document_ids
        )

        stored_embeddings = (
            self._vector_store.get_chunk_embeddings(
                updated_document_ids
            )
        )     
        
        chunking_started = perf_counter()

        documents_to_index = (
            added_documents + updated_documents
        )

        chunks = [
            chunk
            for document in documents_to_index
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

        chunking_ms = (
            perf_counter() - chunking_started
        ) * 1000

        embedding_started = perf_counter()

        reused_embeddings: list[Embedding] = []
        chunks_to_embed: list[Chunk] = []

        current_model_name = (
            self._embedding_provider.model_name
        )

        for chunk in chunks:
            stored_embedding = stored_embeddings.get(
                chunk.chunk_hash
            )

            if (
                stored_embedding is not None
                and stored_embedding.model_name
                == current_model_name
            ):
                reused_embeddings.append(
                    Embedding(
                        chunk_id=chunk.chunk_id,
                        model_name=stored_embedding.model_name,
                        dimensions=stored_embedding.dimensions,
                        vector=stored_embedding.vector,
                    )
                )
            else:
                chunks_to_embed.append(chunk)

        new_embeddings = (
            self._embedding_provider.embed_chunks(
                chunks_to_embed
            )
            if chunks_to_embed
            else []
        )

        embedding_by_chunk_id = {
            embedding.chunk_id: embedding
            for embedding in (
                reused_embeddings + new_embeddings
            )
        }

        embeddings = [
            embedding_by_chunk_id[chunk.chunk_id]
            for chunk in chunks
        ]

        embedding_ms = (
            perf_counter() - embedding_started
        ) * 1000

        indexing_started = perf_counter()

        self._vector_store.delete_documents(
            document_ids_to_delete
        )

        self._vector_store.add(
            chunks=chunks,
            embeddings=embeddings,
        )

        indexing_ms = (
            perf_counter() - indexing_started
        ) * 1000

        total_ms = (
            perf_counter() - total_started
        ) * 1000

        timings = IngestionTimings(
            document_loading_ms=document_loading_ms,
            chunking_ms=chunking_ms,
            embedding_ms=embedding_ms,
            indexing_ms=indexing_ms,
            total_ms=total_ms,
        )

        logger.debug(
            "incremental_ingestion_completed discovered=%d "
            "added=%d updated=%d deleted=%d unchanged=%d "
            "embedded_chunks=%d reused_embeddings=%d "
            "total_ms=%.2f",
            len(documents),
            len(added_documents),
            len(updated_documents),
            len(deleted_document_ids),
            len(unchanged_documents),
            len(new_embeddings),
            len(reused_embeddings),
            total_ms,
        )

        return IncrementalIngestionResult(
            discovered_document_count=len(documents),
            added_document_count=len(added_documents),
            updated_document_count=len(updated_documents),
            deleted_document_count=len(deleted_document_ids),
            unchanged_document_count=len(unchanged_documents),
            embedded_chunk_count=len(new_embeddings),
            reused_embedding_count=len(reused_embeddings),
            embedding_model=self._embedding_provider.model_name,
            table_name=self._settings.vector_store.table_name,
            timings=timings,
        )

    def rebuild(
        self,
        source_path: Path | None = None,
    ) -> IncrementalIngestionResult:
        """Delete and completely rebuild the index."""

        self._vector_store.drop()

        return self.ingest(source_path)
    
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
        retrieval_filter: RetrievalFilter | None = None,
    ) -> list[SearchResult]:
        """Search indexed document chunks."""

        effective_limit = (
            limit or self._settings.retrieval.default_limit
        )

        retriever = self._create_retriever(strategy_name)

        return retriever.search(
            query=query,
            limit=effective_limit,
            retrieval_filter=retrieval_filter,
        )

    def ask(
        self,
        query: str,
        limit: int | None = None,
        retrieval_filter: RetrievalFilter | None = None,
    ) -> GeneratedAnswer:
        """Generate a grounded answer."""

        effective_limit = (
            limit or self._settings.retrieval.default_limit
        )

        return self._answer_service.answer(
            query=query,
            retrieval_limit=effective_limit,
            retrieval_filter=retrieval_filter,
        )

    def explain(
        self,
        query: str,
        limit: int | None = None,
        retrieval_filter: RetrievalFilter | None = None,
    ) -> GenerationTrace:
        """Return the complete retrieval and generation trace."""

        effective_limit = (
            limit or self._settings.retrieval.default_limit
        )

        return self._answer_service.generate_trace(
            query=query,
            retrieval_limit=effective_limit,
            retrieval_filter=retrieval_filter,
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
        use_reranker: bool = False,
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

        evaluator = RetrievalEvaluator(
            retriever=retriever,
            strategy_name=(
                f"{strategy_name}+reranker"
                if use_reranker
                else strategy_name
            ),
            reranker=(
                self._reranker
                if use_reranker
                else None
            ),
            candidate_limit=(
                self._settings.reranking.retrieval_limit
                if use_reranker
                else None
            ),
        )

        return evaluator.evaluate(
            cases=cases,
            top_k=effective_top_k,
        )

    @property
    def embedding_model_name(self) -> str:
        return self._embedding_provider.model_name