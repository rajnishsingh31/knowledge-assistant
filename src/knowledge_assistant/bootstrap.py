from knowledge_assistant.answering import AnswerService
from knowledge_assistant.application import (
    KnowledgeAssistantApplication,
)
from knowledge_assistant.config import Settings
from knowledge_assistant.embeddings import (
    EmbeddingProvider,
    SentenceTransformerEmbeddingProvider,
)
from knowledge_assistant.llm import (
    LLMProvider,
    OllamaProvider,
)
from knowledge_assistant.prompt_builder import PromptBuilder
from knowledge_assistant.retrieval import (
     BM25RetrievalStrategy, 
     HybridRetrievalStrategy,
     RetrievalStrategy, 
     VectorRetrievalStrategy,
     Retriever
)

from knowledge_assistant.vector_store import LanceDBVectorStore
from knowledge_assistant.reranking import (
    Reranker,
    IdentityReranker,
    CrossEncoderReranker,
    )


def create_retrieval_strategy(
    settings: Settings,
    embedding_provider: EmbeddingProvider,
    vector_store: LanceDBVectorStore,
) -> dict[str, RetrievalStrategy]:
    """Create the configured retrieval strategy."""

    vector_strategy = VectorRetrievalStrategy(
        embedding_provider=embedding_provider,
        vector_store=vector_store,
    )

    bm25_strategy = BM25RetrievalStrategy(
        vector_store=vector_store,
    )

    hybrid_strategy = HybridRetrievalStrategy(
    vector_strategy=vector_strategy,
    bm25_strategy=bm25_strategy,
    candidate_limit=settings.retrieval.candidate_limit,
    rrf_k=settings.retrieval.rrf_k,
   )

    retrieval_strategies: dict[str, RetrievalStrategy] = {
        "vector": vector_strategy,
        "bm25": bm25_strategy,
        "hybrid": hybrid_strategy,
    }

    return retrieval_strategies


def create_embedding_provider(
    settings: Settings,
) -> EmbeddingProvider:
    """Create the configured embedding provider."""

    if settings.embeddings.provider == "sentence-transformers":
        return SentenceTransformerEmbeddingProvider(
            model_name=settings.embeddings.model_name
        )

    raise ValueError(
        "Unsupported embedding provider: "
        f"{settings.embeddings.provider}"
    )


def create_vector_store(
    settings: Settings,
) -> LanceDBVectorStore:
    """Create the configured vector store."""

    if settings.vector_store.provider == "lancedb":
        return LanceDBVectorStore(
            database_path=settings.vector_store.database_path,
            table_name=settings.vector_store.table_name,
        )

    raise ValueError(
        "Unsupported vector-store provider: "
        f"{settings.vector_store.provider}"
    )


def create_llm_provider(
    settings: Settings,
) -> LLMProvider:
    """Create the configured LLM provider."""

    if settings.llm.provider == "ollama":
        return OllamaProvider(
            model_name=settings.llm.model_name,
            host=settings.llm.ollama_host,
            temperature=settings.llm.temperature,
        )

    raise ValueError(
        f"Unsupported LLM provider: {settings.llm.provider}"
    )

def create_reranker(
        settings: Settings,
    ) -> Reranker:
        if settings.reranking.strategy == "identity":
            return IdentityReranker()

        if settings.reranking.strategy == "cross-encoder":
            return CrossEncoderReranker(
                model_name=settings.reranking.model_name
            )

        raise ValueError(
            f"Unsupported reranker: {settings.reranking.strategy}"
        )


def create_application(
        settings: Settings,
    ) -> KnowledgeAssistantApplication:
        """Construct the complete application dependency graph."""

        embedding_provider = create_embedding_provider(settings)
        vector_store = create_vector_store(settings)

        retrieval_strategies = create_retrieval_strategy(
            settings=settings,
            embedding_provider=embedding_provider,
            vector_store=vector_store,
        )

        configured_strategy = retrieval_strategies[
            settings.retrieval.strategy
        ]

        retriever = Retriever(
                strategy=configured_strategy,
        )

        reranker = create_reranker(settings)
 
        answer_service = AnswerService(
            retriever=retriever,
            prompt_builder=PromptBuilder(),
            llm_provider=create_llm_provider(settings),
            reranker=reranker,
            retrieval_limit=settings.reranking.retrieval_limit,
            final_limit=settings.reranking.final_limit,
        )

        return KnowledgeAssistantApplication(
            settings=settings,
            embedding_provider=embedding_provider,
            vector_store=vector_store,
            retrieval_strategies=retrieval_strategies,
            reranker=reranker,
            answer_service=answer_service,
        )


from knowledge_assistant.retrieval import (
    BM25RetrievalStrategy,
    HybridRetrievalStrategy,
    RetrievalStrategy,
    VectorRetrievalStrategy,
)


