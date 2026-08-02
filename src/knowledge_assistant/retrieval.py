from knowledge_assistant.embeddings import EmbeddingProvider
from knowledge_assistant.models import SearchResult
from knowledge_assistant.vector_store import LanceDBVectorStore


class Retriever:
    """Retrieve semantically relevant document chunks."""

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
        limit: int = 3,
    ) -> list[SearchResult]:
        normalized_query = query.strip()

        if not normalized_query:
            raise ValueError("Query cannot be empty")

        if limit <= 0:
            raise ValueError("Limit must be greater than zero")

        if not self._vector_store.exists():
            raise ValueError(
                "The index does not exist. Run "
                "'knowledge-assistant ingest' first."
            )

        query_vector = self._embedding_provider.embed_query(
            normalized_query
        )

        return self._vector_store.search(
            query_vector=query_vector,
            limit=limit,
        )