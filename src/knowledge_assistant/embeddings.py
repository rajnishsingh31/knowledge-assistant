from abc import ABC, abstractmethod

from sentence_transformers import SentenceTransformer

from knowledge_assistant.models import Chunk, Embedding


DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class EmbeddingProvider(ABC):
    """Contract for generating document and query embeddings."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the embedding model identifier."""

    @abstractmethod
    def embed_chunks(
        self,
        chunks: list[Chunk],
    ) -> list[Embedding]:
        """Generate embeddings for document chunks."""

    @abstractmethod
    def embed_query(self, query: str) -> tuple[float, ...]:
        """Generate an embedding for a search query."""


class SentenceTransformerEmbeddingProvider(EmbeddingProvider):
    """Generate embeddings locally with Sentence Transformers."""

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME) -> None:
        self._model_name = model_name
        self._model = SentenceTransformer(model_name)

    @property
    def model_name(self) -> str:
        return self._model_name

    def embed_chunks(
        self,
        chunks: list[Chunk],
    ) -> list[Embedding]:
        if not chunks:
            return []

        texts = [chunk.content for chunk in chunks]

        vectors = self._model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        embeddings: list[Embedding] = []

        for chunk, vector in zip(chunks, vectors, strict=True):
            values = tuple(float(value) for value in vector)

            embeddings.append(
                Embedding(
                    chunk_id=chunk.chunk_id,
                    model_name=self.model_name,
                    dimensions=len(values),
                    vector=values,
                )
            )

        return embeddings

    def embed_query(self, query: str) -> tuple[float, ...]:
        normalized_query = query.strip()

        if not normalized_query:
            raise ValueError("Query cannot be empty")

        vector = self._model.encode(
            normalized_query,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        return tuple(float(value) for value in vector)