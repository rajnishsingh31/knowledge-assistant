from pathlib import Path
from typing import Any

import lancedb

from knowledge_assistant.models import Chunk, Embedding, SearchResult, IndexStats


class LanceDBVectorStore:
    """Store and search document chunks using LanceDB."""

    @staticmethod
    def _row_to_chunk(row: dict[str, Any]) -> Chunk:
        return Chunk(
            chunk_id=row["chunk_id"],
            document_id=row["document_id"],
            source_path=Path(row["source_path"]),
            content=row["content"],
            start_line=int(row["start_line"]),
            end_line=int(row["end_line"]),
        )

    def __init__(
        self,
        database_path: Path,
        table_name: str,
    ) -> None:
        self._database_path = database_path
        self._table_name = table_name
        self._database = lancedb.connect(str(database_path))

    def replace(
        self,
        chunks: list[Chunk],
        embeddings: list[Embedding],
    ) -> None:
        """Replace the vector table with the supplied chunks."""

        if len(chunks) != len(embeddings):
            raise ValueError(
                "Chunks and embeddings must contain the same number of items"
            )

        if not chunks:
            raise ValueError("Cannot create an index without chunks")

        records: list[dict[str, Any]] = []

        for chunk, embedding in zip(chunks, embeddings, strict=True):
            if chunk.chunk_id != embedding.chunk_id:
                raise ValueError(
                    "Chunk and embedding IDs do not match: "
                    f"{chunk.chunk_id} != {embedding.chunk_id}"
                )

            records.append(
                {
                    "chunk_id": chunk.chunk_id,
                    "document_id": chunk.document_id,
                    "source_path": str(chunk.source_path),
                    "content": chunk.content,
                    "start_line": chunk.start_line,
                    "end_line": chunk.end_line,
                    "embedding_model": embedding.model_name,
                    "dimensions": embedding.dimensions,
                    "vector": list(embedding.vector),
                }
            )

        table = self._database.create_table(
            self._table_name,
            data=records,
            mode="overwrite",
        )

        # Creates BM25 index for the "content" column to enable text search
        table.create_fts_index(
            "content",
            replace=True,
        )


    def search_vector(
        self,
        query_vector: tuple[float, ...],
        limit: int = 10,
    ) -> list[SearchResult]:
        """Return chunks using vector similarity."""

        if not query_vector:
            raise ValueError("Query vector cannot be empty")

        if limit <= 0:
            raise ValueError("Limit must be greater than zero")

        table = self._database.open_table(self._table_name)

        rows = (
            table.search(list(query_vector))
            .limit(limit)
            .to_list()
        )

        return [
            SearchResult(
                chunk=self._row_to_chunk(row),
                retrieval_method="vector",
                score=-float(row["_distance"]),
                vector_distance=float(row["_distance"]),
            )
            for row in rows
        ]

    def search_text(
        self,
        query: str,
        limit: int = 10,
    ) -> list[SearchResult]:
        """Return chunks using BM25 full-text search."""

        normalized_query = query.strip()

        if not normalized_query:
            raise ValueError("Query cannot be empty")

        if limit <= 0:
            raise ValueError("Limit must be greater than zero")

        table = self._database.open_table(self._table_name)

        rows = (
            table.search(
                normalized_query,
                fts_columns="content",
            )
            .limit(limit)
            .to_list()
        )

        return [
            SearchResult(
                chunk=self._row_to_chunk(row),
                retrieval_method="bm25",
                score=float(row["_score"]),
                bm25_score=float(row["_score"]),
            )
            for row in rows
        ]

    def exists(self) -> bool:
        """Return whether the configured table exists."""

        return self._table_name in self._database.list_tables().tables


    def stats(self) -> IndexStats:
        """Return summary information about the vector table."""

        if not self.exists():
            raise ValueError(
                f"Vector table does not exist: {self._table_name}"
            )

        table = self._database.open_table(self._table_name)

        rows = table.to_arrow().to_pylist()

        if not rows:
            return IndexStats(
                table_name=self._table_name,
                chunk_count=0,
                document_count=0,
                embedding_models=(),
                dimensions=(),
            )

        document_ids = {
            row["document_id"]
            for row in rows
        }

        embedding_models = tuple(
            sorted(
                {
                    row["embedding_model"]
                    for row in rows
                }
            )
        )

        dimensions = tuple(
            sorted(
                {
                    int(row["dimensions"])
                    for row in rows
                }
            )
        )

        return IndexStats(
            table_name=self._table_name,
            chunk_count=len(rows),
            document_count=len(document_ids),
            embedding_models=embedding_models,
            dimensions=dimensions,
        )


    def inspect(self, limit: int = 10) -> list[dict[str, object]]:
        """Return readable chunk records without embedding vectors."""

        if limit <= 0:
            raise ValueError("Limit must be greater than zero")

        if not self.exists():
            raise ValueError(
                f"Vector table does not exist: {self._table_name}"
            )

        table = self._database.open_table(self._table_name)

        columns = [
            "chunk_id",
            "document_id",
            "source_path",
            "content",
            "start_line",
            "end_line",
            "embedding_model",
            "dimensions",
        ]

        rows = table.to_arrow().to_pylist()

        return rows[:limit]