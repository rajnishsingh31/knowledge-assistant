from pathlib import Path
from typing import Any

import lancedb

from knowledge_assistant.models import Chunk, Embedding, SearchResult, IndexStats, StoredChunkEmbedding


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
            document_hash=row["document_hash"],
            chunk_hash=row["chunk_hash"],
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
        """Replace the complete vector table."""

        if not chunks:
            raise ValueError(
                "Cannot create an index without chunks"
            )

        records = self._create_records(
            chunks=chunks,
            embeddings=embeddings,
        )

        table = self._database.create_table(
            self._table_name,
            data=records,
            mode="overwrite",
        )

        table.create_fts_index(
            "content",
            replace=True,
        )
    
    def get_document_hashes(self) -> dict[str, str]:
        """Return indexed document IDs and their document hashes."""

        if not self.exists():
            return {}

        table = self._database.open_table(self._table_name)

        rows = (
            table.search()
            .select(["document_id", "document_hash"])
            .to_list()
        )

        return {
            str(row["document_id"]): str(row["document_hash"])
            for row in rows
    }


    def _create_records(
        self,
        chunks: list[Chunk],
        embeddings: list[Embedding],
    ) -> list[dict[str, Any]]:
        
        if len(chunks) != len(embeddings):
            raise ValueError(
                "Chunks and embeddings must contain the same number of items"
            )

        records: list[dict[str, Any]] = []

        for chunk, embedding in zip(
            chunks,
            embeddings,
            strict=True,
        ):
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
                    "document_hash": chunk.document_hash,
                    "chunk_hash": chunk.chunk_hash,
                    "embedding_model": embedding.model_name,
                    "dimensions": embedding.dimensions,
                    "vector": list(embedding.vector),
                }
            )

        return records

    

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
            "document_hash",
            "chunk_hash",
        ]

        rows = table.to_arrow().to_pylist()

        return rows[:limit]

    def add(
        self,
        chunks: list[Chunk],
        embeddings: list[Embedding],
    ) -> None:
        """Append chunks and embeddings to the vector table."""

        if not chunks:
            return

        records = self._create_records(
            chunks=chunks,
            embeddings=embeddings,
        )

        if not self.exists():
            table = self._database.create_table(
                self._table_name,
                data=records,
            )

            table.create_fts_index(
                "content",
                replace=True,
            )

            return

        table = self._database.open_table(
            self._table_name
        )

        table.add(records)
        table.optimize()

    def delete_documents(
        self,
        document_ids: set[str],
    ) -> None:
        """Delete all chunks belonging to the supplied documents."""

        if not document_ids or not self.exists():
            return

        table = self._database.open_table(
            self._table_name
        )

        escaped_ids = [
            document_id.replace("'", "''")
            for document_id in document_ids
        ]

        id_values = ", ".join(
            f"'{document_id}'"
            for document_id in escaped_ids
        )

        table.delete(
            f"document_id IN ({id_values})"
        )

        table.optimize()

    def get_chunk_embeddings(
        self,
        document_ids: set[str],
    ) -> dict[str, StoredChunkEmbedding]:
        """Return stored embeddings keyed by chunk hash."""

        if not document_ids or not self.exists():
            return {}

        escaped_ids = [
            document_id.replace("'", "''")
            for document_id in document_ids
        ]

        id_values = ", ".join(
            f"'{document_id}'"
            for document_id in escaped_ids
        )

        table = self._database.open_table(
            self._table_name
        )

        rows = (
            table.search()
            .where(
                f"document_id IN ({id_values})"
            )
            .select(
                [
                    "chunk_hash",
                    "embedding_model",
                    "dimensions",
                    "vector",
                ]
            )
            .to_list()
        )

        return {
            str(row["chunk_hash"]): StoredChunkEmbedding(
                chunk_hash=str(row["chunk_hash"]),
                model_name=str(row["embedding_model"]),
                dimensions=int(row["dimensions"]),
                vector=tuple(
                    float(value)
                    for value in row["vector"]
                ),
            )
            for row in rows
        }