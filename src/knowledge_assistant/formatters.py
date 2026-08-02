from pathlib import Path

from knowledge_assistant.models import IndexStats, SearchResult


class ConsoleFormatter:
    """Format application results for terminal output."""

    @staticmethod
    def format_search_results(
        results: list[SearchResult],
    ) -> str:
        if not results:
            return "No results found."

        sections: list[str] = []

        for index, result in enumerate(results, start=1):
            chunk = result.chunk

            sections.append(
                "\n".join(
                    [
                        f"{index}. {chunk.source_path.name}",
                        (
                            f"Lines: "
                            f"{chunk.start_line}-{chunk.end_line}"
                        ),
                        f"Distance: {result.distance:.4f}",
                        "-" * 60,
                        chunk.content,
                    ]
                )
            )

        return "\n\n".join(sections)

    @staticmethod
    def format_stats(stats: IndexStats) -> str:
        models = ", ".join(stats.embedding_models)

        return "\n".join(
            [
                f"Table: {stats.table_name}",
                f"Chunks: {stats.chunk_count}",
                f"Documents: {stats.document_count}",
                f"Models: {models}",
                f"Dimensions: {stats.dimensions}",
            ]
        )

    @staticmethod
    def format_records(
        records: list[dict[str, object]],
    ) -> str:
        if not records:
            return "No indexed chunks found."

        sections: list[str] = []

        for index, record in enumerate(records, start=1):
            source_name = Path(
                str(record["source_path"])
            ).name

            sections.append(
                "\n".join(
                    [
                        f"{index}. {source_name}",
                        (
                            f"Lines: "
                            f"{record['start_line']}-"
                            f"{record['end_line']}"
                        ),
                        f"Chunk ID: {record['chunk_id']}",
                        f"Model: {record['embedding_model']}",
                        f"Dimensions: {record['dimensions']}",
                        "-" * 60,
                        str(record["content"]),
                    ]
                )
            )

        return "\n\n".join(sections)