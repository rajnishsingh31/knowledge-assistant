from pathlib import Path
from knowledge_assistant.application import IngestionResult

from knowledge_assistant.models import( 
    IndexStats, 
    SearchResult,
    SearchResult,
    GenerationTrace,
    RetrievalEvaluationSummary,
    StartupTimings
)


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

            lines = [
                f"{index}. {chunk.source_path.name}",
                f"Lines: {chunk.start_line}-{chunk.end_line}",
                f"Method: {result.retrieval_method}",
                f"Score: {result.score:.6f}",
            ]

            if result.vector_distance is not None:
                lines.append(
                    f"Vector distance: {result.vector_distance:.4f}"
                )

            if result.bm25_score is not None:
                lines.append(
                    f"BM25 score: {result.bm25_score:.4f}"
                )

            lines.extend(
                [
                    "-" * 60,
                    chunk.content,
                ]
            )

            sections.append("\n".join(lines))

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

    @staticmethod
    def format_generation_trace(
        trace: GenerationTrace,
        embedding_model_name: str,
        startup_timings: StartupTimings | None = None,
    ) -> str:
        """Format the complete RAG execution trace."""

        answer = trace.generated_answer
        context = trace.retrieved_context
        prompt = trace.prompt
        timings = trace.timings

        retrieved_results = ConsoleFormatter.format_search_results(
            list(context.results)
        )

        separator = "=" * 70

        lines: list[str] = [
            separator,
            "Knowledge Assistant — Explain",
            separator,
            "",
            "QUESTION",
            "-" * 70,
            context.query,
            "",
            "CONFIGURATION",
            "-" * 70,
            f"Retrieved chunks: {len(context.results)}",
            f"Embedding model: {embedding_model_name}",
            f"LLM provider: {answer.provider_name}",
            f"LLM model: {answer.model_name}",
            "",
        ]

        if startup_timings is not None:
            lines.extend(
                [
                    "APPLICATION STARTUP",
                    "-" * 70,
                    (
                        "Settings loading: "
                        f"{startup_timings.settings_loading_ms:.2f} ms"
                    ),
                    (
                        "Dependency/model construction: "
                        f"{startup_timings.dependency_construction_ms:.2f} ms"
                    ),
                    (
                        "Total startup: "
                        f"{startup_timings.total_startup_ms:.2f} ms"
                    ),
                    "",
                ]
            )

        lines.extend(
            [
                "PIPELINE TIMINGS",
                "-" * 70,
                f"Retrieval: {timings.retrieval_ms:.2f} ms",
                f"Reranking: {timings.reranking_ms:.2f} ms",
                (
                    "Prompt building: "
                    f"{timings.prompt_building_ms:.2f} ms"
                ),
                f"Generation: {timings.generation_ms:.2f} ms",
                f"Pipeline total: {timings.total_ms:.2f} ms",
                "",
                "RETRIEVED CHUNKS",
                "-" * 70,
                retrieved_results,
                "",
                "SYSTEM PROMPT",
                "-" * 70,
                prompt.system,
                "",
                "USER PROMPT",
                "-" * 70,
                prompt.user,
                "",
                "GENERATED ANSWER",
                "-" * 70,
                answer.content,
                "",
                separator,
            ]
        )

        return "\n".join(lines)

    @staticmethod
    def format_retrieval_evaluation(
        summary: RetrievalEvaluationSummary,
        include_details: bool = False,
    ) -> str:
        """Format retrieval evaluation metrics."""

        lines = [
            f"Strategy: {summary.strategy_name}",
            f"Cases: {summary.case_count}",
            (
                f"Top-1 accuracy: "
                f"{summary.top_1_accuracy:.1%} "
                f"({summary.top_1_hits}/{summary.case_count})"
            ),
            (
                f"Top-k accuracy: "
                f"{summary.top_k_accuracy:.1%} "
                f"({summary.top_k_hits}/{summary.case_count})"
            ),
        ]

        if include_details:
            lines.extend(
                [
                    "",
                    "Case results:",
                    "-" * 60,
                ]
            )

            for result in summary.results:
                status = (
                    "PASS"
                    if result.top_k_hit
                    else "FAIL"
                )

                lines.extend(
                    [
                        f"{status} — {result.case_id}",
                        f"Query: {result.query}",
                        (
                            "Expected: "
                            f"{', '.join(result.expected_documents)}"
                        ),
                        (
                            "Retrieved: "
                            f"{', '.join(result.retrieved_documents)}"
                        ),
                        (
                            f"Top-1: "
                            f"{'yes' if result.top_1_hit else 'no'}"
                        ),
                        (
                            f"Top-k: "
                            f"{'yes' if result.top_k_hit else 'no'}"
                        ),
                        "",
                    ]
                )

        return "\n".join(lines).rstrip()

    @staticmethod
    def format_ingestion_result(
        result: IngestionResult,
    ) -> str:
        """Format ingestion results for console output."""

        timings = result.timings

        return "\n".join(
            [
                "Ingestion completed.",
                "",
                f"Documents: {result.document_count}",
                f"Chunks: {result.chunk_count}",
                f"Embeddings: {result.embedding_count}",
                f"Embedding model: {result.embedding_model}",
                f"Table: {result.table_name}",
                "",
                "Pipeline timings",
                "-" * 60,
                (
                    f"Document loading: "
                    f"{timings.document_loading_ms:.2f} ms"
                ),
                f"Chunking: {timings.chunking_ms:.2f} ms",
                f"Embedding: {timings.embedding_ms:.2f} ms",
                f"Indexing: {timings.indexing_ms:.2f} ms",
                f"Total: {timings.total_ms:.2f} ms",
            ]
    )