from pathlib import Path

from knowledge_assistant.models import( 
    IndexStats, 
    SearchResult,
    SearchResult,
    GenerationTrace,
    RetrievalEvaluationSummary
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
    ) -> str:

        """Format the complete RAG execution trace."""

        answer = trace.generated_answer
        context = trace.retrieved_context
        prompt = trace.prompt

        retrieved_results = ConsoleFormatter.format_search_results(
            list(context.results)
        )

        separator = "=" * 70

        return "\n".join(
            [
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
                f"Retrieval strategy: Vector search",
                f"Retrieval limit: {len(context.results)}",
                f"Embedding model: {embedding_model_name}",
                f"LLM provider: {answer.provider_name}",
                f"LLM model: {answer.model_name}",
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