from typing import Any

from knowledge_assistant.models import (
    GeneratedAnswer,
    IndexStats,
    SearchResult,
)


def serialize_search_result(
    result: SearchResult,
) -> dict[str, Any]:
    return {
        "source": result.chunk.source_path.name,
        "start_line": result.chunk.start_line,
        "end_line": result.chunk.end_line,
        "content": result.chunk.content,
        "retrieval_method": result.retrieval_method,
        "score": result.score,
        "vector_distance": result.vector_distance,
        "bm25_score": result.bm25_score,
        "reranker_score": result.reranker_score,
    }


def serialize_index_stats(
    stats: IndexStats,
) -> dict[str, Any]:
    return {
        "table_name": stats.table_name,
        "chunk_count": stats.chunk_count,
        "document_count": stats.document_count,
        "embedding_models": list(
            stats.embedding_models
        ),
        "dimensions": list(
            stats.dimensions
        ),
    }


def serialize_generated_answer(
    answer: GeneratedAnswer,
) -> dict[str, Any]:
    return {
        "answer": answer.content,
        "provider": answer.provider_name,
        "model": answer.model_name,
        "sources": [
            serialize_search_result(source)
            for source in answer.sources
        ],
    }