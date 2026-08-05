from knowledge_assistant.application import (
    IncrementalIngestionResult,
)
from knowledge_assistant.models import (
    IndexStats,
    RetrievalFilter,
    SearchResult,
)
from knowledge_assistant.api.schemas import (
    IngestResponse,
    IngestionTimingsResponse,
    RetrievalFilterRequest,
    SearchResultResponse,
    StatsResponse,
)


def map_retrieval_filter(
    request: RetrievalFilterRequest | None,
) -> RetrievalFilter | None:
    """Convert an API filter request into the domain model."""

    if request is None:
        return None

    source_names = tuple(
        dict.fromkeys(
            value.strip()
            for value in request.source_names
            if value.strip()
        )
    )

    extensions = tuple(
        dict.fromkeys(
            normalized
            for value in request.extensions
            if (
                normalized := _normalize_extension(value)
            )
        )
    )

    retrieval_filter = RetrievalFilter(
        source_names=source_names,
        extensions=extensions,
    )

    return (
        None
        if retrieval_filter.is_empty
        else retrieval_filter
    )


def _normalize_extension(value: str) -> str:
    normalized = value.strip().lower()

    if not normalized:
        return ""

    return (
        normalized
        if normalized.startswith(".")
        else f".{normalized}"
    )


def map_search_result(
    result: SearchResult,
    rank: int,
) -> SearchResultResponse:
    chunk = result.chunk

    return SearchResultResponse(
        rank=rank,
        chunk_id=chunk.chunk_id,
        document_id=chunk.document_id,
        source_path=str(chunk.source_path),
        source_name=chunk.source_path.name,
        start_line=chunk.start_line,
        end_line=chunk.end_line,
        content=chunk.content,
        retrieval_method=result.retrieval_method,
        score=result.score,
        vector_distance=result.vector_distance,
        bm25_score=result.bm25_score,
        reranker_score=result.reranker_score,
    )


def map_search_results(
    results: list[SearchResult],
) -> list[SearchResultResponse]:
    return [
        map_search_result(result, rank)
        for rank, result in enumerate(
            results,
            start=1,
        )
    ]


def map_stats(
    stats: IndexStats,
) -> StatsResponse:
    return StatsResponse(
        table_name=stats.table_name,
        chunk_count=stats.chunk_count,
        document_count=stats.document_count,
        embedding_models=list(stats.embedding_models),
        dimensions=list(stats.dimensions),
    )


def map_ingestion_result(
    result: IncrementalIngestionResult,
) -> IngestResponse:
    return IngestResponse(
        discovered_document_count=(
            result.discovered_document_count
        ),
        added_document_count=result.added_document_count,
        updated_document_count=result.updated_document_count,
        deleted_document_count=result.deleted_document_count,
        unchanged_document_count=(
            result.unchanged_document_count
        ),
        embedded_chunk_count=result.embedded_chunk_count,
        reused_embedding_count=result.reused_embedding_count,
        embedding_model=result.embedding_model,
        table_name=result.table_name,
        timings=IngestionTimingsResponse(
            document_loading_ms=(
                result.timings.document_loading_ms
            ),
            chunking_ms=result.timings.chunking_ms,
            embedding_ms=result.timings.embedding_ms,
            indexing_ms=result.timings.indexing_ms,
            total_ms=result.timings.total_ms,
        ),
    )