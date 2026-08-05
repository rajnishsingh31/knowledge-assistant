from typing import Literal

from pydantic import BaseModel, Field


RetrievalStrategyName = Literal[
    "vector",
    "bm25",
    "hybrid",
]


class HealthResponse(BaseModel):
    """API health status."""

    status: str
    application: str
    version: str


class RetrievalFilterRequest(BaseModel):
    """Optional metadata constraints."""

    source_names: list[str] = Field(default_factory=list)
    extensions: list[str] = Field(default_factory=list)


class SearchRequest(BaseModel):
    """Semantic, lexical, or hybrid search request."""

    query: str = Field(min_length=1)
    limit: int | None = Field(default=None, gt=0)
    strategy: RetrievalStrategyName | None = None
    filters: RetrievalFilterRequest | None = None


class SearchResultResponse(BaseModel):
    """One retrieved source chunk."""

    rank: int
    chunk_id: str
    document_id: str
    source_path: str
    source_name: str
    start_line: int
    end_line: int
    content: str
    retrieval_method: str
    score: float
    vector_distance: float | None = None
    bm25_score: float | None = None
    reranker_score: float | None = None


class SearchResponse(BaseModel):
    """Search results returned by the API."""

    query: str
    result_count: int
    results: list[SearchResultResponse]


class AskRequest(BaseModel):
    """Grounded question-answering request."""

    query: str = Field(min_length=1)
    limit: int | None = Field(default=None, gt=0)
    filters: RetrievalFilterRequest | None = None


class AskResponse(BaseModel):
    """Grounded answer and its supporting sources."""

    query: str
    answer: str
    provider_name: str
    model_name: str
    sources: list[SearchResultResponse]


class StatsResponse(BaseModel):
    """Vector-index statistics."""

    table_name: str
    chunk_count: int
    document_count: int
    embedding_models: list[str]
    dimensions: list[int]


class IngestRequest(BaseModel):
    """Server-local ingestion request."""

    path: str | None = None


class IngestionTimingsResponse(BaseModel):
    document_loading_ms: float
    chunking_ms: float
    embedding_ms: float
    indexing_ms: float
    total_ms: float


class IngestResponse(BaseModel):
    """Incremental ingestion summary."""

    discovered_document_count: int
    added_document_count: int
    updated_document_count: int
    deleted_document_count: int
    unchanged_document_count: int
    embedded_chunk_count: int
    reused_embedding_count: int
    embedding_model: str
    table_name: str
    timings: IngestionTimingsResponse