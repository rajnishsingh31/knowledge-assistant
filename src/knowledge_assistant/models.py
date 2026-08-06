from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

@dataclass(frozen=True)
class PipelineTimings:
    """Execution durations for one answer-generation request."""

    retrieval_ms: float
    reranking_ms: float
    prompt_building_ms: float
    generation_ms: float
    total_ms: float

@dataclass(frozen=True)
class IngestionTimings:
    document_loading_ms: float
    chunking_ms: float
    embedding_ms: float
    indexing_ms: float
    total_ms: float

@dataclass(frozen=True)
class StartupTimings:
    """Application startup durations."""

    settings_loading_ms: float
    dependency_construction_ms: float
    total_startup_ms: float

@dataclass(frozen=True)
class Document:

    """A text document loaded from the local file system."""

    document_id: str
    source_path: Path
    content: str
    content_hash: str

@dataclass(frozen=True)
class Chunk:

    """A traceable section of a document."""

    chunk_id: str
    document_id: str
    source_path: Path
    content: str
    start_line: int
    end_line: int
    document_hash: str
    chunk_hash: str
    
@dataclass(frozen=True)
class Embedding:
    """An embedding generated for one document chunk."""

    chunk_id: str
    model_name: str
    dimensions: int
    vector: tuple[float, ...]

@dataclass(frozen=True)
class StoredChunkEmbedding:
    """An indexed embedding associated with chunk content."""

    chunk_hash: str
    model_name: str
    dimensions: int
    vector: tuple[float, ...]

@dataclass(frozen=True)
class IndexStats:
    """Summary information about a vector index."""

    table_name: str
    chunk_count: int
    document_count: int
    embedding_models: tuple[str, ...]
    dimensions: tuple[int, ...]

@dataclass(frozen=True)
class SearchResult:
    """A chunk returned by a retrieval query."""

    chunk: Chunk
    retrieval_method: str
    score: float
    vector_distance: float | None = None
    bm25_score: float | None = None
    reranker_score: float | None = None

@dataclass(frozen=True)
class RetrievedContext:
    """Retrieved evidence prepared for answer generation."""

    query: str
    results: tuple[SearchResult, ...]


@dataclass(frozen=True)
class GeneratedAnswer:
    """An answer produced from retrieved evidence."""

    content: str
    provider_name: str
    model_name: str
    sources: tuple[SearchResult, ...]

@dataclass(frozen=True)
class Prompt:
    """A provider-neutral prompt sent to an LLM."""

    system: str
    user: str


@dataclass(frozen=True)
class GenerationTrace:
    """Complete trace of one grounded answer-generation request."""

    retrieved_context: RetrievedContext
    prompt: Prompt
    generated_answer: GeneratedAnswer
    timings: PipelineTimings

@dataclass(frozen=True)
class EvaluationCase:
    """One retrieval evaluation case."""

    case_id: str
    query: str
    expected_documents: tuple[str, ...]


@dataclass(frozen=True)
class EvaluationCaseResult:
    """Result of evaluating one query."""

    case_id: str
    query: str
    expected_documents: tuple[str, ...]
    retrieved_documents: tuple[str, ...]
    top_1_hit: bool
    top_k_hit: bool


@dataclass(frozen=True)
class RetrievalEvaluationSummary:
    """Aggregated retrieval evaluation metrics."""

    strategy_name: str
    case_count: int
    top_1_hits: int
    top_k_hits: int
    top_1_accuracy: float
    top_k_accuracy: float
    results: tuple[EvaluationCaseResult, ...]

@dataclass(frozen=True)
class RetrievalFilter:
    """Optional metadata constraints for retrieval."""

    source_names: tuple[str, ...] = ()
    extensions: tuple[str, ...] = ()

    @property
    def is_empty(self) -> bool:
        return not self.source_names and not self.extensions

@dataclass(frozen=True)
class IndexMetadata:
    """Metadata describing the persisted vector index."""

    schema_version: int
    table_name: str
    embedding_model: str

