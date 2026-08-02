from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class Document:

    """A text document loaded from the local file system."""

    document_id: str
    source_path: Path
    content: str

@dataclass(frozen=True)
class Chunk:

    """A traceable section of a document."""

    chunk_id: str
    document_id: str
    source_path: Path
    content: str
    start_line: int
    end_line: int
    
@dataclass(frozen=True)
class Embedding:
    """An embedding generated for one document chunk."""

    chunk_id: str
    model_name: str
    dimensions: int
    vector: tuple[float, ...]

@dataclass(frozen=True)
class SearchResult:
    """A chunk returned by a retrieval query."""

    chunk: Chunk
    distance: float

@dataclass(frozen=True)
class IndexStats:
    """Summary information about a vector index."""

    table_name: str
    chunk_count: int
    document_count: int
    embedding_models: tuple[str, ...]
    dimensions: tuple[int, ...]

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