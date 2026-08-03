from pathlib import Path

from knowledge_assistant.models import (
    Chunk,
    RetrievedContext,
    SearchResult,
)
from knowledge_assistant.prompt_builder import PromptBuilder


def create_result(
    filename: str,
    content: str,
    start_line: int,
    end_line: int,
) -> SearchResult:
    chunk = Chunk(
        chunk_id=f"chunk-{filename}",
        document_id=f"document-{filename}",
        source_path=Path(filename),
        content=content,
        start_line=start_line,
        end_line=end_line,
    )

    return SearchResult(
        chunk=chunk,
        retrieval_method="hybrid+reranked",
        score=0.95,
        reranker_score=0.95,
    )


def test_prompt_builder_numbers_and_formats_sources() -> None:
    context = RetrievedContext(
        query="What is BM25?",
        results=(
            create_result(
                filename="bm25.md",
                content="BM25 is a lexical ranking algorithm.",
                start_line=1,
                end_line=4,
            ),
            create_result(
                filename="rag.md",
                content="Hybrid retrieval combines retrieval methods.",
                start_line=10,
                end_line=14,
            ),
        ),
    )

    prompt = PromptBuilder().build(context)

    assert "Answer only from the supplied context." in prompt.system

    assert "Question:\nWhat is BM25?" in prompt.user

    assert "[Source 1]" in prompt.user
    assert "File: bm25.md" in prompt.user
    assert "Lines: 1-4" in prompt.user
    assert "BM25 is a lexical ranking algorithm." in prompt.user

    assert "[Source 2]" in prompt.user
    assert "File: rag.md" in prompt.user
    assert "Lines: 10-14" in prompt.user


def test_prompt_builder_includes_grounding_instruction() -> None:
    context = RetrievedContext(
        query="What is vector search?",
        results=(),
    )

    prompt = PromptBuilder().build(context)

    assert (
        "Answer the question using only the source context above."
        in prompt.user
    )