from pathlib import Path

from knowledge_assistant.chunking import (
    StructureAwareChunkingStrategy,
)
from knowledge_assistant.models import Document


def create_document(content: str) -> Document:
    return Document(
        document_id="document-1",
        source_path=Path("sample.md"),
        content=content,
        content_hash="document-hash-1",
    )


def test_structure_chunker_splits_on_markdown_headings() -> None:
    document = create_document(
        "\n".join(
            [
                "# Retrieval",
                "Retrieval finds relevant chunks.",
                "",
                "## Vector Search",
                "Vector search uses embeddings.",
                "",
                "## BM25",
                "BM25 uses lexical matching.",
            ]
        )
    )

    chunker = StructureAwareChunkingStrategy(
        max_chunk_lines=3,
        overlap_lines=1,
        max_section_lines=8,
    )

    chunks = chunker.chunk(document)

    assert len(chunks) == 3

    assert chunks[0].content.startswith(
        "# Retrieval"
    )

    assert chunks[1].content.startswith(
        "## Vector Search"
    )

    assert chunks[2].content.startswith(
        "## BM25"
    )


def test_structure_chunker_splits_pdf_pages() -> None:
    document = create_document(
        "\n".join(
            [
                "[Page 1]",
                "Preparation defines ownership.",
                "",
                "[Page 2]",
                "Detection uses logs and metrics.",
            ]
        )
    )

    chunker = StructureAwareChunkingStrategy(
        max_chunk_lines=3,
        overlap_lines=1,
        max_section_lines=8,
    )

    chunks = chunker.chunk(document)

    assert len(chunks) == 2
    assert chunks[0].content.startswith("[Page 1]")
    assert chunks[1].content.startswith("[Page 2]")


def test_structure_chunker_splits_worksheets() -> None:
    document = create_document(
        "\n".join(
            [
                "[Worksheet: Test Types]",
                "Unit Test | Validate one function",
                "",
                "[Worksheet: Quality Metrics]",
                "Pass Rate | Percentage of passing tests",
            ]
        )
    )

    chunker = StructureAwareChunkingStrategy(
        max_chunk_lines=3,
        overlap_lines=1,
        max_section_lines=8,
    )

    chunks = chunker.chunk(document)

    assert len(chunks) == 2
    assert "[Worksheet: Test Types]" in chunks[0].content
    assert "[Worksheet: Quality Metrics]" in chunks[1].content


def test_oversized_section_uses_line_fallback() -> None:
    document = create_document(
        "\n".join(
            [
                "# Long Section",
                "line 2",
                "line 3",
                "line 4",
                "line 5",
                "line 6",
            ]
        )
    )

    chunker = StructureAwareChunkingStrategy(
        max_chunk_lines=3,
        overlap_lines=1,
        max_section_lines=4,
    )

    chunks = chunker.chunk(document)

    assert len(chunks) >= 2
    assert chunks[0].start_line == 1
    assert chunks[-1].end_line == 6