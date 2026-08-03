from pathlib import Path

import pytest

from knowledge_assistant.chunking import chunk_document
from knowledge_assistant.models import Document


def create_document(content: str) -> Document:
    return Document(
        document_id="document-1",
        source_path=Path("sample.md"),
        content=content,
        content_hash="document-hash-1",
    )


def test_chunk_document_creates_overlapping_chunks() -> None:
    document = create_document(
        "\n".join(
            [
                "line 1",
                "line 2",
                "line 3",
                "line 4",
                "line 5",
                "line 6",
            ]
        )
    )

    chunks = chunk_document(
        document=document,
        max_lines=4,
        overlap_lines=1,
    )

    assert len(chunks) == 2

    assert chunks[0].start_line == 1
    assert chunks[0].end_line == 4
    assert chunks[0].content == (
        "line 1\nline 2\nline 3\nline 4"
    )
    assert chunks[0].document_hash == "document-hash-1"

    assert chunks[1].start_line == 4
    assert chunks[1].end_line == 6
    assert chunks[1].content == (
        "line 4\nline 5\nline 6"
    )
    assert chunks[1].document_hash == "document-hash-1"
    assert chunks[0].chunk_hash != chunks[1].chunk_hash


def test_chunk_document_returns_one_chunk_for_short_document() -> None:
    document = create_document("line 1\nline 2")

    chunks = chunk_document(
        document=document,
        max_lines=8,
        overlap_lines=2,
    )

    assert len(chunks) == 1
    assert chunks[0].start_line == 1
    assert chunks[0].end_line == 2


@pytest.mark.parametrize(
    ("max_lines", "overlap_lines", "expected_message"),
    [
        (0, 0, "max_lines must be greater than zero"),
        (4, -1, "overlap_lines cannot be negative"),
        (4, 4, "overlap_lines must be smaller than max_lines"),
        (4, 5, "overlap_lines must be smaller than max_lines"),
    ],
)
def test_chunk_document_rejects_invalid_settings(
    max_lines: int,
    overlap_lines: int,
    expected_message: str,
) -> None:
    document = create_document("content")

    with pytest.raises(ValueError, match=expected_message):
        chunk_document(
            document=document,
            max_lines=max_lines,
            overlap_lines=overlap_lines,
        )