from uuid import NAMESPACE_URL, uuid5
from hashlib import sha256
from knowledge_assistant.models import Chunk, Document


def chunk_document(
    document: Document,
    max_lines: int = 8,
    overlap_lines: int = 2,
) -> list[Chunk]:
    """Split a document into overlapping line-based chunks."""

    if max_lines <= 0:
        raise ValueError("max_lines must be greater than zero")

    if overlap_lines < 0:
        raise ValueError("overlap_lines cannot be negative")

    if overlap_lines >= max_lines:
        raise ValueError("overlap_lines must be smaller than max_lines")

    lines = document.content.splitlines()
    step = max_lines - overlap_lines
    chunks: list[Chunk] = []

    for start_index in range(0, len(lines), step):
        chunk_lines = lines[start_index : start_index + max_lines]
        content = "\n".join(chunk_lines).strip()

        if not content:
            continue

        start_line = start_index + 1
        end_line = start_index + len(chunk_lines)

        chunk_identity = (
            f"{document.document_id}:{start_line}:{end_line}"
        )

        chunk_hash = sha256(
            content.encode("utf-8")
        ).hexdigest()

        chunks.append(
            Chunk(
                chunk_id=str(uuid5(NAMESPACE_URL, chunk_identity)),
                document_id=document.document_id,
                source_path=document.source_path,
                content=content,
                start_line=start_line,
                end_line=end_line,
                document_hash=document.content_hash,
                chunk_hash=chunk_hash,
            )
        )

    return chunks