from abc import ABC, abstractmethod
from uuid import NAMESPACE_URL, uuid5
from hashlib import sha256
import re
from dataclasses import dataclass
from knowledge_assistant.models import Chunk, Document


@dataclass(frozen=True)
class DocumentSection:
    """A natural structural section of a document."""

    start_line: int
    end_line: int
    lines: tuple[str, ...]

    @property
    def content(self) -> str:
        return "\n".join(self.lines).strip()

    @property
    def line_count(self) -> int:
        return self.end_line - self.start_line + 1

class ChunkingStrategy(ABC):
    """Split a document into traceable chunks."""

    @abstractmethod
    def chunk(
        self,
        document: Document,
    ) -> list[Chunk]:
        """Return chunks for one document."""

def _create_chunk(
    document: Document,
    content: str,
    start_line: int,
    end_line: int,
) -> Chunk:
    normalized_content = content.strip()

    chunk_hash = sha256(
        normalized_content.encode("utf-8")
    ).hexdigest()

    chunk_id_source = (
        f"{document.document_id}:"
        f"{start_line}:"
        f"{end_line}:"
        f"{chunk_hash}"
    )

    chunk_id = sha256(
        chunk_id_source.encode("utf-8")
    ).hexdigest()

    return Chunk(
        chunk_id=chunk_id,
        document_id=document.document_id,
        source_path=document.source_path,
        content=normalized_content,
        start_line=start_line,
        end_line=end_line,
        document_hash=document.content_hash,
        chunk_hash=chunk_hash,
    )

def chunk_document(
    document: Document,
    max_lines: int = 8,
    overlap_lines: int = 2,
) -> list[Chunk]:
    """Split a document into overlapping line-based chunks."""

    return LineChunkingStrategy(
        max_lines=max_lines,
        overlap_lines=overlap_lines,
    ).chunk(document)

class LineChunkingStrategy(ChunkingStrategy):
    """Split documents into fixed-size overlapping line windows."""

    def __init__(
        self,
        max_lines: int,
        overlap_lines: int,
    ) -> None:
        if max_lines <= 0:
            raise ValueError(
                "max_lines must be greater than zero"
            )

        if overlap_lines < 0:
            raise ValueError(
                "overlap_lines cannot be negative"
            )

        if overlap_lines >= max_lines:
            raise ValueError(
                "overlap_lines must be smaller than max_lines"
            )

        self._max_lines = max_lines
        self._overlap_lines = overlap_lines

    def chunk(
        self,
        document: Document,
    ) -> list[Chunk]:
        lines = document.content.splitlines()

        if not lines:
            return []

        chunks: list[Chunk] = []
        step = self._max_lines - self._overlap_lines

        for start_index in range(0, len(lines), step):
            end_index = min(
                start_index + self._max_lines,
                len(lines),
            )

            chunk_lines = lines[start_index:end_index]

            if not any(line.strip() for line in chunk_lines):
                continue

            chunks.append(
                _create_chunk(
                    document=document,
                    content="\n".join(chunk_lines),
                    start_line=start_index + 1,
                    end_line=end_index,
                )
            )

            if end_index == len(lines):
                break

        return chunks

class StructureAwareChunkingStrategy(ChunkingStrategy):
    """Chunk documents using headings and structural markers."""

    _BOUNDARY_PATTERNS = (
        re.compile(r"^\s*#{1,6}\s+\S+"),
        re.compile(r"^\s*\[Page\s+\d+\]\s*$"),
        re.compile(r"^\s*\[Worksheet:\s*.+\]\s*$"),
        re.compile(r"^\s*\[Table\s+\d+\]\s*$"),
    )

    def __init__(
        self,
        max_chunk_lines: int,
        overlap_lines: int,
        max_section_lines: int,
    ) -> None:
        if max_chunk_lines <= 0:
            raise ValueError(
                "max_chunk_lines must be greater than zero"
            )

        if overlap_lines < 0:
            raise ValueError(
                "overlap_lines cannot be negative"
            )

        if overlap_lines >= max_chunk_lines:
            raise ValueError(
                "overlap_lines must be smaller than "
                "max_chunk_lines"
            )

        if max_section_lines <= 0:
            raise ValueError(
                "max_section_lines must be greater than zero"
            )

        self._max_chunk_lines = max_chunk_lines
        self._overlap_lines = overlap_lines
        self._max_section_lines = max_section_lines

        self._fallback_strategy = LineChunkingStrategy(
            max_lines=max_chunk_lines,
            overlap_lines=overlap_lines,
        )

    @classmethod
    def _is_boundary(cls, line: str) -> bool:
        return any(
            pattern.match(line)
            for pattern in cls._BOUNDARY_PATTERNS
        )

    def _split_sections(
        self,
        document: Document,
    ) -> list[DocumentSection]:
        lines = document.content.splitlines()

        sections: list[DocumentSection] = []
        current_lines: list[str] = []
        current_start_line = 1

        for line_number, line in enumerate(
            lines,
            start=1,
        ):

            is_boundary = self._is_boundary(line)

            if is_boundary and current_lines:
                sections.append(
                    DocumentSection(
                        start_line=current_start_line,
                        end_line=line_number - 1,
                        lines=tuple(current_lines),
                    )
                )

                current_lines = []
                current_start_line = line_number

            current_lines.append(line)

        if current_lines:
            sections.append(
                DocumentSection(
                    start_line=current_start_line,
                    end_line=len(lines),
                    lines=tuple(current_lines),
                )
            )

        return [
            section
            for section in sections
            if section.content
        ]

    def _merge_sections(
        self,
        sections: list[DocumentSection],
    ) -> list[DocumentSection]:
        merged_sections: list[DocumentSection] = []

        current_lines: list[str] = []
        current_start_line: int | None = None
        current_end_line: int | None = None

        for section in sections:
            proposed_line_count = (
                len(current_lines) + len(section.lines)
            )

            if (
                current_lines
                and proposed_line_count
                > self._max_chunk_lines
            ):
                assert current_start_line is not None
                assert current_end_line is not None

                merged_sections.append(
                    DocumentSection(
                        start_line=current_start_line,
                        end_line=current_end_line,
                        lines=tuple(current_lines),
                    )
                )

                current_lines = []
                current_start_line = None
                current_end_line = None

            if current_start_line is None:
                current_start_line = section.start_line

            current_lines.extend(section.lines)
            current_end_line = section.end_line

        if current_lines:
            assert current_start_line is not None
            assert current_end_line is not None

            merged_sections.append(
                DocumentSection(
                    start_line=current_start_line,
                    end_line=current_end_line,
                    lines=tuple(current_lines),
                )
            )

        return merged_sections

    def _split_oversized_section(
        self,
        document: Document,
        section: DocumentSection,
    ) -> list[Chunk]:
        temporary_document = Document(
            document_id=document.document_id,
            source_path=document.source_path,
            content=section.content,
            content_hash=document.content_hash,
        )

        temporary_chunks = self._fallback_strategy.chunk(
            temporary_document
        )

        adjusted_chunks: list[Chunk] = []

        line_offset = section.start_line - 1

        for chunk in temporary_chunks:
            adjusted_start_line = (
                chunk.start_line + line_offset
            )

            adjusted_end_line = (
                chunk.end_line + line_offset
            )

            adjusted_chunks.append(
                _create_chunk(
                    document=document,
                    content=chunk.content,
                    start_line=adjusted_start_line,
                    end_line=adjusted_end_line,
                )
            )

        return adjusted_chunks

    def chunk(
        self,
        document: Document,
    ) -> list[Chunk]:
        sections = self._split_sections(document)
        merged_sections = self._merge_sections(sections)

        chunks: list[Chunk] = []

        for section in merged_sections:
            if (
                section.line_count
                > self._max_section_lines
            ):
                chunks.extend(
                    self._split_oversized_section(
                        document=document,
                        section=section,
                    )
                )
            else:
                chunks.append(
                    _create_chunk(
                        document=document,
                        content=section.content,
                        start_line=section.start_line,
                        end_line=section.end_line,
                    )
                )

        return chunks