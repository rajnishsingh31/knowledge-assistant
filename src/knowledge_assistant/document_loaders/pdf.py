from pathlib import Path

from pypdf import PdfReader

from knowledge_assistant.document_loaders.base import DocumentLoader
from knowledge_assistant.document_loaders.common import (
    create_document,
    validate_file,
)
from knowledge_assistant.models import Document


class PdfDocumentLoader(DocumentLoader):
    """Extract text from text-based PDF files."""

    @property
    def supported_extensions(self) -> frozenset[str]:
        return frozenset({".pdf"})

    def load(self, path: Path) -> Document:
        resolved_path = validate_file(path)
        reader = PdfReader(str(resolved_path))

        page_sections: list[str] = []

        for page_number, page in enumerate(
            reader.pages,
            start=1,
        ):
            page_text = page.extract_text() or ""
            normalized_text = page_text.strip()

            if not normalized_text:
                continue

            page_sections.append(
                "\n".join(
                    [
                        f"[Page {page_number}]",
                        normalized_text,
                    ]
                )
            )

        return create_document(
            path=resolved_path,
            content="\n\n".join(page_sections),
        )