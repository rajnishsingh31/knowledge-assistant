from pathlib import Path

from knowledge_assistant.document_loaders.base import (
    DocumentLoader,
)
from knowledge_assistant.document_loaders.common import (
    create_document,
    validate_file,
)
from knowledge_assistant.models import Document


class TextDocumentLoader(DocumentLoader):
    """Load UTF-8 Markdown and plain-text files."""

    @property
    def supported_extensions(self) -> frozenset[str]:
        return frozenset({".md", ".txt"})

    def load(self, path: Path) -> Document:
        resolved_path = validate_file(path)

        content = resolved_path.read_text(
            encoding="utf-8"
        )

        return create_document(
            path=resolved_path,
            content=content,
        )