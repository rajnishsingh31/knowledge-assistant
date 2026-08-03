from pathlib import Path

from knowledge_assistant.document_loaders.factory import (
    DocumentLoaderFactory,
)
from knowledge_assistant.models import Document


class DocumentService:
    """Load supported files and directories."""

    def __init__(
        self,
        loader_factory: DocumentLoaderFactory,
    ) -> None:
        self._loader_factory = loader_factory

    def load_document(
        self,
        path: Path,
    ) -> Document:
        return self._loader_factory.load(path)

    def load_documents(
        self,
        directory: Path,
    ) -> list[Document]:
        if not directory.exists():
            raise FileNotFoundError(
                f"Directory not found: {directory}"
            )

        if not directory.is_dir():
            raise ValueError(
                f"Path is not a directory: {directory}"
            )

        return [
            self._loader_factory.load(path)
            for path in sorted(directory.iterdir())
            if (
                path.is_file()
                and path.suffix.lower()
                in self._loader_factory.supported_extensions
            )
        ]