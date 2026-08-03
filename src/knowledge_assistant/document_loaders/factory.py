from pathlib import Path

from knowledge_assistant.document_loaders.base import (
    DocumentLoader,
)
from knowledge_assistant.models import Document


class DocumentLoaderFactory:
    """Select a document loader based on file extension."""

    def __init__(
        self,
        loaders: list[DocumentLoader],
    ) -> None:
        if not loaders:
            raise ValueError(
                "At least one document loader is required"
            )

        self._loaders = tuple(loaders)

    @property
    def supported_extensions(self) -> frozenset[str]:
        return frozenset(
            extension
            for loader in self._loaders
            for extension in loader.supported_extensions
        )

    def load(self, path: Path) -> Document:
        for loader in self._loaders:
            if loader.supports(path):
                return loader.load(path)

        supported = ", ".join(
            sorted(self.supported_extensions)
        )

        raise ValueError(
            f"Unsupported document type: {path.suffix}. "
            f"Supported types: {supported}"
        )