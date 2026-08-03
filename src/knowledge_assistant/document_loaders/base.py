from abc import ABC, abstractmethod
from pathlib import Path

from knowledge_assistant.models import Document


class DocumentLoader(ABC):
    """Load one file into the application's Document model."""

    @property
    @abstractmethod
    def supported_extensions(self) -> frozenset[str]:
        """Return file extensions supported by this loader."""

    def supports(self, path: Path) -> bool:
        """Return whether this loader supports the supplied file."""

        return path.suffix.lower() in self.supported_extensions

    @abstractmethod
    def load(self, path: Path) -> Document:
        """Load and normalize one document."""