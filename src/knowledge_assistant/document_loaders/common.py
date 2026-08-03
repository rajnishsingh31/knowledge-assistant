from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

from knowledge_assistant.models import Document


def validate_file(path: Path) -> Path:
    """Validate and resolve an input file."""

    if not path.exists():
        raise FileNotFoundError(f"Document not found: {path}")

    if not path.is_file():
        raise ValueError(f"Path is not a file: {path}")

    return path.resolve()


def create_document(
    path: Path,
    content: str,
) -> Document:
    """Create a normalized Document with a stable identifier."""

    normalized_content = content.strip()

    if not normalized_content:
        raise ValueError(
            f"No usable text could be extracted from: {path}"
        )

    resolved_path = path.resolve()

    return Document(
        document_id=str(
            uuid5(NAMESPACE_URL, resolved_path.as_uri())
        ),
        source_path=resolved_path,
        content=normalized_content,
    )