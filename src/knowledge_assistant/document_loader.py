from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

from knowledge_assistant.models import Document


SUPPORTED_EXTENSIONS = {".md", ".txt"}


def load_document(file_path: Path) -> Document:
    """Load one supported UTF-8 document."""

    if not file_path.exists():
        raise FileNotFoundError(f"Document not found: {file_path}")

    if not file_path.is_file():
        raise ValueError(f"Path is not a file: {file_path}")

    if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported document type: {file_path.suffix}. "
            f"Supported types: {sorted(SUPPORTED_EXTENSIONS)}"
        )

    content = file_path.read_text(encoding="utf-8").strip()

    if not content:
        raise ValueError(f"Document is empty: {file_path}")

    resolved_path = file_path.resolve()

    return Document(
        document_id=str(uuid5(NAMESPACE_URL, resolved_path.as_uri())),
        source_path=resolved_path,
        content=content,
    )


def load_documents(directory: Path) -> list[Document]:
    """Load supported documents from a directory."""

    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")

    if not directory.is_dir():
        raise ValueError(f"Path is not a directory: {directory}")

    documents: list[Document] = []

    for file_path in sorted(directory.iterdir()):
        if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
            documents.append(load_document(file_path))

    return documents