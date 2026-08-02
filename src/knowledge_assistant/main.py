import argparse
from pathlib import Path
from typing import Sequence

from knowledge_assistant.chunking import chunk_document
from knowledge_assistant.document_loader import load_documents
from knowledge_assistant.embeddings import (
    EmbeddingProvider,
    SentenceTransformerEmbeddingProvider,
)
from knowledge_assistant.vector_store import LanceDBVectorStore
from knowledge_assistant.formatters import ConsoleFormatter
from knowledge_assistant.retrieval import Retriever


DOCUMENTS_PATH = Path("documents")
DATABASE_PATH = Path("data/lancedb")
TABLE_NAME = "knowledge_chunks_minilm_v1"


def create_vector_store() -> LanceDBVectorStore:
    """Create the configured local vector-store adapter."""

    return LanceDBVectorStore(
        database_path=DATABASE_PATH,
        table_name=TABLE_NAME,
    )


def create_embedding_provider() -> EmbeddingProvider:
    """Create the configured embedding provider."""

    return SentenceTransformerEmbeddingProvider()


def ingest_documents(source_path: Path) -> None:
    """Load, chunk, embed, and index local documents."""

    if source_path.is_file():
        documents = [load_document(source_path)]
    else:
        documents = load_documents(source_path)

    if not documents:
        print("No Markdown or text documents found.")
        return

    chunks = [
        chunk
        for document in documents
        for chunk in chunk_document(document)
    ]

    embedding_provider = create_embedding_provider()
    embeddings = embedding_provider.embed_chunks(chunks)

    vector_store = create_vector_store()
    vector_store.replace(chunks, embeddings)

    print("Ingestion completed.")
    print(f"Documents: {len(documents)}")
    print(f"Chunks: {len(chunks)}")
    print(f"Embeddings: {len(embeddings)}")
    print(f"Model: {embedding_provider.model_name}")
    print(f"Table: {TABLE_NAME}")


def search_documents(query: str, limit: int) -> None:
    """Search indexed documents."""

    retriever = Retriever(
        embedding_provider=create_embedding_provider(),
        vector_store=create_vector_store(),
    )

    results = retriever.search(
        query=query,
        limit=limit,
    )

    print(ConsoleFormatter.format_search_results(results))


def show_stats() -> None:
    """Print vector-index statistics."""

    stats = create_vector_store().stats()
    print(ConsoleFormatter.format_stats(stats))


def inspect_chunks(limit: int) -> None:
    """Print readable records from the vector table."""

    records = create_vector_store().inspect(limit=limit)
    print(ConsoleFormatter.format_records(records))


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""

    parser = argparse.ArgumentParser(
        prog="knowledge-assistant",
        description="Search local knowledge using semantic retrieval.",
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    ingest_parser = subparsers.add_parser(
        "ingest",
        help="Load and index documents.",
    )

    ingest_parser.add_argument(
        "path",
        nargs="?",
        type=Path,
        default=DOCUMENTS_PATH,
        help="File or directory to ingest. Default: documents/",
    )

    search_parser = subparsers.add_parser(
        "search",
        help="Search indexed documents.",
    )
    search_parser.add_argument(
        "query",
        help="Natural-language search query.",
    )
    search_parser.add_argument(
        "--limit",
        type=int,
        default=3,
        help="Maximum results to return. Default: 3.",
    )

    subparsers.add_parser(
        "stats",
        help="Show index statistics.",
    )

    inspect_parser = subparsers.add_parser(
        "inspect",
        help="Inspect indexed chunks.",
    )
    inspect_parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Maximum chunks to display. Default: 10.",
    )

    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Run the command-line application."""

    parser = build_parser()
    arguments = parser.parse_args(argv)

    try:
        if arguments.command == "ingest":
            ingest_documents(arguments.path)

        elif arguments.command == "search":
            search_documents(
                query=arguments.query,
                limit=arguments.limit,
            )

        elif arguments.command == "stats":
            show_stats()

        elif arguments.command == "inspect":
            inspect_chunks(limit=arguments.limit)

    except (FileNotFoundError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()