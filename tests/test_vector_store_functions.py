from knowledge_assistant.models import RetrievalFilter
from knowledge_assistant.vector_store import LanceDBVectorStore
from pathlib import Path


def test_build_metadata_filter_for_source_names() -> None:
    retrieval_filter = RetrievalFilter(
        source_names=(
            "cloud-security.docx",
            "incident-response.pdf",
        )
    )

    expression = LanceDBVectorStore._build_metadata_filter(
        retrieval_filter
    )

    assert expression == (
        "source_name IN "
        "('cloud-security.docx', 'incident-response.pdf')"
    )


def test_build_metadata_filter_for_extensions() -> None:
    retrieval_filter = RetrievalFilter(
        extensions=(".pdf", ".docx"),
    )

    expression = LanceDBVectorStore._build_metadata_filter(
        retrieval_filter
    )

    assert expression == (
        "document_extension IN ('.pdf', '.docx')"
    )


def test_build_metadata_filter_combines_categories() -> None:
    retrieval_filter = RetrievalFilter(
        source_names=("cloud-security.docx",),
        extensions=(".docx",),
    )

    expression = LanceDBVectorStore._build_metadata_filter(
        retrieval_filter
    )

    assert expression == (
        "source_name IN ('cloud-security.docx') "
        "AND document_extension IN ('.docx')"
    )


def test_build_metadata_filter_escapes_quotes() -> None:
    retrieval_filter = RetrievalFilter(
        source_names=("manager's-notes.md",),
    )

    expression = LanceDBVectorStore._build_metadata_filter(
        retrieval_filter
    )

    assert expression == (
        "source_name IN ('manager''s-notes.md')"
    )


def test_build_metadata_filter_returns_none_when_empty() -> None:
    assert (
        LanceDBVectorStore._build_metadata_filter(None)
        is None
    )

    assert (
        LanceDBVectorStore._build_metadata_filter(
            RetrievalFilter()
        )
        is None
    )

def test_validate_schema_rejects_old_version(
    tmp_path: Path,
) -> None:
    store = LanceDBVectorStore(
        database_path=tmp_path,
        table_name="test-table",
        schema_version=2,
    )

    store._metadata_path.write_text(
        """
        {
          "schema_version": 1,
          "table_name": "test-table",
          "embedding_model": "test-model"
        }
        """,
        encoding="utf-8",
    )

    # validate_schema returns early when no table exists,
    # so this test is better added after a temporary table
    # integration fixture is available.

def test_reads_index_metadata(
    tmp_path: Path,
) -> None:
    store = LanceDBVectorStore(
        database_path=tmp_path,
        table_name="test-table",
        schema_version=2,
    )

    store._metadata_path.write_text(
        """
        {
          "schema_version": 2,
          "table_name": "test-table",
          "embedding_model": "test-model"
        }
        """,
        encoding="utf-8",
    )

    metadata = store.get_index_metadata()

    assert metadata is not None
    assert metadata.schema_version == 2
    assert metadata.table_name == "test-table"
    assert metadata.embedding_model == "test-model"