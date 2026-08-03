from knowledge_assistant.models import RetrievalFilter
from knowledge_assistant.vector_store import LanceDBVectorStore


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