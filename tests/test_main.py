from knowledge_assistant.main import create_retrieval_filter


def test_create_retrieval_filter_normalizes_extensions() -> None:
    retrieval_filter = create_retrieval_filter(
        source_names=["cloud-security.docx"],
        extensions=["PDF", ".docx"],
    )

    assert retrieval_filter is not None
    assert retrieval_filter.source_names == (
        "cloud-security.docx",
    )
    assert retrieval_filter.extensions == (
        ".pdf",
        ".docx",
    )


def test_create_retrieval_filter_returns_none_when_empty() -> None:
    retrieval_filter = create_retrieval_filter(
        source_names=[],
        extensions=[],
    )

    assert retrieval_filter is None