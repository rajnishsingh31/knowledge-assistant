from knowledge_assistant.agent.citations import (
    append_citations_if_missing,
    deduplicate_citations,
    format_agent_citation,
    remove_local_source_references,
)
from knowledge_assistant.agent.models import (
    AgentCitation,
)


def test_formats_citation_with_line_range() -> None:
    citation = AgentCitation(
        source_name="cloud-security.docx",
        start_line=7,
        end_line=14,
    )

    assert format_agent_citation(citation) == (
        "[cloud-security.docx, lines 7-14]"
    )


def test_formats_citation_without_lines() -> None:
    citation = AgentCitation(
        source_name="cloud-security.docx",
    )

    assert format_agent_citation(citation) == (
        "[cloud-security.docx]"
    )


def test_removes_local_source_references() -> None:
    content = (
        "Least privilege limits permissions "
        "[Source 1] and should be reviewed [Source 2]."
    )

    assert remove_local_source_references(content) == (
        "Least privilege limits permissions"
        " and should be reviewed."
    )


def test_deduplicates_citations() -> None:
    citation = AgentCitation(
        source_name="cloud-security.docx",
        start_line=7,
        end_line=14,
    )

    result = deduplicate_citations(
        (citation, citation)
    )

    assert result == (citation,)


def test_appends_citations_when_missing() -> None:
    citation = AgentCitation(
        source_name="cloud-security.docx",
        start_line=7,
        end_line=14,
    )

    result = append_citations_if_missing(
        content="Least privilege limits permissions.",
        citations=(citation,),
    )

    assert result == (
        "Least privilege limits permissions.\n\n"
        "Sources: "
        "[cloud-security.docx, lines 7-14]"
    )


def test_does_not_append_existing_citation() -> None:
    citation = AgentCitation(
        source_name="cloud-security.docx",
        start_line=7,
        end_line=14,
    )

    content = (
        "Least privilege limits permissions "
        "[cloud-security.docx, lines 7-14]."
    )

    assert append_citations_if_missing(
        content=content,
        citations=(citation,),
    ) == content