from knowledge_assistant.agent.models import AgentCitation
import re

_LOCAL_SOURCE_REFERENCE = re.compile(
    r"\s*\[Source\s+\d+\]",
    flags=re.IGNORECASE,
)


def remove_local_source_references(
    content: str,
) -> str:
    """Remove observation-local source-number references."""

    return _LOCAL_SOURCE_REFERENCE.sub(
        "",
        content,
    ).strip()


def format_agent_citation(
    citation: AgentCitation,
) -> str:
    """Format a stable document citation."""

    if (
        citation.start_line is not None
        and citation.end_line is not None
    ):
        return (
            f"[{citation.source_name}, "
            f"lines {citation.start_line}-{citation.end_line}]"
        )

    return f"[{citation.source_name}]"


def deduplicate_citations(
    citations: tuple[AgentCitation, ...],
) -> tuple[AgentCitation, ...]:
    """Remove duplicate citations while preserving order."""

    return tuple(
        dict.fromkeys(citations)
    )

def append_citations_if_missing(
    content: str,
    citations: tuple[AgentCitation, ...],
) -> str:
    """Append available citations when none appear in the answer."""

    normalized_content = content.strip()

    if not citations:
        return normalized_content

    expected_labels = tuple(
        format_agent_citation(citation)
        for citation in citations
    )

    if any(
        label in normalized_content
        for label in expected_labels
    ):
        return normalized_content

    citation_text = " ".join(expected_labels)

    return f"{normalized_content}\n\nSources: {citation_text}"