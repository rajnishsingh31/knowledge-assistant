from knowledge_assistant.agent.models import (
    AgentCitation,
    AgentObservation,
)


def test_observation_stores_content_and_citations() -> None:
    observation = AgentObservation(
        content="Least privilege limits permissions.",
        citations=(
            AgentCitation(
                source_name="cloud-security.docx",
                start_line=4,
                end_line=8,
            ),
        ),
        metadata={
            "result_count": 1,
        },
    )

    assert (
        observation.content
        == "Least privilege limits permissions."
    )
    assert len(observation.citations) == 1
    assert (
        observation.citations[0].source_name
        == "cloud-security.docx"
    )
    assert observation.metadata["result_count"] == 1
    assert observation.is_error is False