from knowledge_assistant.agent.evidence import (
    select_synthesis_steps,
)
from knowledge_assistant.agent.models import (
    AgentObservation,
    AgentStep,
    AgentToolCall,
    AgentToolResult,
)


def create_step(
    step_number: int,
    tool_name: str,
    content: str,
) -> AgentStep:
    return AgentStep(
        step_number=step_number,
        tool_call=AgentToolCall(
            tool_name=tool_name,  # type: ignore[arg-type]
            arguments={},
        ),
        tool_result=AgentToolResult(
            tool_name=tool_name,  # type: ignore[arg-type]
            observation=AgentObservation(
                content=content,
            ),
        ),
    )


def test_prefers_answer_evidence_over_search_results() -> None:
    search_step = create_step(
        step_number=1,
        tool_name="search_documents",
        content="Broad search evidence",
    )

    answer_step = create_step(
        step_number=2,
        tool_name="answer_from_documents",
        content="Focused reranked evidence",
    )

    selected = select_synthesis_steps(
        (search_step, answer_step)
    )

    assert selected == (answer_step,)


def test_uses_search_results_when_no_answer_evidence_exists() -> None:
    search_step = create_step(
        step_number=1,
        tool_name="search_documents",
        content="Search evidence",
    )

    selected = select_synthesis_steps(
        (search_step,)
    )

    assert selected == (search_step,)