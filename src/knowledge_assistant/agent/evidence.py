from knowledge_assistant.agent.models import AgentStep


def select_synthesis_steps(
    steps: tuple[AgentStep, ...],
) -> tuple[AgentStep, ...]:
    """
    Select the strongest available evidence for final synthesis.

    Prefer focused, reranked answer evidence over broad search results.
    """

    answer_steps = tuple(
        step
        for step in steps
        if (
            step.tool_call.tool_name
            == "answer_from_documents"
        )
    )

    if answer_steps:
        return answer_steps

    return steps