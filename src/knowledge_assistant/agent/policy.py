from knowledge_assistant.agent.models import (
    AgentContext,
    AgentToolCall,
    FinalAnswerDecision,
    PlannerDecision,
    ToolCallDecision,
)


_CONVERSATIONAL_QUERIES = {
    "hello",
    "hi",
    "hey",
    "thanks",
    "thank you",
    "good morning",
    "good afternoon",
    "good evening",
}

_CONVERSATIONAL_PREFIXES = (
    "say hello",
    "say hi",
    "greet me",
    "introduce yourself",
)


def _is_conversational_query(query: str) -> bool:
    normalized_query = (
        query.strip()
        .lower()
        .rstrip("!?.")
    )

    if normalized_query in _CONVERSATIONAL_QUERIES:
        return True

    return normalized_query.startswith(
        _CONVERSATIONAL_PREFIXES
    )


def enforce_grounded_tool_policy(
    context: AgentContext,
    decision: PlannerDecision,
) -> PlannerDecision:
    """Require a tool before answering factual requests."""

    if not isinstance(decision, FinalAnswerDecision):
        return decision

    if context.steps:
        return decision

    if _is_conversational_query(context.query):
        return decision

    return ToolCallDecision(
        decision_type="call_tool",
        tool_call=AgentToolCall(
            tool_name="answer_from_documents",
            arguments={
                "query": context.query,
            },
        ),
    )

def has_answer_evidence(
    context: AgentContext,
) -> bool:
    return any(
        step.tool_call.tool_name
        == "answer_from_documents"
        for step in context.steps
    )

def enforce_answer_evidence_policy(
    context: AgentContext,
    decision: PlannerDecision,
) -> PlannerDecision:
    """
    Require focused answer evidence only when the agent
    currently has broad search results and attempts to finish.
    """

    if not isinstance(
        decision,
        FinalAnswerDecision,
    ):
        return decision

    if not context.steps:
        return decision

    tool_names = {
        step.tool_call.tool_name
        for step in context.steps
    }

    # Focused evidence already exists.
    if "answer_from_documents" in tool_names:
        return decision

    # Stats, inspection, or other non-search tools may already
    # provide sufficient structured evidence.
    if tool_names != {"search_documents"}:
        return decision

    # The only evidence collected so far is broad search output.
    # Require focused/reranked evidence before finishing.
    return ToolCallDecision(
        decision_type="call_tool",
        tool_call=AgentToolCall(
            tool_name="answer_from_documents",
            arguments={
                "query": context.query,
            },
        ),
    )