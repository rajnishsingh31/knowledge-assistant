# agent/policy.py

from knowledge_assistant.agent.models import (
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
    """Return whether the request needs no factual information."""

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
    query: str,
    decision: PlannerDecision,
) -> PlannerDecision:
    """Require document tools for non-conversational requests."""

    if not isinstance(decision, FinalAnswerDecision):
        return decision

    if _is_conversational_query(query):
        return decision

    return ToolCallDecision(
        decision_type="call_tool",
        tool_call=AgentToolCall(
            tool_name="answer_from_documents",
            arguments={
                "query": query.strip(),
            },
        ),
    )