from knowledge_assistant.agent.models import (
    AgentContext,
    FinalAnswerDecision,
    ToolCallDecision,
    AgentToolCall,
)
from knowledge_assistant.agent.policy import (
    enforce_grounded_tool_policy,
)

def test_policy_allows_conversational_answer() -> None:
    decision = FinalAnswerDecision(
        decision_type="final_answer",
        answer="Hello.",
    )

    result = enforce_grounded_tool_policy(
        context=AgentContext(
            query="Say hello",
            steps=(),
        ),
        decision=decision,
    )

    assert result == decision


def test_policy_forces_tool_for_knowledge_question() -> None:
    decision = FinalAnswerDecision(
        decision_type="final_answer",
        answer="Least privilege limits permissions.",
    )

    result = enforce_grounded_tool_policy(
        context=AgentContext(
            query="What is least privilege?",
            steps=(),
        ),
        decision=decision,
    )

    assert isinstance(result, ToolCallDecision)
    assert (
        result.tool_call.tool_name
        == "answer_from_documents"
    )
    assert result.tool_call.arguments == {
        "query": "What is least privilege?"
    }


def test_policy_preserves_existing_tool_call() -> None:
    decision = ToolCallDecision(
        decision_type="call_tool",
        tool_call=AgentToolCall(
            tool_name="get_index_stats",
            arguments={},
        ),
    )

    result = enforce_grounded_tool_policy(
        context=AgentContext(
            query="How many documents are indexed?",
            steps=(),
        ),
        decision=decision,
    )

    assert result == decision