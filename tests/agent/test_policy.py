from knowledge_assistant.agent.models import (
    AgentContext,
    FinalAnswerDecision,
    ToolCallDecision,
    AgentToolCall,
    AgentStep,
    AgentToolResult,
    AgentObservation,
)
from knowledge_assistant.agent.policy import (
    enforce_grounded_tool_policy,
    enforce_answer_evidence_policy,
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

def test_policy_requires_focused_evidence_after_search() -> None:
    search_step = AgentStep(
        step_number=1,
        tool_call=AgentToolCall(
            tool_name="search_documents",
            arguments={
                "query": "least privilege",
            },
        ),
        tool_result=AgentToolResult(
            tool_name="search_documents",
            observation=AgentObservation(
                content="Broad search evidence",
            ),
        ),
    )

    context = AgentContext(
        query="Explain least privilege",
        steps=(search_step,),
    )

    decision = FinalAnswerDecision(
        decision_type="final_answer",
        answer="Evidence is sufficient.",
    )

    result = enforce_answer_evidence_policy(
        context=context,
        decision=decision,
    )

    assert isinstance(
        result,
        ToolCallDecision,
    )

    assert (
        result.tool_call.tool_name
        == "answer_from_documents"
    )

    assert result.tool_call.arguments == {
        "query": "Explain least privilege"
    }

def test_policy_allows_finish_after_answer_evidence() -> None:
    answer_step = AgentStep(
        step_number=1,
        tool_call=AgentToolCall(
            tool_name="answer_from_documents",
            arguments={
                "query": "least privilege",
            },
        ),
        tool_result=AgentToolResult(
            tool_name="answer_from_documents",
            observation=AgentObservation(
                content="Focused reranked evidence",
            ),
        ),
    )

    context = AgentContext(
        query="Explain least privilege",
        steps=(answer_step,),
    )

    decision = FinalAnswerDecision(
        decision_type="final_answer",
        answer="Evidence is sufficient.",
    )

    result = enforce_answer_evidence_policy(
        context=context,
        decision=decision,
    )

    assert result == decision

def test_policy_allows_finish_after_index_stats() -> None:
    stats_step = AgentStep(
        step_number=1,
        tool_call=AgentToolCall(
            tool_name="get_index_stats",
            arguments={},
        ),
        tool_result=AgentToolResult(
            tool_name="get_index_stats",
            observation=AgentObservation(
                content='{"document_count": 10}',
            ),
        ),
    )

    context = AgentContext(
        query="How many documents are indexed?",
        steps=(stats_step,),
    )

    decision = FinalAnswerDecision(
        decision_type="final_answer",
        answer="Evidence is sufficient.",
    )

    result = enforce_answer_evidence_policy(
        context=context,
        decision=decision,
    )

    assert result == decision