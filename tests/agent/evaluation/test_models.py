from knowledge_assistant.agent.evaluation.models import (
    AgentEvaluationResult,
)


def test_evaluation_result_passes_when_all_checks_pass() -> None:
    result = AgentEvaluationResult(
        case_id="case-1",
        query="What is BM25?",
        actual_tools=(
            "search_documents",
            "answer_from_documents",
        ),
        actual_documents=("bm25.md",),
        stop_reason="final_answer",
        is_grounded=True,
        iteration_count=3,
        duration_ms=100.0,
        tool_match=True,
        document_match=True,
        stop_reason_match=True,
        grounding_match=True,
    )

    assert result.passed is True


def test_evaluation_result_fails_when_any_check_fails() -> None:
    result = AgentEvaluationResult(
        case_id="case-1",
        query="What is BM25?",
        actual_tools=("search_documents",),
        actual_documents=("bm25.md",),
        stop_reason="final_answer",
        is_grounded=True,
        iteration_count=2,
        duration_ms=100.0,
        tool_match=False,
        document_match=True,
        stop_reason_match=True,
        grounding_match=True,
    )

    assert result.passed is False