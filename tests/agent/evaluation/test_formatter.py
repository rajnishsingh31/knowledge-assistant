from knowledge_assistant.agent.evaluation.formatter import (
    AgentEvaluationFormatter,
)
from knowledge_assistant.agent.evaluation.models import (
    AgentEvaluationResult,
    AgentEvaluationSummary,
)


def test_formatter_outputs_summary() -> None:
    result = AgentEvaluationResult(
        case_id="bm25",
        query="What is BM25?",
        actual_tools=(
            "answer_from_documents",
        ),
        actual_documents=("bm25.md",),
        stop_reason="final_answer",
        is_grounded=True,
        iteration_count=1,
        duration_ms=100.0,
        tool_match=True,
        document_match=True,
        stop_reason_match=True,
        grounding_match=True,
    )

    summary = AgentEvaluationSummary(
        case_count=1,
        passed_count=1,
        failed_count=0,
        tool_accuracy=1.0,
        document_accuracy=1.0,
        stop_reason_accuracy=1.0,
        grounding_accuracy=1.0,
        overall_accuracy=1.0,
        average_iterations=1.0,
        average_duration_ms=100.0,
        results=(result,),
    )

    output = (
        AgentEvaluationFormatter
        .format_summary(summary)
    )

    assert "Cases: 1" in output
    assert "Passed: 1" in output
    assert "Overall accuracy: 100.0%" in output