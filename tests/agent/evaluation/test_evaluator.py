import pytest
from knowledge_assistant.agent.evaluation.evaluator import (
    AgentEvaluator,
)
from knowledge_assistant.agent.evaluation.models import (
    AgentEvaluationCase,
)
from knowledge_assistant.agent.models import (
    AgentCitation,
    AgentIteration,
    AgentObservation,
    AgentResponse,
    AgentStep,
    AgentToolCall,
    AgentToolResult,
    GroundingValidationTrace,
    ToolCallDecision,
)
from knowledge_assistant.llm.models import (
    GroundingClaimResult,
)


class StubAgentRuntime:
    def run(
        self,
        query: str,
    ) -> AgentResponse:
        step = AgentStep(
            step_number=1,
            tool_call=AgentToolCall(
                tool_name="answer_from_documents",
                arguments={
                    "query": query,
                },
            ),
            tool_result=AgentToolResult(
                tool_name="answer_from_documents",
                observation=AgentObservation(
                    content="BM25 evidence",
                    citations=(
                        AgentCitation(
                            source_name="bm25.md",
                            start_line=1,
                            end_line=8,
                        ),
                    ),
                ),
            ),
        )

        decision = ToolCallDecision(
            decision_type="call_tool",
            tool_call=step.tool_call,
        )

        iteration = AgentIteration(
            iteration_number=1,
            decision=decision,
            tool_result=step.tool_result,
        )

        grounding = GroundingValidationTrace(
            is_grounded=True,
            claims=(
                GroundingClaimResult(
                    sentence="BM25 is a retrieval algorithm.",
                    supported=True,
                    reason="Supported by evidence.",
                ),
            ),
        )

        return AgentResponse(
            query=query,
            answer="BM25 is a retrieval algorithm.",
            steps=(step,),
            iterations=(iteration,),
            provider_name="stub",
            model_name="stub-model",
            stop_reason="final_answer",
            grounding_validation=grounding,
        )

def test_evaluator_extracts_agent_behavior() -> None:
    evaluator = AgentEvaluator(
        runtime=StubAgentRuntime(),  # type: ignore[arg-type]
    )

    case = AgentEvaluationCase(
        case_id="bm25",
        query="What is BM25?",
        expected_tools=(
            "answer_from_documents",
        ),
        expected_documents=(
            "bm25.md",
        ),
    )

    result = evaluator.evaluate(case)

    assert result.passed is True
    assert result.actual_tools == (
        "answer_from_documents",
    )
    assert result.actual_documents == (
        "bm25.md",
    )
    assert result.is_grounded is True
    assert result.iteration_count == 1
    assert result.duration_ms >= 0

def test_evaluator_aggregates_suite() -> None:
    evaluator = AgentEvaluator(
        runtime=StubAgentRuntime(),  # type: ignore[arg-type]
    )

    cases = (
        AgentEvaluationCase(
            case_id="case-1",
            query="What is BM25?",
            expected_tools=(
                "answer_from_documents",
            ),
            expected_documents=(
                "bm25.md",
            ),
        ),
        AgentEvaluationCase(
            case_id="case-2",
            query="Explain BM25.",
            expected_tools=(
                "answer_from_documents",
            ),
            expected_documents=(
                "bm25.md",
            ),
        ),
    )

    summary = evaluator.evaluate_suite(
        cases
    )

    assert summary.case_count == 2
    assert summary.passed_count == 2
    assert summary.failed_count == 0

    assert summary.overall_accuracy == 1.0
    assert summary.tool_accuracy == 1.0
    assert summary.document_accuracy == 1.0
    assert summary.grounding_accuracy == 1.0

    assert summary.average_iterations == 1.0
    assert summary.average_duration_ms >= 0


def test_evaluator_rejects_empty_suite() -> None:
    evaluator = AgentEvaluator(
        runtime=StubAgentRuntime(),  # type: ignore[arg-type]
    )

    with pytest.raises(
        ValueError,
        match="at least one case",
    ):
        evaluator.evaluate_suite(())