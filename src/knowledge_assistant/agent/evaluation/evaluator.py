from time import perf_counter

from knowledge_assistant.agent.evaluation.models import (
    AgentEvaluationCase,
    AgentEvaluationResult,
    AgentEvaluationSummary,
)
from knowledge_assistant.agent.runtime import AgentRuntime


class AgentEvaluator:
    """Evaluate end-to-end agent behavior."""

    def __init__(
        self,
        runtime: AgentRuntime,
    ) -> None:
        self._runtime = runtime

    def evaluate(
        self,
        case: AgentEvaluationCase,
    ) -> AgentEvaluationResult:
        started = perf_counter()

        response = self._runtime.run(
            query=case.query,
        )

        duration_ms = (
            perf_counter() - started
        ) * 1000

        actual_tools = tuple(
            step.tool_call.tool_name
            for step in response.steps
        )

        actual_documents = tuple(
            sorted(
                {
                    citation.source_name
                    for step in response.steps
                    for citation in (
                        step.tool_result
                        .observation
                        .citations
                    )
                }
            )
        )

        grounding = response.grounding_validation

        is_grounded = (
            grounding.is_grounded
            if grounding is not None
            else None
        )

        return AgentEvaluationResult(
            case_id=case.case_id,
            query=case.query,
            actual_tools=actual_tools,
            actual_documents=actual_documents,
            stop_reason=response.stop_reason,
            is_grounded=is_grounded,
            iteration_count=len(
                response.iterations
            ),
            duration_ms=duration_ms,
            tool_match=(
                actual_tools
                == case.expected_tools
            ),
            document_match=all(
                document in actual_documents
                for document in case.expected_documents
            ),
            stop_reason_match=(
                response.stop_reason
                == case.expected_stop_reason
            ),
            grounding_match=(
                is_grounded is True
                if case.require_grounded
                else True
            ),
        )

    def evaluate_suite(
        self,
        cases: tuple[AgentEvaluationCase, ...],
    ) -> AgentEvaluationSummary:
        if not cases:
            raise ValueError(
                "Agent evaluation requires at least one case"
            )

        results = tuple(
            self.evaluate(case)
            for case in cases
        )

        case_count = len(results)

        passed_count = sum(
            result.passed
            for result in results
        )

        failed_count = (
            case_count - passed_count
        )

        tool_accuracy = (
            sum(result.tool_match for result in results)
            / case_count
        )

        document_accuracy = (
            sum(
                result.document_match
                for result in results
            )
            / case_count
        )

        stop_reason_accuracy = (
            sum(
                result.stop_reason_match
                for result in results
            )
            / case_count
        )

        grounding_accuracy = (
            sum(
                result.grounding_match
                for result in results
            )
            / case_count
        )

        overall_accuracy = (
            passed_count / case_count
        )

        average_iterations = (
            sum(
                result.iteration_count
                for result in results
            )
            / case_count
        )

        average_duration_ms = (
            sum(
                result.duration_ms
                for result in results
            )
            / case_count
        )

        return AgentEvaluationSummary(
            case_count=case_count,
            passed_count=passed_count,
            failed_count=failed_count,
            tool_accuracy=tool_accuracy,
            document_accuracy=document_accuracy,
            stop_reason_accuracy=(
                stop_reason_accuracy
            ),
            grounding_accuracy=grounding_accuracy,
            overall_accuracy=overall_accuracy,
            average_iterations=average_iterations,
            average_duration_ms=average_duration_ms,
            results=results,
        )