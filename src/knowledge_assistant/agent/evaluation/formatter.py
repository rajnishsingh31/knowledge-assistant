from knowledge_assistant.agent.evaluation.models import (
    AgentEvaluationSummary,
)


class AgentEvaluationFormatter:
    @staticmethod
    def format_summary(
        summary: AgentEvaluationSummary,
        include_details: bool = False,
    ) -> str:
        lines = [
            "Agent Evaluation",
            "-" * 60,
            f"Cases: {summary.case_count}",
            (
                "Passed: "
                f"{summary.passed_count}"
            ),
            (
                "Failed: "
                f"{summary.failed_count}"
            ),
            (
                "Overall accuracy: "
                f"{summary.overall_accuracy:.1%}"
            ),
            (
                "Tool accuracy: "
                f"{summary.tool_accuracy:.1%}"
            ),
            (
                "Document accuracy: "
                f"{summary.document_accuracy:.1%}"
            ),
            (
                "Stop reason accuracy: "
                f"{summary.stop_reason_accuracy:.1%}"
            ),
            (
                "Grounding accuracy: "
                f"{summary.grounding_accuracy:.1%}"
            ),
            (
                "Average iterations: "
                f"{summary.average_iterations:.2f}"
            ),
            (
                "Average duration: "
                f"{summary.average_duration_ms:.2f} ms"
            ),
        ]

        if not include_details:
            return "\n".join(lines)

        lines.extend(
            [
                "",
                "Case Results",
                "-" * 60,
            ]
        )

        for result in summary.results:
            status = (
                "PASS"
                if result.passed
                else "FAIL"
            )

            lines.extend(
                [
                    (
                        f"{status} — "
                        f"{result.case_id}"
                    ),
                    f"Query: {result.query}",
                    (
                        "Tools: "
                        + ", ".join(
                            result.actual_tools
                        )
                    ),
                    (
                        "Documents: "
                        + ", ".join(
                            result.actual_documents
                        )
                    ),
                    (
                        "Grounded: "
                        f"{result.is_grounded}"
                    ),
                    (
                        "Iterations: "
                        f"{result.iteration_count}"
                    ),
                    (
                        "Duration: "
                        f"{result.duration_ms:.2f} ms"
                    ),
                    "",
                ]
            )

        return "\n".join(lines).rstrip()