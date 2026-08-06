from knowledge_assistant.agent.models import (
    AgentResponse,
    FinalAnswerDecision,
    ToolCallDecision,
)


class AgentConsoleFormatter:
    """Format agent execution results for the CLI."""

    @staticmethod
    def format_response(
        response: AgentResponse,
        include_trace: bool = False,
    ) -> str:
        lines = [
            response.answer,
            "",
            (
                "Planned by: "
                f"{response.provider_name}/"
                f"{response.model_name}"
            ),
            "",
            (
                "Stopped reason: "
                f"{response.stop_reason}"
            ),
        ]

        if include_trace:
            lines.extend(
                [
                    "",
                    "AGENT TRACE",
                    "-" * 60,
                ]
            )

            for iteration in response.iterations:
                lines.append(
                    f"Iteration: {iteration.iteration_number}"
                )

                decision = iteration.decision

                if isinstance(decision, ToolCallDecision):
                    lines.extend(
                        [
                            "Decision: call_tool",
                            f"Tool: {decision.tool_call.tool_name}",
                            (
                                "Arguments: "
                                f"{decision.tool_call.arguments}"
                            ),
                        ]
                    )

                    if iteration.tool_result is not None:
                        lines.extend(
                            [
                                "Observation:",
                                iteration.tool_result.content,
                            ]
                        )
                    else:
                        lines.append(
                            "Observation: Tool was not executed."
                        )

                elif isinstance(
                    decision,
                    FinalAnswerDecision,
                ):
                    lines.extend(
                        [
                            "Decision: final_answer",
                            f"Answer: {decision.answer}",
                        ]
                    )

                lines.append("")

            
            
        return "\n".join(lines).rstrip()