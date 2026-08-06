from knowledge_assistant.agent.models import (
    AgentResponse,
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
        ]

        if include_trace:
            lines.extend(
                [
                    "",
                    "AGENT TRACE",
                    "-" * 60,
                ]
            )

            if not response.steps:
                lines.append(
                    "No tool call was required."
                )

            for step in response.steps:
                lines.extend(
                    [
                        f"Step: {step.step_number}",
                        (
                            "Tool: "
                            f"{step.tool_call.tool_name}"
                        ),
                        (
                            "Arguments: "
                            f"{step.tool_call.arguments}"
                        ),
                        "Observation:",
                        step.tool_result.content,
                        "",
                    ]
                )

        return "\n".join(lines).rstrip()