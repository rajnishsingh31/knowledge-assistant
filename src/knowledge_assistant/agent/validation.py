import json
from typing import Any, cast

from knowledge_assistant.agent.models import (
    AgentToolCall,
    AgentToolName,
    FinalAnswerDecision,
    PlannerDecision,
    ToolCallDecision,
)
from knowledge_assistant.agent.tools.specifications import (
    ToolSpecification,
)


def parse_planner_decision(
    content: str,
    specifications: tuple[ToolSpecification, ...],
) -> PlannerDecision:
    """Parse and validate JSON returned by the planner model."""

    normalized_content = content.strip()

    if not normalized_content:
        raise ValueError("Planner returned an empty response")

    try:
        payload: Any = json.loads(normalized_content)
    except json.JSONDecodeError as error:
        raise ValueError(
            "Planner response is not valid JSON"
        ) from error

    if not isinstance(payload, dict):
        raise ValueError(
            "Planner response must be a JSON object"
        )

    decision_type = payload.get("decision_type")

    if decision_type == "final_answer":
        return _parse_final_answer(payload)

    if decision_type == "call_tool":
        return _parse_tool_call(
            payload=payload,
            specifications=specifications,
        )

    raise ValueError(
        "Planner decision_type must be "
        "'call_tool' or 'final_answer'"
    )


def _parse_final_answer(
    payload: dict[str, Any],
) -> FinalAnswerDecision:
    answer = str(payload.get("answer", "")).strip()

    if not answer:
        raise ValueError(
            "final_answer decision requires a non-empty answer"
        )

    return FinalAnswerDecision(
        decision_type="final_answer",
        answer=answer,
    )


def _parse_tool_call(
    payload: dict[str, Any],
    specifications: tuple[ToolSpecification, ...],
) -> ToolCallDecision:
    raw_tool_name = str(
        payload.get("tool_name", "")
    ).strip()

    specification_by_name = {
        specification.name: specification
        for specification in specifications
    }

    specification = specification_by_name.get(
        raw_tool_name  # type: ignore[arg-type]
    )

    if specification is None:
        raise ValueError(
            f"Planner selected unknown tool: {raw_tool_name}"
        )

    arguments = payload.get("arguments", {})

    if not isinstance(arguments, dict):
        raise ValueError(
            "Tool arguments must be a JSON object"
        )

    _validate_tool_arguments(
        arguments=arguments,
        specification=specification,
    )

    tool_name = cast(AgentToolName, raw_tool_name)

    return ToolCallDecision(
        decision_type="call_tool",
        tool_call=AgentToolCall(
            tool_name=tool_name,
            arguments=arguments,
        ),
    )


def _validate_tool_arguments(
    arguments: dict[str, Any],
    specification: ToolSpecification,
) -> None:
    parameters_by_name = {
        parameter.name: parameter
        for parameter in specification.parameters
    }

    unknown_arguments = (
        set(arguments) - set(parameters_by_name)
    )

    if unknown_arguments:
        unknown_values = ", ".join(
            sorted(unknown_arguments)
        )

        raise ValueError(
            f"Unknown arguments for {specification.name}: "
            f"{unknown_values}"
        )

    missing_arguments = [
        parameter.name
        for parameter in specification.parameters
        if (
            parameter.required
            and parameter.name not in arguments
        )
    ]

    if missing_arguments:
        missing_values = ", ".join(missing_arguments)

        raise ValueError(
            f"Missing required arguments for "
            f"{specification.name}: {missing_values}"
        )

    for name, value in arguments.items():
        parameter = parameters_by_name[name]

        if parameter.type_name == "string":
            if not isinstance(value, str):
                raise ValueError(
                    f"Argument '{name}' must be a string"
                )

        elif parameter.type_name == "integer":
            if (
                not isinstance(value, int)
                or isinstance(value, bool)
            ):
                raise ValueError(
                    f"Argument '{name}' must be an integer"
                )