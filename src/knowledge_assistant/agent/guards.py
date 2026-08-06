from knowledge_assistant.agent.models import (
    AgentContext,
    AgentToolCall,
)


def is_repeated_tool_call(
    context: AgentContext,
    tool_call: AgentToolCall,
) -> bool:
    """Return whether an identical call already occurred."""

    return any(
        step.tool_call.tool_name == tool_call.tool_name
        and step.tool_call.arguments == tool_call.arguments
        for step in context.steps
    )