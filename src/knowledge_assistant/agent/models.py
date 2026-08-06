from dataclasses import dataclass
from typing import Any, Literal


AgentToolName = Literal[
    "search_documents",
    "answer_from_documents",
    "inspect_index",
    "get_index_stats",
]


@dataclass(frozen=True)
class AgentToolCall:

    """A validated tool call selected by the planner."""

    tool_name: AgentToolName
    arguments: dict[str, Any]

@dataclass(frozen=True)
class ToolCallDecision:
    """Planner decision to execute a tool."""

    decision_type: Literal["call_tool"]
    tool_call: AgentToolCall


@dataclass(frozen=True)
class FinalAnswerDecision:
    """Planner decision to answer without another tool call."""

    decision_type: Literal["final_answer"]
    answer: str


PlannerDecision = ToolCallDecision | FinalAnswerDecision


@dataclass(frozen=True)
class AgentToolResult:
    """Result returned by an agent tool."""

    tool_name: AgentToolName
    content: str


@dataclass(frozen=True)
class AgentStep:
    """One executed tool-use step."""

    step_number: int
    tool_call: AgentToolCall
    tool_result: AgentToolResult


@dataclass(frozen=True)
class AgentResponse:
    """Final response returned by the agent runtime."""

    query: str
    answer: str
    steps: tuple[AgentStep, ...]
    provider_name: str
    model_name: str