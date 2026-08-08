from dataclasses import dataclass, field
from typing import Any, Literal
from knowledge_assistant.llm.models import GroundingClaimResult


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
    """Planner decision to finish the execution."""

    decision_type: Literal["final_answer"]
    answer: str

@dataclass(frozen=True)
class GroundingValidationTrace:
    is_grounded: bool
    claims: tuple[GroundingClaimResult, ...]

    @property
    def unsupported_claims(
        self,
    ) -> tuple[GroundingClaimResult, ...]:
        return tuple(
            claim
            for claim in self.claims
            if not claim.supported
        )

    @property
    def supported_claims(
        self,
    ) -> tuple[GroundingClaimResult, ...]:
        return tuple(
            claim
            for claim in self.claims
            if claim.supported
        )

PlannerDecision = ToolCallDecision | FinalAnswerDecision

@dataclass(frozen=True)
class AgentCitation:
    """One source reference returned by an agent tool."""

    source_name: str
    start_line: int | None = None
    end_line: int | None = None


@dataclass(frozen=True)
class AgentObservation:
    """Structured information produced by a tool."""

    content: str
    citations: tuple[AgentCitation, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)
    is_error: bool = False


@dataclass(frozen=True)
class AgentToolResult:
    """Result returned by an agent tool."""

    tool_name: AgentToolName
    observation: AgentObservation


@dataclass(frozen=True)
class AgentStep:
    """One completed tool-use step."""

    step_number: int
    tool_call: AgentToolCall
    tool_result: AgentToolResult


@dataclass(frozen=True)
class AgentContext:
    """Accumulated state for one agent execution."""

    query: str
    steps: tuple[AgentStep, ...]

    @property
    def next_step_number(self) -> int:
        return len(self.steps) + 1


@dataclass(frozen=True)
class AgentIteration:
    """One planner decision and its optional tool observation."""

    iteration_number: int
    decision: PlannerDecision
    tool_result: AgentToolResult | None = None




@dataclass(frozen=True)
class AgentResponse:
    """Final response returned by the agent runtime."""

    query: str
    answer: str
    steps: tuple[AgentStep, ...]
    iterations: tuple[AgentIteration, ...]
    provider_name: str
    model_name: str
    stop_reason: str
    grounding_validation: (
        GroundingValidationTrace | None
    ) = None



