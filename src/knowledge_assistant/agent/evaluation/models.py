from dataclasses import dataclass


@dataclass(frozen=True)
class AgentEvaluationCase:
    case_id: str
    query: str
    expected_tools: tuple[str, ...] = ()
    expected_documents: tuple[str, ...] = ()
    expected_stop_reason: str = "final_answer"
    require_grounded: bool = True


@dataclass(frozen=True)
class AgentEvaluationResult:
    case_id: str
    query: str
    actual_tools: tuple[str, ...]
    actual_documents: tuple[str, ...]
    stop_reason: str
    is_grounded: bool | None
    iteration_count: int
    duration_ms: float

    tool_match: bool
    document_match: bool
    stop_reason_match: bool
    grounding_match: bool

    @property
    def passed(self) -> bool:
        return (
            self.tool_match
            and self.document_match
            and self.stop_reason_match
            and self.grounding_match
        )

@dataclass(frozen=True)
class AgentEvaluationSummary:
    case_count: int
    passed_count: int
    failed_count: int

    tool_accuracy: float
    document_accuracy: float
    stop_reason_accuracy: float
    grounding_accuracy: float
    overall_accuracy: float

    average_iterations: float
    average_duration_ms: float

    results: tuple[AgentEvaluationResult, ...]