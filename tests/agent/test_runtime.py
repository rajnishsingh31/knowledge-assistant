from typing import Any

from knowledge_assistant.agent.models import (
    AgentToolCall,
    AgentToolResult,
    FinalAnswerDecision,
    PlannerDecision,
    ToolCallDecision,
    AgentObservation,
    AgentStep,
)
from knowledge_assistant.llm.planner import (
    AgentPlanner,
)
from knowledge_assistant.agent.registry import (
    AgentToolRegistry,
)
from knowledge_assistant.agent.runtime import (
    AgentRuntime,
)
from knowledge_assistant.agent.tools.base import (
    AgentTool,
)
from knowledge_assistant.agent.tools.specifications import (
    ToolSpecification,
)
from knowledge_assistant.llm.synthesizer import (
    AgentResponseSynthesizer,
)

from knowledge_assistant.agent.models import AgentContext

from knowledge_assistant.llm.grounding_validator import (
    GroundingValidator,
)
from knowledge_assistant.llm.models import (
    GroundingValidationResult,
)
from knowledge_assistant.llm.models import (
    GroundingValidationResult,
    GroundingClaimResult,
)

@property
def is_grounded(self) -> bool:
    return all(
        claim.supported
        for claim in self.claims
    )

class StubSynthesizer(AgentResponseSynthesizer):
    @property
    def provider_name(self) -> str:
        return "stub"

    @property
    def model_name(self) -> str:
        return "stub-model"

    def synthesize(
        self,
        query: str,
        steps: tuple[AgentStep, ...],
    ) -> str:
        assert steps
        return "There are 10 indexed documents."

class StubTool(AgentTool):
    @property
    def specification(self) -> ToolSpecification:
        return ToolSpecification(
            name="get_index_stats",
            description="Return index statistics",
            parameters=(),
        )

    def execute(
        self,
        arguments: dict[str, Any],
    ) -> AgentToolResult:
        return AgentToolResult(
            tool_name="get_index_stats",
            observation=AgentObservation(
                content='{"document_count": 10}',
                metadata={
                    "document_count": 10,
                },
            ),
        )


class StubPlanner(AgentPlanner):
    def __init__(
        self,
        decision: PlannerDecision,
    ) -> None:
        self._decision = decision

    @property
    def provider_name(self) -> str:
        return "stub"

    @property
    def model_name(self) -> str:
        return "stub-model"

    def plan(
        self,
        context: AgentContext,
        specifications: tuple[
            ToolSpecification,
            ...
        ],
    ) -> PlannerDecision:
        return self._decision

class StubGroundingValidator(GroundingValidator):
    @property
    def provider_name(self) -> str:
        return "stub"

    @property
    def model_name(self) -> str:
        return "stub-model"

    def validate(
        self,
        answer: str,
        steps: tuple[AgentStep, ...],
    ) -> GroundingValidationResult:
        return GroundingValidationResult(
            is_grounded=True,
            claims=(),
        )

class FailingGroundingValidator(
    GroundingValidator
):
    @property
    def provider_name(self) -> str:
        return "stub"

    @property
    def model_name(self) -> str:
        return "stub-model"

    def validate(
        self,
        answer: str,
        steps: tuple[AgentStep, ...],
    ) -> GroundingValidationResult:
        return GroundingValidationResult(
            is_grounded=False,
            claims=(
                GroundingClaimResult(
                    sentence=(
                        "This reduces unauthorized access."
                    ),
                    supported=False,
                    reason=(
                        "The evidence does not explicitly "
                        "state this."
                    ),
                ),
            ),
        )


def test_runtime_executes_selected_tool() -> None:
    planner = StubPlanner(
        ToolCallDecision(
            decision_type="call_tool",
            tool_call=AgentToolCall(
                tool_name="get_index_stats",
                arguments={},
            ),
        )
    )

    registry = AgentToolRegistry(
        tools=[StubTool()]
    )

    runtime = AgentRuntime(
        planner=planner,
        tool_registry=registry,
        response_synthesizer=StubSynthesizer(),
        grounding_validator=StubGroundingValidator(),
        max_iterations=1,
    )

    response = runtime.run(
        "How many documents are indexed?"
    )

    assert (
        response.answer
        == "There are 10 indexed documents."
    )



def test_runtime_returns_direct_final_answer() -> None:
    planner = StubPlanner(
        FinalAnswerDecision(
            decision_type="final_answer",
            answer="Hello.",
        )
    )

    runtime = AgentRuntime(
        planner=planner,
        tool_registry=AgentToolRegistry(
            tools=[StubTool()]
        ),
        response_synthesizer=StubSynthesizer(),
        grounding_validator=StubGroundingValidator(),
    )

    response = runtime.run("Say hello")

    assert response.answer == "Hello."
    assert response.steps == ()
    assert len(response.iterations) == 1
    assert isinstance(
        response.iterations[0].decision,
        FinalAnswerDecision,
    )

class SequentialPlanner(AgentPlanner):
    def __init__(
        self,
        decisions: list[PlannerDecision],
    ) -> None:
        self._decisions = decisions
        self._index = 0

    @property
    def provider_name(self) -> str:
        return "stub"

    @property
    def model_name(self) -> str:
        return "stub-model"

    def plan(
        self,
        context: AgentContext,
        specifications: tuple[
            ToolSpecification,
            ...
        ],
    ) -> PlannerDecision:
        decision = self._decisions[self._index]
        self._index += 1
        return decision

def test_runtime_stops_at_max_iterations() -> None:
    planner = SequentialPlanner(
        decisions=[
            ToolCallDecision(
                decision_type="call_tool",
                tool_call=AgentToolCall(
                    tool_name="get_index_stats",
                    arguments={},
                ),
            ),
            ToolCallDecision(
                decision_type="call_tool",
                tool_call=AgentToolCall(
                    tool_name="get_index_stats",
                    arguments={
                        "unused": "different",
                    },
                ),
            ),
        ]
    )

    runtime = AgentRuntime(
        planner=planner,
        tool_registry=AgentToolRegistry(
            tools=[StubTool()]
        ),
        response_synthesizer=StubSynthesizer(),
        grounding_validator=StubGroundingValidator(),
        max_iterations=1,
    )

    response = runtime.run(
        "Inspect the index."
    )

    assert len(response.steps) == 1
    assert response.stop_reason == "max_iterations"
    assert len(response.iterations) == 1
    assert isinstance(
        response.iterations[0].decision,
        ToolCallDecision,
    )

def test_runtime_executes_tool_then_finishes() -> None:
    planner = SequentialPlanner(
        decisions=[
            ToolCallDecision(
                decision_type="call_tool",
                tool_call=AgentToolCall(
                    tool_name="get_index_stats",
                    arguments={},
                ),
            ),
            FinalAnswerDecision(
                decision_type="final_answer",
                answer="Planner answer must not be used.",
            ),
        ]
    )

    runtime = AgentRuntime(
        planner=planner,
        tool_registry=AgentToolRegistry(
            tools=[StubTool()]
        ),
        response_synthesizer=StubSynthesizer(),
        grounding_validator=StubGroundingValidator(),
        max_iterations=3,
    )

    response = runtime.run(
        "How many documents are indexed?"
    )

    assert len(response.steps) == 1

    # Final answer must come from the synthesizer,
    # not from the planner.
    assert response.answer == (
        "There are 10 indexed documents."
    )
    assert response.answer != (
        "Planner answer must not be used."
    )

    assert response.stop_reason == "final_answer"
    assert len(response.iterations) == 2

    assert isinstance(
        response.iterations[0].decision,
        ToolCallDecision,
    )

    assert isinstance(
        response.iterations[1].decision,
        FinalAnswerDecision,
    )

    assert response.grounding_validation is not None
    assert response.grounding_validation.is_grounded is True

def test_runtime_returns_answer_when_grounding_fails() -> None:
    planner = SequentialPlanner(
        decisions=[
            ToolCallDecision(
                decision_type="call_tool",
                tool_call=AgentToolCall(
                    tool_name="get_index_stats",
                    arguments={},
                ),
            ),
            FinalAnswerDecision(
                decision_type="final_answer",
                answer="Evidence is sufficient.",
            ),
        ]
    )

    runtime = AgentRuntime(
        planner=planner,
        tool_registry=AgentToolRegistry(
            tools=[StubTool()]
        ),
        response_synthesizer=StubSynthesizer(),
        grounding_validator=(
            FailingGroundingValidator()
        ),
        max_iterations=3,
    )

    response = runtime.run(
        "How many documents are indexed?"
    )

    assert response.answer == (
        "There are 10 indexed documents."
    )

    assert response.grounding_validation is not None

    grounding = response.grounding_validation

    assert grounding.is_grounded is False
    assert len(grounding.claims) == 1
    assert len(grounding.unsupported_claims) == 1

    claim = grounding.claims[0]

    assert claim.supported is False
    assert claim.sentence == (
        "This reduces unauthorized access."
    )