from typing import Any

from knowledge_assistant.agent.models import (
    AgentToolCall,
    AgentToolResult,
    FinalAnswerDecision,
    PlannerDecision,
    ToolCallDecision,
)
from knowledge_assistant.agent.planner import (
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
from knowledge_assistant.agent.synthesizer import (
    AgentResponseSynthesizer,
)

from knowledge_assistant.agent.models import AgentContext

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
        tool_call: AgentToolCall,
        tool_result: AgentToolResult,
    ) -> str:
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
            content='{"document_count": 10}',
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
                answer=(
                    "There are 10 indexed documents."
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
        max_iterations=3,
    )

    response = runtime.run(
        "How many documents are indexed?"
    )

    assert len(response.steps) == 1
    assert response.answer == (
        "There are 10 indexed documents."
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