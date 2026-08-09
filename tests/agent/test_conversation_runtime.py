from typing import Any
import pytest

from knowledge_assistant.agent.models import (
    AgentContext,
    AgentObservation,
    AgentStep,
    AgentToolResult,
    FinalAnswerDecision,
    PlannerDecision,
)
from knowledge_assistant.llm.planner import AgentPlanner
from knowledge_assistant.agent.registry import AgentToolRegistry
from knowledge_assistant.agent.runtime import AgentRuntime
from knowledge_assistant.agent.tools.base import AgentTool
from knowledge_assistant.agent.tools.specifications import (
    ToolSpecification,
)
from knowledge_assistant.conversation import ConversationHistory
from knowledge_assistant.llm.grounding_validator import GroundingValidator
from knowledge_assistant.llm.models import (
    GroundingValidationResult,
)
from knowledge_assistant.llm.synthesizer import (
    AgentResponseSynthesizer,
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

class ConversationCapturingPlanner(AgentPlanner):
    def __init__(self) -> None:
        self.received_context: AgentContext | None = None

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
        self.received_context = context

        raise RuntimeError(
            "Stop after capturing planner context"
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
        return "Synthesized answer."


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
            ),
        )

def test_runtime_passes_conversation_history_to_planner() -> None:
    history = ConversationHistory()

    history.append_user(
        "What is BM25?"
    )
    history.append_assistant(
        "BM25 is a lexical retrieval algorithm."
    )

    planner = ConversationCapturingPlanner()

    runtime = AgentRuntime(
        planner=planner,
        tool_registry=AgentToolRegistry(
            tools=[StubTool()]
        ),
        response_synthesizer=StubSynthesizer(),
        grounding_validator=StubGroundingValidator(),
    )

    with pytest.raises(
        RuntimeError,
        match="Stop after capturing planner context",
    ):
        runtime.run(
            query="How is it different from vector search?",
            history=history,
        )

    assert planner.received_context is not None

    context = planner.received_context

    assert context.query == (
        "How is it different from vector search?"
    )

    assert len(context.conversation) == 2

    assert (
        context.conversation[0].content
        == "What is BM25?"
    )

    assert (
        context.conversation[1].content
        == "BM25 is a lexical retrieval algorithm."
    )

def test_runtime_records_completed_turn_in_history() -> None:
    history = ConversationHistory()

    runtime = AgentRuntime(
        planner=StubPlanner(
            FinalAnswerDecision(
                decision_type="final_answer",
                answer="Hello.",
            )
        ),
        tool_registry=AgentToolRegistry(
            tools=[StubTool()]
        ),
        response_synthesizer=StubSynthesizer(),
        grounding_validator=StubGroundingValidator(),
    )

    runtime.run(
        query="Say hello",
        history=history,
    )

    assert len(history.messages) == 2
    assert history.messages[0].content == "Say hello"
    assert history.messages[1].content == "Hello."