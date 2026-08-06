from knowledge_assistant.agent.models import (
    ToolCallDecision,
)
from knowledge_assistant.agent.planner import (
    LLMAgentPlanner,
)
from knowledge_assistant.agent.tools.specifications import (
    ToolParameter,
    ToolSpecification,
)
from knowledge_assistant.models import Prompt

from knowledge_assistant.agent.models import AgentContext


class StubLLMProvider:
    @property
    def provider_name(self) -> str:
        return "stub"

    @property
    def model_name(self) -> str:
        return "stub-model"

    def generate(self, prompt: Prompt) -> str:
        return """
        {
          "decision_type": "call_tool",
          "tool_name": "search_documents",
          "arguments": {
            "query": "What is BM25?",
            "limit": 3
          }
        }
        """


def test_planner_returns_validated_tool_call() -> None:
    planner = LLMAgentPlanner(
        llm_provider=StubLLMProvider(),  # type: ignore[arg-type]
    )

    specifications = (
        ToolSpecification(
            name="search_documents",
            description="Search documents",
            parameters=(
                ToolParameter(
                    name="query",
                    description="Query",
                    type_name="string",
                    required=True,
                ),
                ToolParameter(
                    name="limit",
                    description="Limit",
                    type_name="integer",
                    required=False,
                ),
            ),
        ),
    )

    decision = planner.plan(
        context=AgentContext(
            query="Find information about BM25",
            steps=(),
        ),
        specifications=specifications,
    )

    assert isinstance(decision, ToolCallDecision)
    assert decision.tool_call.tool_name == "search_documents"