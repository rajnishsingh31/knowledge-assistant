from knowledge_assistant.agent.models import (
    AgentToolCall,
    AgentToolResult,
)
from knowledge_assistant.agent.synthesizer import (
    LLMAgentResponseSynthesizer,
)
from knowledge_assistant.models import Prompt


class StubLLMProvider:
    @property
    def provider_name(self) -> str:
        return "stub"

    @property
    def model_name(self) -> str:
        return "stub-model"

    def generate(self, prompt: Prompt) -> str:
        assert "document_count" in prompt.user
        return (
            "There are currently 12 documents "
            "and 46 chunks indexed."
        )


def test_synthesizer_creates_readable_answer() -> None:
    synthesizer = LLMAgentResponseSynthesizer(
        llm_provider=StubLLMProvider(),  # type: ignore[arg-type]
    )

    answer = synthesizer.synthesize(
        query="How many documents are indexed?",
        tool_call=AgentToolCall(
            tool_name="get_index_stats",
            arguments={},
        ),
        tool_result=AgentToolResult(
            tool_name="get_index_stats",
            content=(
                '{"document_count": 12, '
                '"chunk_count": 46}'
            ),
        ),
    )

    assert answer == (
        "There are currently 12 documents "
        "and 46 chunks indexed."
    )