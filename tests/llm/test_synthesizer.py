from knowledge_assistant.agent.models import (
    AgentToolCall,
    AgentToolResult,
    AgentStep,
    AgentObservation,
    AgentCitation,
)
from knowledge_assistant.llm.synthesizer import (
    LLMAgentResponseSynthesizer,
)
from knowledge_assistant.models import Prompt
from knowledge_assistant.llm import LLMProvider


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

    step = AgentStep(
        step_number=1,
        tool_call=AgentToolCall(
            tool_name="get_index_stats",
            arguments={},
        ),
        tool_result=AgentToolResult(
            tool_name="get_index_stats",
            observation=AgentObservation(
                content=(
                    '{"document_count": 12, '
                    '"chunk_count": 46}'
                ),
                metadata={
                    "document_count": 12,
                    "chunk_count": 46,
                },
            ),
        ),
    )

    answer = synthesizer.synthesize(
        query="How many documents are indexed?",
        steps=(step,),
    )

    assert answer == (
            "There are currently 12 documents "
            "and 46 chunks indexed."
    )

class CitationCapturingLLMProvider(LLMProvider):
    def __init__(self) -> None:
        self.received_prompt: Prompt | None = None

    @property
    def provider_name(self) -> str:
        return "stub"

    @property
    def model_name(self) -> str:
        return "stub-model"

    def generate(self, prompt: Prompt) -> str:
        self.received_prompt = prompt

        return (
            "Least privilege grants only the permissions required "
            "for assigned tasks "
            "[cloud-security.docx, lines 7-14]."
        )


def test_synthesizer_includes_canonical_citations_in_prompt() -> None:
    llm_provider = CitationCapturingLLMProvider()

    synthesizer = LLMAgentResponseSynthesizer(
        llm_provider=llm_provider,
    )

    step = AgentStep(
        step_number=1,
        tool_call=AgentToolCall(
            tool_name="answer_from_documents",
            arguments={
                "query": "What is least privilege?",
            },
        ),
        tool_result=AgentToolResult(
            tool_name="answer_from_documents",
            observation=AgentObservation(
                content=(
                    "Least privilege grants users, applications, "
                    "and services only the permissions required "
                    "for their assigned tasks."
                ),
                citations=(
                    AgentCitation(
                        source_name="cloud-security.docx",
                        start_line=7,
                        end_line=14,
                    ),
                ),
            ),
        ),
    )

    answer = synthesizer.synthesize(
        query="What is least privilege?",
        steps=(step,),
    )

    assert llm_provider.received_prompt is not None

    assert (
        "[cloud-security.docx, lines 7-14]"
        in llm_provider.received_prompt.user
    )

    assert (
        '"source_name": "cloud-security.docx"'
        in llm_provider.received_prompt.user
    )

    assert (
        '"start_line": 7'
        in llm_provider.received_prompt.user
    )

    assert (
        '"end_line": 14'
        in llm_provider.received_prompt.user
    )

    assert answer == (
        "Least privilege grants only the permissions required "
        "for assigned tasks "
        "[cloud-security.docx, lines 7-14]."
    )

class CitationOmittingLLMProvider(LLMProvider):
    @property
    def provider_name(self) -> str:
        return "stub"

    @property
    def model_name(self) -> str:
        return "stub-model"

    def generate(self, prompt: Prompt) -> str:
        return (
            "Least privilege grants only the permissions "
            "required for assigned tasks."
        )


def test_synthesizer_appends_citation_when_model_omits_it() -> None:
    synthesizer = LLMAgentResponseSynthesizer(
        llm_provider=CitationOmittingLLMProvider(),
    )

    step = AgentStep(
        step_number=1,
        tool_call=AgentToolCall(
            tool_name="answer_from_documents",
            arguments={
                "query": "What is least privilege?",
            },
        ),
        tool_result=AgentToolResult(
            tool_name="answer_from_documents",
            observation=AgentObservation(
                content=(
                    "Least privilege grants only the permissions "
                    "required for assigned tasks."
                ),
                citations=(
                    AgentCitation(
                        source_name="cloud-security.docx",
                        start_line=7,
                        end_line=14,
                    ),
                ),
            ),
        ),
    )

    answer = synthesizer.synthesize(
        query="What is least privilege?",
        steps=(step,),
    )

    assert answer == (
        "Least privilege grants only the permissions "
        "required for assigned tasks.\n\n"
        "Sources: "
        "[cloud-security.docx, lines 7-14]"
    )