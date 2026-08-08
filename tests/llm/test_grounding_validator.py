from knowledge_assistant.agent.models import (
    AgentCitation,
    AgentObservation,
    AgentStep,
    AgentToolCall,
    AgentToolResult,
)
from knowledge_assistant.llm.base import LLMProvider
from knowledge_assistant.llm.grounding_validator import (
    LLMGroundingValidator,
    extract_candidate_sentences,
)
from knowledge_assistant.llm.models import Prompt
import pytest

def create_evidence_step() -> AgentStep:
    return AgentStep(
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
                    "to perform their assigned tasks."
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


class GroundedLLMProvider(LLMProvider):
    @property
    def provider_name(self) -> str:
        return "stub"

    @property
    def model_name(self) -> str:
        return "stub-model"

    def generate(
        self,
        prompt: Prompt,
    ) -> str:
        return """
        {
        "verdicts": [
            {
            "index": 0,
            "supported": true,
            "reason": "The evidence explicitly supports this claim."
            }
        ]
        }
        """


class UnsupportedLLMProvider(LLMProvider):
    @property
    def provider_name(self) -> str:
        return "stub"

    @property
    def model_name(self) -> str:
        return "stub-model"

    def generate(
        self,
        prompt: Prompt,
    ) -> str:
        return """
        {
        "verdicts": [
            {
            "index": 0,
            "supported": false,
            "reason": "The supplied evidence does not state this."
            }
        ]
        }
        """


def test_validator_returns_grounded_result() -> None:
    validator = LLMGroundingValidator(
        llm_provider=GroundedLLMProvider(),
    )

    result = validator.validate(
        answer=(
            "Least privilege grants users only the "
            "permissions required for their tasks."
        ),
        steps=(create_evidence_step(),),
    )

    assert result.is_grounded is True
    assert len(result.claims) == 1
    assert result.unsupported_claims == ()


def test_validator_returns_unsupported_claims() -> None:
    validator = LLMGroundingValidator(
        llm_provider=UnsupportedLLMProvider(),
    )

    result = validator.validate(
        answer="It prevents privilege escalation.",
        steps=(create_evidence_step(),),
    )

    assert result.is_grounded is False
    assert len(result.claims) == 1
    assert len(result.unsupported_claims) == 1

    claim = result.unsupported_claims[0]

    assert claim.sentence == (
        "It prevents privilege escalation."
    )
    assert claim.supported is False

class InvalidJSONLLMProvider(LLMProvider):
    @property
    def provider_name(self) -> str:
        return "stub"

    @property
    def model_name(self) -> str:
        return "stub-model"

    def generate(
        self,
        prompt: Prompt,
    ) -> str:
        return "This is not JSON."


def test_validator_rejects_invalid_json() -> None:
    validator = LLMGroundingValidator(
        llm_provider=InvalidJSONLLMProvider(),
    )

    with pytest.raises(
        ValueError,
        match="Grounding validator returned invalid JSON",
    ):
        validator.validate(
            answer="Some answer.",
            steps=(create_evidence_step(),),
        )

def test_validator_requires_evidence() -> None:
    validator = LLMGroundingValidator(
        llm_provider=GroundedLLMProvider(),
    )

    with pytest.raises(
        ValueError,
        match="requires evidence",
    ):
        validator.validate(
            answer="Some answer.",
            steps=(),
        )

def test_validator_fails_when_any_claim_is_unsupported() -> None:
    class MixedLLMProvider(LLMProvider):
        @property
        def provider_name(self) -> str:
            return "stub"

        @property
        def model_name(self) -> str:
            return "stub-model"

        def generate(
            self,
            prompt: Prompt,
        ) -> str:
            return """
            {
                "verdicts": [
                    {
                    "index": 0,
                    "supported": true,
                    "reason": "Explicitly supported."
                    },
                    {
                    "index": 1,
                    "supported": false,
                    "reason": "Not explicitly supported."
                    }
                ]
            }
            """

    validator = LLMGroundingValidator(
        llm_provider=MixedLLMProvider(),
    )

    result = validator.validate(
        answer=(
            "Least privilege limits permissions. "
            "It prevents privilege escalation."
        ),
        steps=(create_evidence_step(),),
    )

    assert result.is_grounded is False
    assert len(result.unsupported_claims) == 1

def test_extract_candidate_sentences() -> None:
    answer = (
        "Least privilege limits permissions. "
        "It reduces unauthorized access. "
        "Is this supported?"
    )

    result = extract_candidate_sentences(answer)

    assert result == (
        "Least privilege limits permissions.",
        "It reduces unauthorized access.",
        "Is this supported?",
    )

class MixedGroundingLLMProvider(LLMProvider):
    @property
    def provider_name(self) -> str:
        return "stub"

    @property
    def model_name(self) -> str:
        return "stub-model"

    def generate(
        self,
        prompt: Prompt,
    ) -> str:
        assert '"index": 0' in prompt.user
        assert '"index": 1' in prompt.user

        return """
        {
          "verdicts": [
            {
              "index": 0,
              "supported": true,
              "reason": "Explicitly supported by the evidence."
            },
            {
              "index": 1,
              "supported": false,
              "reason": "The evidence does not state this outcome."
            }
          ]
        }
        """


def test_validator_checks_each_sentence_independently() -> None:
    validator = LLMGroundingValidator(
        llm_provider=MixedGroundingLLMProvider(),
    )

    result = validator.validate(
        answer=(
            "Least privilege grants only required permissions. "
            "It reduces unauthorized access."
        ),
        steps=(create_evidence_step(),),
    )

    assert len(result.claims) == 2
    assert result.is_grounded is False

    assert result.claims[0].sentence == (
        "Least privilege grants only required permissions."
    )
    assert result.claims[0].supported is True

    assert result.claims[1].sentence == (
        "It reduces unauthorized access."
    )
    assert result.claims[1].supported is False

    assert len(result.unsupported_claims) == 1