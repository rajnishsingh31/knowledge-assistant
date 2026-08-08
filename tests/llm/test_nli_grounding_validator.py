from knowledge_assistant.llm.nli_grounding_validator import (
    NLIGroundingValidator,
)
from knowledge_assistant.agent.models import (
    AgentCitation,
    AgentObservation,
    AgentStep,
    AgentToolCall,
    AgentToolResult,
)
from knowledge_assistant.llm.nli_grounding_validator import (
    normalize_claim,
    extract_candidate_sentences,
)

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
                    "Least privilege grants users only the permissions required for their assigned tasks."
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

def create_multi_evidence_step() -> AgentStep:
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
                content="""
[
  {
    "content": "Least privilege grants users only the permissions required for their assigned tasks."
  },
  {
    "content": "Permissions should be reviewed regularly and removed when no longer required."
  },
  {
    "content": "Managed identities allow cloud applications to authenticate without storing passwords."
  }
]
""",
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

class StubNLIModel:
    def __init__(
        self,
        scores: list[list[float]],
    ) -> None:
        self._scores = scores

    def predict(
        self,
        pairs: list[tuple[str, str]],
        apply_softmax: bool = False,
    ) -> list[list[float]]:
        assert apply_softmax is True
        return self._scores

def test_nli_validator_marks_entailed_claim_supported() -> None:
    validator = NLIGroundingValidator(
        entailment_threshold=0.70,
        model=StubNLIModel(
            scores=[
                [0.01, 0.96, 0.03],
            ]
        ),
    )

    result = validator.validate(
        answer=(
            "Least privilege grants only required permissions."
        ),
        steps=(create_evidence_step(),),
    )

    assert result.is_grounded is True
    assert len(result.claims) == 1
    assert result.claims[0].supported is True

def test_nli_validator_rejects_unentailed_claim() -> None:
    validator = NLIGroundingValidator(
        entailment_threshold=0.70,
        model=StubNLIModel(
            scores=[
                [0.02, 0.15, 0.83],
            ]
        ),
    )

    result = validator.validate(
        answer=(
            "Least privilege prevents malicious activity."
        ),
        steps=(create_evidence_step(),),
    )

    assert result.is_grounded is False
    assert result.claims[0].supported is False

def test_nli_validator_uses_best_entailment_score() -> None:
    validator = NLIGroundingValidator(
        entailment_threshold=0.70,
        model=StubNLIModel(
            scores=[
                [0.02, 0.10, 0.88],
                [0.01, 0.91, 0.08],
                [0.03, 0.20, 0.77],
            ]
        ),
    )

    result = validator.validate(
        answer=(
            "Least privilege grants only required permissions."
        ),
        steps=(create_multi_evidence_step(),),
    )

    assert result.is_grounded is True

def test_extract_evidence_returns_individual_chunks() -> None:
    validator = NLIGroundingValidator(
        model=StubNLIModel(scores=[]),
    )

    evidence = validator._extract_evidence(
        (create_evidence_step(),)
    )

    assert evidence == (
        "Least privilege grants users only the permissions required for their assigned tasks.",
    )

def test_normalize_claim_removes_markdown_and_citation() -> None:
    claim = (
        "1. **Definition**: Least privilege grants only "
        "required permissions "
        "[cloud-security.docx, lines 7-14]."
    )

    assert normalize_claim(claim) == (
        "Least privilege grants only required permissions."
    )

def test_extract_candidate_sentences_ignores_list_numbers() -> None:
    answer = (
        "1. First factual statement. "
        "2. Second factual statement. "
        "3. Third factual statement."
    )

    result = extract_candidate_sentences(
        answer
    )

    assert "2." not in result
    assert "3." not in result