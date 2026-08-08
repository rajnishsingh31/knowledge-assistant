from abc import ABC, abstractmethod
import json

from knowledge_assistant.agent.models import AgentStep
from knowledge_assistant.llm import LLMProvider
from knowledge_assistant.llm.grounding_prompts import (
    build_grounding_prompt,
)
from knowledge_assistant.llm.models import (
    GroundingValidationResult,
     GroundingClaimResult,
)
import re

def extract_candidate_sentences(
    answer: str,
) -> tuple[str, ...]:
    normalized = " ".join(
        line.strip()
        for line in answer.splitlines()
        if line.strip()
    )

    if not normalized:
        return ()

    raw_sentences = re.split(
        r"(?<=[.!?])\s+",
        normalized,
    )

    sentences: list[str] = []

    for sentence in raw_sentences:
        candidate = sentence.strip()

        if not candidate:
            continue

        # Ignore standalone numbered-list markers.
        if re.fullmatch(
            r"\d+\.",
            candidate,
        ):
            continue

        sentences.append(candidate)

    return tuple(sentences)

class GroundingValidator(ABC):
    """Validate an answer against supplied evidence."""

    @property
    @abstractmethod
    def provider_name(self) -> str:
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        ...

    @abstractmethod
    def validate(
        self,
        answer: str,
        steps: tuple[AgentStep, ...],
    ) -> GroundingValidationResult:
        ...


class LLMGroundingValidator(GroundingValidator):
    def __init__(
        self,
        llm_provider: LLMProvider,
    ) -> None:
        self._llm_provider = llm_provider

    @property
    def provider_name(self) -> str:
        return self._llm_provider.provider_name

    @property
    def model_name(self) -> str:
        return self._llm_provider.model_name

    def validate(
        self,
        answer: str,
        steps: tuple[AgentStep, ...],
    ) -> GroundingValidationResult:
        normalized_answer = answer.strip()

        if not normalized_answer:
            raise ValueError(
                "Grounding validator requires a non-empty answer"
            )

        if not steps:
            raise ValueError(
                "Grounding validator requires evidence"
            )

        candidate_claims = extract_candidate_sentences(
            normalized_answer
        )

        if not candidate_claims:
            raise ValueError(
                "Grounding validator found no candidate claims"
            )

        prompt = build_grounding_prompt(
            steps=steps,
            candidate_claims=candidate_claims,
        )

        content = self._llm_provider.generate(prompt)

        try:
            payload = json.loads(content)
        except json.JSONDecodeError as error:
            raise ValueError(
                "Grounding validator returned invalid JSON"
            ) from error

        raw_verdicts = payload.get("verdicts")

        if not isinstance(raw_verdicts, list):
            raise ValueError(
                "Grounding validator response must contain a verdicts list"
            )

        verdicts_by_index: dict[int, dict[str, object]] = {}

        for item in raw_verdicts:
            index = int(item["index"])
            if index < 0 or index >= len(
                candidate_claims
            ):
                raise ValueError(
                    f"Invalid grounding verdict index: {index}"
                )

            if index in verdicts_by_index:
                raise ValueError(
                    f"Duplicate grounding verdict index: {index}"
                )
            verdicts_by_index[index] = item

        if len(verdicts_by_index) != len(
            candidate_claims
        ):
            raise ValueError(
                "Grounding validator did not return one verdict "
                "for every candidate claim"
            )

        claims: list[GroundingClaimResult] = []

        for index, sentence in enumerate(
            candidate_claims
        ):
            verdict = verdicts_by_index.get(index)

            if verdict is None:
                raise ValueError(
                    f"Missing grounding verdict for claim {index}"
                )

            claims.append(
                GroundingClaimResult(
                    sentence=sentence,
                    supported=bool(
                        verdict["supported"]
                    ),
                    reason=str(
                        verdict["reason"]
                    ),
                )
            )

        claim_results = tuple(claims)

        return GroundingValidationResult(
            is_grounded=all(
                claim.supported
                for claim in claim_results
            ),
            claims=claim_results,
        )