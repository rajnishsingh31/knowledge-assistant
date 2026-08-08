from sentence_transformers import CrossEncoder

from knowledge_assistant.agent.models import AgentStep
from knowledge_assistant.llm.grounding_validator import (
    GroundingValidator,
    extract_candidate_sentences,
)
from knowledge_assistant.llm.models import (
    GroundingClaimResult,
    GroundingValidationResult,
)
import json
from typing import Any
import re


def normalize_claim(text: str) -> str:
        """Normalize synthesized text before NLI evaluation."""

        normalized = text.strip()

        # Remove Markdown bold markers.
        normalized = normalized.replace("**", "")

        # Remove canonical citations, e.g.
        # [cloud-security.docx, lines 7-14]
        normalized = re.sub(
            r"\[[^\]]+\]",
            "",
            normalized,
        )

        # Remove numbered-list prefixes such as "1. "
        normalized = re.sub(
            r"^\d+\.\s*",
            "",
            normalized,
        )

        # Remove common generated section labels.
        normalized = re.sub(
            (
                r"^(Definition|Examples|Practical Application)"
                r":\s*"
            ),
            "",
            normalized,
            flags=re.IGNORECASE,
        )

        # Collapse whitespace introduced by removals.
        normalized = " ".join(
            normalized.split()
        )

        normalized = re.sub(
            r"\s+([.,!?;:])",
            r"\1",
            normalized,
        )

        return normalized

class NLIGroundingValidator(GroundingValidator):
    """Validate synthesized claims using natural language inference."""

    def __init__(
        self,
        model_name: str = "cross-encoder/nli-deberta-v3-small",
        entailment_threshold: float = 0.70,
        model: Any | None = None
    ) -> None:
        if not 0.0 <= entailment_threshold <= 1.0:
            raise ValueError(
                "entailment_threshold must be between 0 and 1"
            )

        self._model_name = model_name
        self._entailment_threshold = entailment_threshold

        self._model = (
            model
            if model is not None
            else CrossEncoder(model_name)
        )

    @property
    def provider_name(self) -> str:
        return "sentence-transformers"

    @property
    def model_name(self) -> str:
        return self._model_name

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

        claims = extract_candidate_sentences(
            normalized_answer
        )

        if not claims:
            raise ValueError(
                "Grounding validator found no candidate claims"
            )

        evidence_chunks = self._extract_evidence(
            steps
        )

        if not evidence_chunks:
            raise ValueError(
                "Grounding validator found no evidence content"
            )

        claim_results = tuple(
            self._validate_claim(
                claim=claim,
                evidence_chunks=evidence_chunks,
            )
            for claim in claims
        )

        return GroundingValidationResult(
            is_grounded=all(
                claim.supported
                for claim in claim_results
            ),
            claims=claim_results,
        )

    def _validate_claim(
        self,
        claim: str,
        evidence_chunks: tuple[str, ...],
    ) -> GroundingClaimResult:
        normalized_claim = normalize_claim(
            claim
        )

        pairs = [
            (evidence, normalized_claim)
            for evidence in evidence_chunks
        ]

        scores = self._model.predict(
            pairs,
            apply_softmax=True,
        )

        # Model label order:
        # 0 = contradiction
        # 1 = entailment
        # 2 = neutral
        best_entailment_score = max(
            float(score[1])
            for score in scores
        )

        supported = (
            best_entailment_score
            >= self._entailment_threshold
        )

        reason = (
            "Entailed by supplied evidence "
            f"(score={best_entailment_score:.3f})."
            if supported
            else
            "No supplied evidence explicitly entails this claim "
            f"(best entailment score={best_entailment_score:.3f})."
        )

        return GroundingClaimResult(
            sentence=claim,
            supported=supported,
            reason=reason,
        )

    @staticmethod
    def _extract_evidence(
        steps: tuple[AgentStep, ...],
    ) -> tuple[str, ...]:
        evidence: list[str] = []

        for step in steps:
            content = (
                step.tool_result.observation.content.strip()
            )

            if not content:
                continue

            try:
                payload: Any = json.loads(content)
            except json.JSONDecodeError:
                evidence.append(content)
                continue

            if isinstance(payload, list):
                for item in payload:
                    if not isinstance(item, dict):
                        continue

                    chunk_content = item.get(
                        "content"
                    )

                    if isinstance(
                        chunk_content,
                        str,
                    ) and chunk_content.strip():
                        evidence.append(
                            chunk_content.strip()
                        )
            else:
                evidence.append(content)

        return tuple(evidence)

    