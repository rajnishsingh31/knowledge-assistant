from dataclasses import dataclass

@dataclass(frozen=True)
class Prompt:
    """A provider-neutral prompt sent to an LLM."""

    system: str
    user: str

@dataclass(frozen=True)
class GroundingClaimResult:
    """Grounding verdict for one factual sentence."""

    sentence: str
    supported: bool
    reason: str


@dataclass(frozen=True)
class GroundingValidationResult:
    """Result of validating an answer against evidence."""

    is_grounded: bool
    claims: tuple[GroundingClaimResult, ...]

    @property
    def unsupported_claims(
        self,
    ) -> tuple[GroundingClaimResult, ...]:
        return tuple(
            claim
            for claim in self.claims
            if not claim.supported
        )

