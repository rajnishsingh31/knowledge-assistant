from dataclasses import dataclass

@dataclass(frozen=True)
class Prompt:
    """A provider-neutral prompt sent to an LLM."""

    system: str
    user: str

