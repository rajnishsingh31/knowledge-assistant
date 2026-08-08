from dataclasses import dataclass
from enum import Enum


class ConversationRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"


@dataclass(frozen=True)
class ConversationMessage:
    """One message in a conversation."""

    role: ConversationRole
    content: str

    def __post_init__(self) -> None:
        if not self.content.strip():
            raise ValueError(
                "Conversation message cannot be empty"
            )