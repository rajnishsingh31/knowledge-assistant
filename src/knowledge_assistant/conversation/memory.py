from knowledge_assistant.conversation.models import (
    ConversationMessage,
    ConversationRole,
)


class ConversationHistory:
    """Bounded in-memory conversation history."""

    def __init__(
        self,
        max_messages: int = 8,
    ) -> None:
        if max_messages <= 0:
            raise ValueError(
                "max_messages must be greater than zero"
            )

        self._max_messages = max_messages
        self._messages: list[
            ConversationMessage
        ] = []

    @property
    def messages(
        self,
    ) -> tuple[ConversationMessage, ...]:
        return tuple(self._messages)

    def append(
        self,
        message: ConversationMessage,
    ) -> None:
        self._messages.append(message)

        if len(self._messages) > self._max_messages:
            self._messages = self._messages[
                -self._max_messages:
            ]

    def append_user(
        self,
        content: str,
    ) -> None:
        self.append(
            ConversationMessage(
                role=ConversationRole.USER,
                content=content,
            )
        )

    def append_assistant(
        self,
        content: str,
    ) -> None:
        self.append(
            ConversationMessage(
                role=ConversationRole.ASSISTANT,
                content=content,
            )
        )

    def recent(
        self,
        limit: int,
    ) -> tuple[ConversationMessage, ...]:
        if limit <= 0:
            raise ValueError(
                "limit must be greater than zero"
            )

        return tuple(
            self._messages[-limit:]
        )

    def clear(self) -> None:
        self._messages.clear()