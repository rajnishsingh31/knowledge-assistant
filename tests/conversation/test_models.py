import pytest

from knowledge_assistant.conversation.models import (
    ConversationMessage,
    ConversationRole,
)


def test_conversation_message_stores_role_and_content() -> None:
    message = ConversationMessage(
        role=ConversationRole.USER,
        content="What is BM25?",
    )

    assert message.role == ConversationRole.USER
    assert message.content == "What is BM25?"


def test_conversation_message_rejects_empty_content() -> None:
    with pytest.raises(
        ValueError,
        match="cannot be empty",
    ):
        ConversationMessage(
            role=ConversationRole.USER,
            content="   ",
        )