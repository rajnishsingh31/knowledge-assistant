import pytest

from knowledge_assistant.conversation.memory import (
    ConversationHistory,
)
from knowledge_assistant.conversation.models import (
    ConversationRole,
)


def test_history_appends_user_and_assistant_messages() -> None:
    history = ConversationHistory(
        max_messages=4
    )

    history.append_user(
        "What is BM25?"
    )
    history.append_assistant(
        "BM25 is a lexical retrieval algorithm."
    )

    assert len(history.messages) == 2
    assert (
        history.messages[0].role
        == ConversationRole.USER
    )
    assert (
        history.messages[1].role
        == ConversationRole.ASSISTANT
    )


def test_history_keeps_only_recent_messages() -> None:
    history = ConversationHistory(
        max_messages=2
    )

    history.append_user("Message 1")
    history.append_assistant("Message 2")
    history.append_user("Message 3")

    assert len(history.messages) == 2
    assert (
        history.messages[0].content
        == "Message 2"
    )
    assert (
        history.messages[1].content
        == "Message 3"
    )


def test_recent_returns_requested_messages() -> None:
    history = ConversationHistory(
        max_messages=5
    )

    history.append_user("Message 1")
    history.append_assistant("Message 2")
    history.append_user("Message 3")

    recent = history.recent(2)

    assert len(recent) == 2
    assert recent[0].content == "Message 2"
    assert recent[1].content == "Message 3"


def test_history_can_be_cleared() -> None:
    history = ConversationHistory()

    history.append_user(
        "What is BM25?"
    )

    history.clear()

    assert history.messages == ()


def test_history_rejects_invalid_max_messages() -> None:
    with pytest.raises(
        ValueError,
        match="max_messages must be greater than zero",
    ):
        ConversationHistory(
            max_messages=0
        )


def test_recent_rejects_invalid_limit() -> None:
    history = ConversationHistory()

    with pytest.raises(
        ValueError,
        match="limit must be greater than zero",
    ):
        history.recent(0)