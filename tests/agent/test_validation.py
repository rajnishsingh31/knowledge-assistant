import pytest

from knowledge_assistant.agent.models import (
    FinalAnswerDecision,
    ToolCallDecision,
)
from knowledge_assistant.agent.tools.specifications import (
    ToolParameter,
    ToolSpecification,
)
from knowledge_assistant.agent.validation import (
    parse_planner_decision,
)


def create_specifications() -> tuple[
    ToolSpecification,
    ...
]:
    return (
        ToolSpecification(
            name="search_documents",
            description="Search documents",
            parameters=(
                ToolParameter(
                    name="query",
                    description="Search query",
                    type_name="string",
                    required=True,
                ),
                ToolParameter(
                    name="limit",
                    description="Maximum results",
                    type_name="integer",
                    required=False,
                ),
            ),
        ),
    )


def test_parses_tool_call_decision() -> None:
    decision = parse_planner_decision(
        content="""
        {
          "decision_type": "call_tool",
          "tool_name": "search_documents",
          "arguments": {
            "query": "What is BM25?",
            "limit": 3
          }
        }
        """,
        specifications=create_specifications(),
    )

    assert isinstance(decision, ToolCallDecision)
    assert (
        decision.tool_call.tool_name
        == "search_documents"
    )
    assert decision.tool_call.arguments["limit"] == 3


def test_parses_final_answer_decision() -> None:
    decision = parse_planner_decision(
        content="""
        {
          "decision_type": "final_answer",
          "answer": "Hello."
        }
        """,
        specifications=create_specifications(),
    )

    assert isinstance(
        decision,
        FinalAnswerDecision,
    )
    assert decision.answer == "Hello."


def test_rejects_unknown_tool() -> None:
    with pytest.raises(
        ValueError,
        match="unknown tool",
    ):
        parse_planner_decision(
            content="""
            {
              "decision_type": "call_tool",
              "tool_name": "delete_everything",
              "arguments": {}
            }
            """,
            specifications=create_specifications(),
        )


def test_rejects_missing_required_argument() -> None:
    with pytest.raises(
        ValueError,
        match="Missing required arguments",
    ):
        parse_planner_decision(
            content="""
            {
              "decision_type": "call_tool",
              "tool_name": "search_documents",
              "arguments": {
                "limit": 3
              }
            }
            """,
            specifications=create_specifications(),
        )


def test_rejects_unknown_argument() -> None:
    with pytest.raises(
        ValueError,
        match="Unknown arguments",
    ):
        parse_planner_decision(
            content="""
            {
              "decision_type": "call_tool",
              "tool_name": "search_documents",
              "arguments": {
                "query": "BM25",
                "delete": true
              }
            }
            """,
            specifications=create_specifications(),
        )