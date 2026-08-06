import json

from knowledge_assistant.agent.models import AgentContext
from knowledge_assistant.agent.tools.specifications import (
    ToolSpecification,
)
from knowledge_assistant.models import Prompt


PLANNER_SYSTEM_PROMPT = """
You are a tool-selection planner for a local knowledge assistant.

You are not the final answering model. Your responsibility is to decide
the next action.

Choose exactly one action:

1. Call one available tool.
2. Return a final answer when the accumulated observations are sufficient.

Rules:
- Use get_index_stats for index counts, table information, embedding
  models, or dimensions.
- Use inspect_index only for raw index inspection requests.
- Use search_documents when relevant passages or sources are needed.
- Use answer_from_documents for a grounded answer from indexed documents.
- Review all previous tool calls and observations before selecting another
  action.
- Do not repeat an identical tool call unless the previous observation
  contained an error.
- Do not invent facts or tool results.
- Use final_answer only when prior observations contain enough information.
- Return exactly one valid JSON object.
- Do not wrap JSON in Markdown.
""".strip()


RESPONSE_INSTRUCTIONS = """
Return exactly one JSON object.

To call a tool:

{
  "decision_type": "call_tool",
  "tool_name": "search_documents",
  "arguments": {
    "query": "search query",
    "limit": 5
  }
}

To finish:

{
  "decision_type": "final_answer",
  "answer": "Final answer based only on the observations"
}

Important:
- Return one object only.
- decision_type must be a top-level property.
- Never include text outside the JSON object.
""".strip()


def build_planner_prompt(
    context: AgentContext,
    specifications: tuple[ToolSpecification, ...],
) -> Prompt:
    """Build the next-action planning prompt."""

    if not context.query.strip():
        raise ValueError("Planner query cannot be empty")

    if not specifications:
        raise ValueError(
            "Planner requires at least one tool specification"
        )

    tools_payload = [
        {
            "name": specification.name,
            "description": specification.description,
            "parameters": [
                {
                    "name": parameter.name,
                    "description": parameter.description,
                    "type": parameter.type_name,
                    "required": parameter.required,
                }
                for parameter in specification.parameters
            ],
        }
        for specification in specifications
    ]

    previous_steps = [
        {
            "step_number": step.step_number,
            "tool_name": step.tool_call.tool_name,
            "arguments": step.tool_call.arguments,
            "observation": step.tool_result.content,
        }
        for step in context.steps
    ]

    user_prompt = "\n\n".join(
        [
            "Available tools:",
            json.dumps(tools_payload, indent=2),
            "Original user request:",
            context.query,
            "Previous tool steps:",
            (
                json.dumps(previous_steps, indent=2)
                if previous_steps
                else "No tools have been called yet."
            ),
            "Output instructions:",
            RESPONSE_INSTRUCTIONS,
            "Choose the next action.",
        ]
    )

    return Prompt(
        system=PLANNER_SYSTEM_PROMPT,
        user=user_prompt,
    )