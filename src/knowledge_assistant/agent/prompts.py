import json

from knowledge_assistant.agent.tools.specifications import (
    ToolSpecification,
)
from knowledge_assistant.models import Prompt


PLANNER_SYSTEM_PROMPT = """
You are a tool-selection planner for a local knowledge assistant.
You are not an answering model. Your primary responsibility is selecting tools.
For any factual, technical, document-related, index-related, or knowledge
question, you must call a tool.
Use final_answer only for conversational responses that require no factual
information, such as greetings or acknowledgements.

Choose exactly one action:

1. Call one available tool.
2. Return a final answer only when no tool is required.

Rules:
- Use get_index_stats for questions about document count, chunk count,
  embedding models, dimensions, table names, or index statistics.
- Use inspect_index only for requests to inspect raw indexed records.
- Use search_documents when the user wants relevant excerpts, evidence,
  or source passages.
- Use answer_from_documents for direct questions that should be answered
  from indexed documents.
- When the request asks for indexed data or document knowledge, call the
  appropriate tool before answering.
- Never guess or invent tool results.
- Never invent a tool name.
- Include only parameters defined by the selected tool.
- Return exactly one JSON object.
- Return valid JSON only.
- Do not wrap JSON in Markdown.
- Do not include explanations or text outside the JSON object.
""".strip()


TOOL_CALL_RESPONSE_EXAMPLE = """
{
  "decision_type": "call_tool",
  "tool_name": "get_index_stats",
  "arguments": {}
}
""".strip()


FINAL_ANSWER_RESPONSE_EXAMPLE = """
{
  "decision_type": "final_answer",
  "answer": "Your response"
}
""".strip()


RESPONSE_INSTRUCTIONS = f"""
Return exactly one JSON object using one of the following formats.

When a tool is required, return only:

{TOOL_CALL_RESPONSE_EXAMPLE}

When no tool is required, return only:

{FINAL_ANSWER_RESPONSE_EXAMPLE}

Important:
- Return only one object, never both.
- decision_type must be a top-level property.
- For a tool call, decision_type must be exactly "call_tool".
- For a direct answer, decision_type must be exactly "final_answer".
- Do not wrap the response inside "tool_call" or "final_answer".
- Do not use alternative property names such as "action", "type",
  or "decision".
- Never invent information that an available tool can retrieve.
- Do not include Markdown, code fences, comments, or explanatory text.
""".strip()


def build_planner_prompt(
    query: str,
    specifications: tuple[ToolSpecification, ...],
) -> Prompt:
    """Build a prompt that asks the model to select one action."""

    normalized_query = query.strip()

    if not normalized_query:
        raise ValueError(
            "Planner query cannot be empty"
        )

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

    user_prompt = "\n\n".join(
        [
            "Available tools:",
            json.dumps(
                tools_payload,
                indent=2,
            ),
            "Output instructions:",
            RESPONSE_INSTRUCTIONS,
            "User request:",
            normalized_query,
        ]
    )

    return Prompt(
        system=PLANNER_SYSTEM_PROMPT,
        user=user_prompt,
    )