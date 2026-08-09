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
- Use previous conversation only to resolve references, follow-up questions,
  and omitted context in the current user request.
- The current user request is always the task to execute.
- Previous assistant responses are conversation context, not authoritative
  document evidence. Use tools when factual grounding is required.
- Use get_index_stats for index counts, table information, embedding
  models, or dimensions.
- Use inspect_index only for raw index inspection requests.
- Use search_documents when relevant passages or sources are needed.
- Use answer_from_documents for direct factual questions that need the
  strongest retrieved and reranked evidence.
- answer_from_documents returns evidence, not a finished answer.
- After sufficient evidence has been collected, return final_answer using
  only the observations.
- Review all previous tool calls and observations before selecting another
  action.
- Do not repeat an identical tool call unless the previous observation
  contained an error.
- Do not invent facts or tool results.
- When observations are sufficient, select final_answer.
- The final-answer text is only a completion signal after tools have run.
- Do not compose a detailed factual answer; another component will
  synthesize it from the observations.
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
  "answer": "Evidence is sufficient."
}

Important:
- Return one object only.
- decision_type must be a top-level property.
- Never include text outside the JSON object.
""".strip()


def _build_conversation_context(
    context: AgentContext,
) -> str:
    if not context.conversation:
        return "No previous conversation."

    return "\n".join(
        (
            f"{message.role.value}: "
            f"{message.content}"
        )
        for message in context.conversation
    )

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
            "observation": {
                "content": step.tool_result.observation.content,
                "citations": [
                    {
                        "source_name": citation.source_name,
                        "start_line": citation.start_line,
                        "end_line": citation.end_line,
                    }
                    for citation
                    in step.tool_result.observation.citations
                ],
                "metadata": step.tool_result.observation.metadata,
                "is_error": step.tool_result.observation.is_error,
            },
        }
        for step in context.steps
    ]

    conversation_context = _build_conversation_context(
        context
    )

    user_prompt = "\n\n".join(
        [
            "Available tools:",
            json.dumps(tools_payload, indent=2),

            "Previous conversation:",
            conversation_context,

            "Current user request:",
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