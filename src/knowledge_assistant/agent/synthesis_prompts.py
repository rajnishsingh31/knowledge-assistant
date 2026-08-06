from knowledge_assistant.agent.models import (
    AgentToolCall,
    AgentToolResult,
)
from knowledge_assistant.models import Prompt


SYNTHESIS_SYSTEM_PROMPT = """
You are the final-response synthesizer for a local knowledge assistant.

Use only the supplied tool observation.

Rules:
- Answer the original user request directly.
- Do not add facts that are absent from the tool observation.
- If the observation contains document citations, include them naturally in
  your answer.
- Do not mention tool names, planners, prompts, runtimes, observations,
  JSON, or internal processing steps.
- Present the information as a direct answer to the user.
- Do not mention internal planner or runtime implementation details.
- If the observation contains an error, explain it clearly.
- Keep the response concise and readable.
""".strip()


def build_synthesis_prompt(
    query: str,
    tool_call: AgentToolCall,
    tool_result: AgentToolResult,
) -> Prompt:
    """Build the final-answer synthesis prompt."""

    user_prompt = "\n\n".join(
        [
            "Original user request:",
            query,
            "Executed tool:",
            tool_call.tool_name,
            "Tool arguments:",
            str(tool_call.arguments),
            "Tool observation:",
            tool_result.content,
            "Produce the final user-facing answer.",
        ]
    )

    return Prompt(
        system=SYNTHESIS_SYSTEM_PROMPT,
        user=user_prompt,
    )