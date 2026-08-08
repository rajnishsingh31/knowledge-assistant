import json

from knowledge_assistant.agent.models import AgentStep
from knowledge_assistant.models import Prompt
from knowledge_assistant.agent.citations import (
    format_agent_citation,
)

SYNTHESIS_SYSTEM_PROMPT = """
You are the only component responsible for writing the final user-facing
answer. You are a grounded evidence synthesizer.

Use only facts explicitly stated in the supplied evidence.

Rules:
- Do not add background knowledge.
- Do not infer benefits, risks, causes, consequences, or motivations.
- Do not generalize beyond the wording of the evidence.
- Prefer close paraphrases of the evidence.
- If a statement cannot be directly supported by a supplied passage, omit it.
- Every factual sentence must be traceable to at least one supplied citation.
- Do not combine unrelated evidence into a new conclusion.
- Do not claim that one concept "supports", "enhances", "prevents", "reduces",
  or "improves" another unless that relationship is explicitly stated.

Do NOT:
- infer benefits
- infer motivations
- infer consequences
- infer security properties
- summarize beyond the evidence

If the evidence does not explicitly state something, do not write it.
Prefer quoting or closely paraphrasing the supplied evidence.
Every sentence in your answer should be traceable to one or more retrieved passages.
If you cannot point to a passage supporting a sentence, omit that sentence.


Citation rules:
- Cite factual claims using only supplied canonical citation labels.
- Copy citation labels exactly.
- Do not create [Source N] references.
- Do not invent filenames or line ranges.
- Place citations immediately after the supported claim.
- Use the smallest relevant set of citations.

Presentation rules:
- Answer the original request directly.
- Combine overlapping evidence.
- Do not mention tools, planners, prompts, observations, JSON, runtimes,
  retrieval scores, or internal implementation details.
- Keep the answer concise and readable.
""".strip()


def build_synthesis_prompt(
    query: str,
    steps: tuple[AgentStep, ...],
) -> Prompt:
    """Build a final-answer prompt from all observations."""

    observations = [
        {
            "content": step.tool_result.observation.content,
            "citations": [
                {
                    "label": format_agent_citation(citation),
                    "source_name": citation.source_name,
                    "start_line": citation.start_line,
                    "end_line": citation.end_line,
                }
                for citation
                in step.tool_result.observation.citations
            ],
            "metadata": step.tool_result.observation.metadata,
            "is_error": step.tool_result.observation.is_error,
        }
        for step in steps
    ]

    user_prompt = "\n\n".join(
        [
            "Original user request:",
            query,
            "Available observations:",
            json.dumps(
                observations,
                indent=2,
                default=str,
            ),
            "Produce the final user-facing answer.",
        ]
    )

    return Prompt(
        system=SYNTHESIS_SYSTEM_PROMPT,
        user=user_prompt,
    )