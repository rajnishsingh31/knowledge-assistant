import json

from knowledge_assistant.agent.models import AgentStep
from knowledge_assistant.llm.models import Prompt


GROUNDING_SYSTEM_PROMPT = """
You are a strict grounding auditor.

For each supplied candidate claim, determine whether every factual
relationship in that claim is explicitly supported by the supplied evidence.

A claim may be generally true and still be unsupported.

Rules:
- Use only the supplied evidence.
- Do not use outside knowledge.
- Do not reward plausibility.
- Do not infer benefits, risks, causes, consequences, motivations,
  conceptual alignment, prevention, reduction, or improvement.
- If any factual part of a claim is unsupported, mark the whole claim
  unsupported.
- When uncertain, mark unsupported.

Example:

Evidence:
"Managed identities allow applications to authenticate without storing
passwords."

Claim:
"Managed identities align with least privilege by avoiding credential
exposure."

Verdict:
UNSUPPORTED.

Reason:
The evidence states how managed identities authenticate, but does not
explicitly state that they align with least privilege or that avoiding
credential exposure is the reason.

Example:

Evidence:
"Secret Rotation | Limit credential lifetime."

Claim:
"Rotating credentials limits unnecessary permissions."

Verdict:
UNSUPPORTED.

Reason:
Credential lifetime and permission scope are different claims.

Return valid JSON only.
Do not include Markdown or explanatory text outside the JSON object.
""".strip()


def build_grounding_prompt(
    steps: tuple[AgentStep, ...],
    candidate_claims: tuple[str, ...],
) -> Prompt:
    evidence = [
        {
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
        }
        for step in steps
    ]

    claims = [
        {
            "index": index,
            "sentence": sentence,
        }
        for index, sentence in enumerate(
            candidate_claims
        )
    ]

    user_prompt = "\n\n".join(
        [
            "Evidence:",
            json.dumps(
                evidence,
                indent=2,
                default=str,
            ),
            "Candidate claims:",
            json.dumps(
                claims,
                indent=2,
            ),
            """
Return exactly:

{
  "verdicts": [
    {
      "index": 0,
      "supported": true,
      "reason": "Reason based only on supplied evidence"
    }
  ]
}

Rules:
- Return one verdict for every candidate claim.
- Use the exact provided index.
- Do not omit claims.
- Do not add claims.
- Judge only explicit support from the evidence.
""".strip(),
        ]
    )

    return Prompt(
        system=GROUNDING_SYSTEM_PROMPT,
        user=user_prompt,
    )