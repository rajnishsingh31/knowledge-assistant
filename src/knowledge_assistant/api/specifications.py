from dataclasses import dataclass

from knowledge_assistant.agent.models import AgentToolName


@dataclass(frozen=True)
class ToolParameter:
    name: str
    description: str
    type_name: str
    required: bool


@dataclass(frozen=True)
class ToolSpecification:
    name: AgentToolName
    description: str
    parameters: tuple[ToolParameter, ...]