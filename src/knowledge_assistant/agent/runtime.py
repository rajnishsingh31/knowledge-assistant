import logging

from knowledge_assistant.agent.models import (
    AgentResponse,
    AgentStep,
    FinalAnswerDecision,
    ToolCallDecision,
)
from knowledge_assistant.agent.planner import AgentPlanner
from knowledge_assistant.agent.registry import AgentToolRegistry
from knowledge_assistant.agent.policy import (
    enforce_grounded_tool_policy,
)
from knowledge_assistant.agent.synthesizer import (
    AgentResponseSynthesizer,
)


logger = logging.getLogger(__name__)


class AgentRuntime:
    """Coordinate planning and controlled tool execution."""

    def __init__(
        self,
        planner: AgentPlanner,
        tool_registry: AgentToolRegistry,
        response_synthesizer: AgentResponseSynthesizer,
    ) -> None:
        self._planner = planner
        self._tool_registry = tool_registry
        self._response_synthesizer = response_synthesizer

    def run(
        self,
        query: str,
    ) -> AgentResponse:
        """Execute one validated planner decision."""

        normalized_query = query.strip()

        if not normalized_query:
            raise ValueError(
                "Agent query cannot be empty"
            )

        decision = self._planner.plan(
            query=normalized_query,
            specifications=(
                self._tool_registry.specifications
            ),
        )

        original_decision = decision

        decision = enforce_grounded_tool_policy(
            query=normalized_query,
            decision=decision,
        )

        if decision != original_decision:
            logger.warning(
                "planner_policy_override "
                "original_decision=%s replacement_tool=%s",
                original_decision.decision_type,
                decision.tool_call.tool_name,
            )


        if isinstance(
            decision,
            FinalAnswerDecision,
        ):
            logger.debug(
                "agent_final_answer_without_tool"
            )

            return AgentResponse(
                query=normalized_query,
                answer=decision.answer,
                steps=(),
                provider_name=(
                    self._planner.provider_name
                ),
                model_name=self._planner.model_name,
            )

        if isinstance(
            decision,
            ToolCallDecision,
        ):
            tool_call = decision.tool_call

            logger.debug(
                "agent_tool_selected tool=%s",
                tool_call.tool_name,
            )

            tool_result = self._tool_registry.execute(
                tool_name=tool_call.tool_name,
                arguments=tool_call.arguments,
            )

            step = AgentStep(
                step_number=1,
                tool_call=tool_call,
                tool_result=tool_result,
            )

            final_answer = (
            self._response_synthesizer.synthesize(
                query=normalized_query,
                tool_call=tool_call,
                tool_result=tool_result,
                )
            )

            return AgentResponse(
                query=normalized_query,
                answer=final_answer,
                steps=(step,),
                provider_name=(
                    self._planner.provider_name
                ),
                model_name=self._planner.model_name,
            )

        raise RuntimeError(
            "Planner returned an unsupported decision"
        )