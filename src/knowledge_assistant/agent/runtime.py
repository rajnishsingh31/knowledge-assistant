import logging

from knowledge_assistant.agent.guards import (
    is_repeated_tool_call,
)
from knowledge_assistant.agent.models import (
    AgentContext,
    AgentResponse,
    AgentStep,
    FinalAnswerDecision,
    ToolCallDecision,
    AgentIteration,
)
from knowledge_assistant.agent.planner import AgentPlanner
from knowledge_assistant.agent.policy import (
    enforce_grounded_tool_policy,
)
from knowledge_assistant.agent.registry import (
    AgentToolRegistry,
)
from knowledge_assistant.agent.synthesizer import (
    AgentResponseSynthesizer,
)


logger = logging.getLogger(__name__)


class AgentRuntime:
    """Coordinate controlled multi-step agent execution."""

    def __init__(
        self,
        planner: AgentPlanner,
        tool_registry: AgentToolRegistry,
        response_synthesizer: AgentResponseSynthesizer,
        max_iterations: int = 3,
    ) -> None:
        if max_iterations <= 0:
            raise ValueError(
                "max_iterations must be greater than zero"
            )

        self._planner = planner
        self._tool_registry = tool_registry
        self._response_synthesizer = response_synthesizer
        self._max_iterations = max_iterations

    def run(
        self,
        query: str,
    ) -> AgentResponse:
        normalized_query = query.strip()

        if not normalized_query:
            raise ValueError(
                "Agent query cannot be empty"
            )

        iterations: list[AgentIteration] = []

        context = AgentContext(
            query=normalized_query,
            steps=(),
        )

        for iteration_number in range(
            1,
            self._max_iterations + 1,
        ):
            logger.debug(
                "agent_iteration_started iteration=%d",
                iteration_number,
            )

            original_decision = self._planner.plan(
                context=context,
                specifications=(
                    self._tool_registry.specifications
                ),
            )

            decision = enforce_grounded_tool_policy(
                context=context,
                decision=original_decision,
            )

            if decision != original_decision:
                logger.warning(
                    "planner_policy_override "
                    "iteration=%d original_decision=%s",
                    iteration_number,
                    original_decision.decision_type,
                )

            if isinstance(
                decision,
                FinalAnswerDecision,
            ):

                iterations.append(
                    AgentIteration(
                        iteration_number=iteration_number,
                        decision=decision,
                        tool_result=None,
                        )
                )
                
                return AgentResponse(
                    query=normalized_query,
                    answer=decision.answer,
                    steps=context.steps,
                    iterations=tuple(iterations),
                    provider_name=(
                        self._planner.provider_name
                    ),
                    model_name=self._planner.model_name,
                    stop_reason="final_answer",
                )

            if not isinstance(
                decision,
                ToolCallDecision,
            ):
                raise RuntimeError(
                    "Planner returned an unsupported decision"
                )

            tool_call = decision.tool_call

            if is_repeated_tool_call(
                context=context,
                tool_call=tool_call,
            ):
                iterations.append(
                    AgentIteration(
                        iteration_number=iteration_number,
                        decision=decision,
                        tool_result=None,
                    )
                )

                logger.warning(
                    "agent_repeated_tool_call tool=%s",
                    tool_call.tool_name,
                )

                return self._finish_from_observations(
                    context=context,
                    iterations=tuple(iterations),
                    stop_reason="repeated_tool_call",
                )

            tool_result = self._tool_registry.execute(
                tool_name=tool_call.tool_name,
                arguments=tool_call.arguments,
            )

            iterations.append(
                AgentIteration(
                    iteration_number=iteration_number,
                    decision=decision,
                    tool_result=tool_result,
                )
            )

            step = AgentStep(
                step_number=context.next_step_number,
                tool_call=tool_call,
                tool_result=tool_result,
            )

            context = AgentContext(
                query=context.query,
                steps=context.steps + (step,),
            )

        return self._finish_from_observations(
            context=context,
            iterations=tuple(iterations),
            stop_reason="max_iterations",
        )

    def _finish_from_observations(
        self,
        context: AgentContext,
        iterations: tuple[AgentIteration, ...],
        stop_reason: str,
    ) -> AgentResponse:
        if not context.steps:
            raise RuntimeError(
                "Agent cannot synthesize without observations"
            )

        latest_step = context.steps[-1]

        final_answer = (
            self._response_synthesizer.synthesize(
                query=context.query,
                tool_call=latest_step.tool_call,
                tool_result=latest_step.tool_result,
            )
        )

        return AgentResponse(
            query=context.query,
            answer=final_answer,
            steps=context.steps,
            iterations=iterations,
            provider_name=(
                self._response_synthesizer.provider_name
            ),
            model_name=(
                self._response_synthesizer.model_name
            ),
            stop_reason=stop_reason,
        )