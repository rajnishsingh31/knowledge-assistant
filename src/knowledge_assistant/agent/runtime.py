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
    GroundingValidationTrace,
)
from knowledge_assistant.llm.planner import AgentPlanner
from knowledge_assistant.agent.policy import (
    enforce_grounded_tool_policy,
    enforce_answer_evidence_policy,
)
from knowledge_assistant.agent.registry import (
    AgentToolRegistry,
)
from knowledge_assistant.llm.synthesizer import (
    AgentResponseSynthesizer,
)
from knowledge_assistant.agent.evidence import (
    select_synthesis_steps,
)
from knowledge_assistant.llm.grounding_validator import (
    GroundingValidator,
)
from knowledge_assistant.conversation import (
    ConversationHistory,
)


logger = logging.getLogger(__name__)


class AgentRuntime:
    """Coordinate controlled multi-step agent execution."""

    def __init__(
        self,
        planner: AgentPlanner,
        tool_registry: AgentToolRegistry,
        response_synthesizer: AgentResponseSynthesizer,
        grounding_validator: GroundingValidator,
        max_iterations: int = 3,
    ) -> None:
        if max_iterations <= 0:
            raise ValueError(
                "max_iterations must be greater than zero"
            )

        self._planner = planner
        self._tool_registry = tool_registry
        self._response_synthesizer = response_synthesizer
        self._grounding_validator = grounding_validator
        self._max_iterations = max_iterations

    def _record_conversation(
        self,
        history: ConversationHistory | None,
        query: str,
        answer: str,
    ) -> None:
        if history is None:
            return

        history.append_user(query)
        history.append_assistant(answer)

    def run(
        self,
        query: str,
        history: ConversationHistory | None = None,
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
            conversation=(
                history.messages
                if history is not None
                else ()
            ),
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

            focused_decision  = enforce_answer_evidence_policy(
                context=context,
                decision=decision,
            )

            if focused_decision != decision:
                if isinstance(
                    focused_decision,
                    ToolCallDecision,
                ):
                    logger.warning(
                        "answer_evidence_policy_override "
                        "iteration=%d replacement_tool=%s",
                        iteration_number,
                        focused_decision.tool_call.tool_name,
                    )

            decision = focused_decision


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

               
                # Direct conversational response with no tool evidence.
                if not context.steps:
                    self._record_conversation(
                        history=history,
                        query=normalized_query,
                        answer=decision.answer,
                    )

                    return AgentResponse(
                        query=normalized_query,
                        answer=decision.answer,
                        steps=(),
                        iterations=tuple(iterations),
                        provider_name=self._planner.provider_name,
                        model_name=self._planner.model_name,
                        stop_reason="final_answer",
                    )

                # Once evidence exists, only the synthesizer writes the answer.
                return self._finish_from_observations(
                    context=context,
                    iterations=tuple(iterations),
                    stop_reason="final_answer",
                    history=history,
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
                    history=history,
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
                conversation=context.conversation,
            )

        return self._finish_from_observations(
            context=context,
            iterations=tuple(iterations),
            stop_reason="max_iterations",
            history=history,
        )

    def _finish_from_observations(
        self,
        context: AgentContext,
        iterations: tuple[AgentIteration, ...],
        stop_reason: str,
        history: ConversationHistory | None = None,
    ) -> AgentResponse:
        if not context.steps:
            raise RuntimeError(
                "Agent cannot synthesize without observations"
            )

        # Retrieve best evidence (answer_from_documents) if available, else return all.
        synthesis_steps = select_synthesis_steps(
            context.steps
        )


        final_answer = self._response_synthesizer.synthesize(
            query=context.query,
            steps=synthesis_steps,
        )

        grounding_result = (
            self._grounding_validator.validate(
                answer=final_answer,
                steps=synthesis_steps,
            )
        )

        if grounding_result.is_grounded:
            logger.debug(
                "grounding_validation_passed"
            )
        else:
            logger.warning(
                "grounding_validation_failed "
                "unsupported_claims=%d",
                len(
                    grounding_result.unsupported_claims
                ),
            )

            for claim in (
                grounding_result.unsupported_claims
            ):
                logger.warning(
                    "unsupported_claim sentence=%r reason=%r",
                    claim.sentence,
                    claim.reason,
                )

        grounding_trace = GroundingValidationTrace(
            is_grounded=grounding_result.is_grounded,
            claims=grounding_result.claims,
        )

        self._record_conversation(
            history=history,
            query=context.query,
            answer=final_answer,
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
            grounding_validation=grounding_trace,
        )