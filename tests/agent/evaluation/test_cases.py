from knowledge_assistant.agent.evaluation.cases import (
    DEFAULT_AGENT_EVALUATION_CASES,
)


def test_default_agent_evaluation_cases_are_unique() -> None:
    case_ids = [
        case.case_id
        for case in DEFAULT_AGENT_EVALUATION_CASES
    ]

    assert len(case_ids) == len(set(case_ids))

def test_default_agent_evaluation_cases_are_not_empty() -> None:
    assert DEFAULT_AGENT_EVALUATION_CASES