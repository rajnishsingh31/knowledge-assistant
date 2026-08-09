from knowledge_assistant.agent.evaluation.models import (
    AgentEvaluationCase,
)


DEFAULT_AGENT_EVALUATION_CASES = (
    AgentEvaluationCase(
        case_id="index-stats",
        query="How many documents are currently indexed?",
        expected_tools=(
            "get_index_stats",
        ),
        expected_documents=(),
        expected_stop_reason="final_answer",
        require_grounded=True,
    ),
    AgentEvaluationCase(
        case_id="least-privilege",
        query=(
            "Find evidence about least privilege "
            "and explain it."
        ),
        expected_tools=(
            "search_documents",
            "answer_from_documents",
        ),
        expected_documents=(
            "cloud-security.docx",
        ),
        expected_stop_reason="final_answer",
        require_grounded=True,
    ),
    AgentEvaluationCase(
        case_id="bm25",
        query="What is BM25?",
        expected_tools=(
            "search_documents",
            "answer_from_documents",
        ),
        expected_documents=(
            "bm25.md",
        ),
        expected_stop_reason="final_answer",
        require_grounded=True,
    ),
)