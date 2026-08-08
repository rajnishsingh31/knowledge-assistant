from pathlib import Path

from knowledge_assistant.answering import AnswerService
from knowledge_assistant.models import (
    Chunk,
    Prompt,
    RetrievalFilter,
    SearchResult,
)
from knowledge_assistant.reranking import IdentityReranker
from knowledge_assistant.llm import LLMProvider

class StubRetriever:
    def search(
        self,
        query: str,
        limit: int,
        retrieval_filter: RetrievalFilter | None = None,
    ) -> list[SearchResult]:
        self.received_filter = retrieval_filter
        return [
            SearchResult(
                chunk=Chunk(
                    chunk_id="chunk-1",
                    document_id="document-1",
                    source_path=Path("sample.md"),
                    content="BM25 is a lexical retrieval algorithm.",
                    start_line=1,
                    end_line=2,
                    document_hash="document-hash-1",
                    chunk_hash=f"chunk-hash-1",
                ),
                retrieval_method="hybrid",
                score=1.0,
            )
        ]


class StubPromptBuilder:
    def build(self, context: object) -> Prompt:
        return Prompt(
            system="system",
            user="user",
        )


class StubLLMProvider:
    @property
    def provider_name(self) -> str:
        return "stub"

    @property
    def model_name(self) -> str:
        return "stub-model"

    def generate(self, prompt: Prompt) -> str:
        return "Generated answer"


def test_answer_service_passes_filter_to_retriever() -> None:
    retriever = StubRetriever()

    service = AnswerService(
        retriever=retriever,  # type: ignore[arg-type]
        reranker=IdentityReranker(),
        prompt_builder=StubPromptBuilder(),  # type: ignore[arg-type]
        llm_provider=StubLLMProvider(),  # type: ignore[arg-type]
        retrieval_limit=10,
        final_limit=3,
    )

    retrieval_filter = RetrievalFilter(
        extensions=(".md",),
    )

    service.generate_trace(
        query="What is BM25?",
        retrieval_filter=retrieval_filter,
    )

    assert retriever.received_filter == retrieval_filter

def test_generate_trace_includes_non_negative_timings() -> None:
    service = AnswerService(
        retriever=StubRetriever(),  # type: ignore[arg-type]
        reranker=IdentityReranker(),
        prompt_builder=StubPromptBuilder(),  # type: ignore[arg-type]
        llm_provider=StubLLMProvider(),  # type: ignore[arg-type]
        retrieval_limit=10,
        final_limit=3,
    )

    trace = service.generate_trace("What is BM25?")

    assert trace.timings.retrieval_ms >= 0
    assert trace.timings.reranking_ms >= 0
    assert trace.timings.prompt_building_ms >= 0
    assert trace.timings.generation_ms >= 0
    assert trace.timings.total_ms >= 0
    assert trace.generated_answer.content == "Generated answer"

class FailingLLMProvider(LLMProvider):
    @property
    def provider_name(self) -> str:
        return "failing"

    @property
    def model_name(self) -> str:
        return "failing-model"

    def generate(self, prompt: Prompt) -> str:
        raise AssertionError(
            "LLM generation must not occur during retrieve_context()"
        )

def test_retrieve_context_does_not_generate_answer() -> None:
    service = AnswerService(
        retriever=StubRetriever(),  # type: ignore[arg-type]
        reranker=None, # type: ignore[arg-type]
        prompt_builder=StubPromptBuilder(),  # type: ignore[arg-type]
        llm_provider=FailingLLMProvider(),
        retrieval_limit=3,
        final_limit=5,
    )

    retrieval = service.retrieve_context(
        query="What is BM25?",
    )

    assert retrieval.context.query == "What is BM25?"
    assert retrieval.context.results
    assert len(retrieval.context.results) == 1
    assert retrieval.retrieval_ms >= 0
    assert retrieval.reranking_ms == 0.0