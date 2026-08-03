from knowledge_assistant.llm import LLMProvider
from knowledge_assistant.models import (
    GeneratedAnswer,
    GenerationTrace,
    RetrievedContext,
)
from knowledge_assistant.prompt_builder import PromptBuilder
from knowledge_assistant.retrieval import Retriever
from knowledge_assistant.reranking import Reranker


class AnswerService:
    """Retrieve evidence and generate grounded answers."""

    def __init__(
        self,
        retriever: Retriever,
        prompt_builder: PromptBuilder,
        llm_provider: LLMProvider,
        reranker: Reranker,
        retrieval_limit: int,
        final_limit: int,
    ) -> None:
        self._retriever = retriever
        self._prompt_builder = prompt_builder
        self._llm_provider = llm_provider
        self._reranker = reranker
        self._retrieval_limit = retrieval_limit
        self._final_limit = final_limit

    def generate_trace(
        self,
        query: str,
        retrieval_limit: int | None = None,
        final_limit: int | None = None,
    ) -> GenerationTrace:
        """Retrieve, rerank, and generate an answer."""

        effective_retrieval_limit = (
            retrieval_limit or self._retrieval_limit
        )

        effective_final_limit = (
            final_limit or self._final_limit
        )

        candidates = self._retriever.search(
            query=query,
            limit=effective_retrieval_limit,
        )

        results = self._reranker.rerank(
            query=query,
            results=candidates,
            limit=effective_final_limit,
       )

        retrieved_context = RetrievedContext(
            query=query,
            results=tuple(results),
        )

        prompt = self._prompt_builder.build(retrieved_context)
        content = self._llm_provider.generate(prompt)

        generated_answer = GeneratedAnswer(
            content=content,
            provider_name=self._llm_provider.provider_name,
            model_name=self._llm_provider.model_name,
            sources=tuple(results),
        )

        return GenerationTrace(
            retrieved_context=retrieved_context,
            prompt=prompt,
            generated_answer=generated_answer,
        )

    def answer(
        self,
        query: str,
        retrieval_limit: int | None = None,
        final_limit: int | None = None,
    ) -> GeneratedAnswer:
        """Return only the generated answer."""

        trace = self.generate_trace(
            query=query,
            retrieval_limit=retrieval_limit,
            final_limit=final_limit,
        )

        return trace.generated_answer