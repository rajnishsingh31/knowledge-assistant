from knowledge_assistant.llm import LLMProvider
from knowledge_assistant.models import (
    GeneratedAnswer,
    GenerationTrace,
    RetrievedContext,
)
from knowledge_assistant.prompt_builder import PromptBuilder
from knowledge_assistant.retrieval import Retriever


class AnswerService:
    """Retrieve evidence and generate grounded answers."""

    def __init__(
        self,
        retriever: Retriever,
        prompt_builder: PromptBuilder,
        llm_provider: LLMProvider,
    ) -> None:
        self._retriever = retriever
        self._prompt_builder = prompt_builder
        self._llm_provider = llm_provider

    def generate_trace(
        self,
        query: str,
        retrieval_limit: int = 3,
    ) -> GenerationTrace:
        """Run retrieval and generation while retaining trace data."""

        results = self._retriever.search(
            query=query,
            limit=retrieval_limit,
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
        retrieval_limit: int = 3,
    ) -> GeneratedAnswer:
        """Return only the generated answer."""

        trace = self.generate_trace(
            query=query,
            retrieval_limit=retrieval_limit,
        )

        return trace.generated_answer