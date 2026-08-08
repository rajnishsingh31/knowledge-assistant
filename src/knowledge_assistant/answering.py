import logging
from time import perf_counter

from knowledge_assistant.llm import LLMProvider
from knowledge_assistant.models import (
    GeneratedAnswer,
    GenerationTrace,
    PipelineTimings,
    RetrievedContext,
    IngestionTimings,
    RetrievalFilter,
    RetrievalTrace,
)
from knowledge_assistant.llm.prompt_builder import PromptBuilder
from knowledge_assistant.reranking import Reranker
from knowledge_assistant.retrieval import Retriever
from dataclasses import dataclass


logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class IngestionResult:
    document_count: int
    chunk_count: int
    embedding_count: int
    embedding_model: str
    table_name: str
    timings: IngestionTimings

class AnswerService:
    def __init__(
        self,
        retriever: Retriever,
        reranker: Reranker,
        prompt_builder: PromptBuilder,
        llm_provider: LLMProvider,
        retrieval_limit: int,
        final_limit: int,
    ) -> None:
        self._retriever = retriever
        self._reranker = reranker
        self._prompt_builder = prompt_builder
        self._llm_provider = llm_provider
        self._retrieval_limit = retrieval_limit
        self._final_limit = final_limit

    def retrieve_context(
        self,
        query: str,
        retrieval_limit: int | None = None,
        final_limit: int | None = None,
        retrieval_filter: RetrievalFilter | None = None,
    ) -> RetrievalTrace:
        """Retrieve and rerank grounded evidence without generating."""

        normalized_query = query.strip()

        if not normalized_query:
            raise ValueError("Query cannot be empty")

        effective_final_limit = (
            final_limit or self._final_limit
        )

        effective_retrieval_limit = (
            retrieval_limit
            or self._retrieval_limit
            if self._reranker is not None
            else effective_final_limit
        )

        retrieval_started = perf_counter()
        candidates = self._retriever.search(
            query=query,
            limit=effective_retrieval_limit,
            retrieval_filter=retrieval_filter,
        )
                
        retrieval_ms = (
            perf_counter() - retrieval_started
        ) * 1000
        
        logger.debug(
            "retrieval_completed query=%r candidates=%d duration_ms=%.2f",
            query,
            len(candidates),
            retrieval_ms,
        )
        
        reranking_ms = 0.0
        reranking_started = perf_counter()
        if self._reranker is not None:
            results = self._reranker.rerank(
            query=normalized_query,
            results=candidates,
            limit=effective_final_limit,
            )

            reranking_ms = (
                        perf_counter() - reranking_started
                    ) * 1000
                    
            logger.debug(
                "reranking_completed candidates=%d results=%d "
                "model=%s duration_ms=%.2f",
                len(candidates),
                len(results),
                self._reranker.model_name,
                reranking_ms,
            )  
        else:
            results = candidates[:effective_final_limit]

            

        return RetrievalTrace(
            context=RetrievedContext(
                query=normalized_query,
                results=tuple(results),
            ),
            retrieval_ms=retrieval_ms,
            reranking_ms=reranking_ms,
        )  
        


    def generate_trace(
        self,
        query: str,
        retrieval_limit: int | None = None,
        final_limit: int | None = None,
        retrieval_filter: RetrievalFilter | None = None,
    ) -> GenerationTrace:
        

        total_started = perf_counter()

        retrieved_trace = self.retrieve_context(
            query=query,
            retrieval_limit=retrieval_limit,
            final_limit=final_limit,
            retrieval_filter=retrieval_filter,
        )

        prompt_started = perf_counter()
        prompt = self._prompt_builder.build(retrieved_trace.context)
        prompt_building_ms = (
            perf_counter() - prompt_started
        ) * 1000

        generation_started = perf_counter()
        content = self._llm_provider.generate(prompt)
        generation_ms = (
            perf_counter() - generation_started
        ) * 1000

        logger.debug(
            "generation_completed provider=%s model=%s "
            "duration_ms=%.2f",
            self._llm_provider.provider_name,
            self._llm_provider.model_name,
            generation_ms,
        )

        generated_answer = GeneratedAnswer(
            content=content,
            provider_name=self._llm_provider.provider_name,
            model_name=self._llm_provider.model_name,
            sources=tuple(retrieved_trace.context.results),
        )

        total_ms = (perf_counter() - total_started) * 1000

        return GenerationTrace(
            retrieved_context=retrieved_trace.context,
            prompt=prompt,
            generated_answer=generated_answer,
            timings=PipelineTimings(
                retrieval_ms=retrieved_trace.retrieval_ms,
                reranking_ms=retrieved_trace.reranking_ms,
                prompt_building_ms=prompt_building_ms,
                generation_ms=generation_ms,
                total_ms=total_ms,
            ),
        )


    def answer(
        self,
        query: str,
        retrieval_limit: int | None = None,
        final_limit: int | None = None,
        retrieval_filter: RetrievalFilter | None = None,
    ) -> GeneratedAnswer:
        """Return only the generated answer."""

        return self.generate_trace(
            query=query,
            retrieval_limit=retrieval_limit,
            final_limit=final_limit,
            retrieval_filter=retrieval_filter,
        ).generated_answer
