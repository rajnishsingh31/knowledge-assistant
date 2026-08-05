import logging
from contextlib import asynccontextmanager
from pathlib import Path
from time import perf_counter
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException

from knowledge_assistant.api.dependencies import (
    KnowledgeApplicationDependency,
)
from knowledge_assistant.api.mappers import (
    map_ingestion_result,
    map_retrieval_filter,
    map_search_results,
    map_stats,
)
from knowledge_assistant.api.schemas import (
    AskRequest,
    AskResponse,
    HealthResponse,
    IngestRequest,
    IngestResponse,
    SearchRequest,
    SearchResponse,
    StatsResponse,
)
from knowledge_assistant.bootstrap import create_application
from knowledge_assistant.config import get_settings
from knowledge_assistant.models import StartupTimings


logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(
    app: FastAPI,
) -> AsyncIterator[None]:
    """Create expensive application dependencies once."""

    startup_started = perf_counter()

    settings_started = perf_counter()
    settings = get_settings()
    settings_loading_ms = (
        perf_counter() - settings_started
    ) * 1000

    construction_started = perf_counter()
    application = create_application(settings)
    dependency_construction_ms = (
        perf_counter() - construction_started
    ) * 1000

    total_startup_ms = (
        perf_counter() - startup_started
    ) * 1000

    application.record_startup_timings(
        StartupTimings(
            settings_loading_ms=settings_loading_ms,
            dependency_construction_ms=(
                dependency_construction_ms
            ),
            total_startup_ms=total_startup_ms,
        )
    )

    app.state.knowledge_application = application

    logger.info(
        "api_started dependency_construction_ms=%.2f "
        "total_startup_ms=%.2f",
        dependency_construction_ms,
        total_startup_ms,
    )

    yield

    logger.info("api_stopped")


app = FastAPI(
    title="Knowledge Assistant API",
    description=(
        "Local-first document ingestion, retrieval, "
        "reranking, and grounded question answering."
    ),
    version="0.9.0",
    lifespan=lifespan,
)


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["Operations"],
)
def health() -> HealthResponse:
    return HealthResponse(
        status="healthy",
        application="knowledge-assistant",
        version=app.version,
    )


@app.get(
    "/stats",
    response_model=StatsResponse,
    tags=["Operations"],
)
def stats(
    application: KnowledgeApplicationDependency,
) -> StatsResponse:
    try:
        return map_stats(application.stats())
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        raise HTTPException(
            status_code=409,
            detail=str(error),
        ) from error


@app.post(
    "/search",
    response_model=SearchResponse,
    tags=["Retrieval"],
)
def search(
    request: SearchRequest,
    application: KnowledgeApplicationDependency,
) -> SearchResponse:
    try:
        results = application.search(
            query=request.query,
            limit=request.limit,
            strategy_name=request.strategy,
            retrieval_filter=map_retrieval_filter(
                request.filters
            ),
        )

        mapped_results = map_search_results(results)

        return SearchResponse(
            query=request.query,
            result_count=len(mapped_results),
            results=mapped_results,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error


@app.post(
    "/ask",
    response_model=AskResponse,
    tags=["Generation"],
)
def ask(
    request: AskRequest,
    application: KnowledgeApplicationDependency,
) -> AskResponse:
    try:
        answer = application.ask(
            query=request.query,
            limit=request.limit,
            retrieval_filter=map_retrieval_filter(
                request.filters
            ),
        )

        return AskResponse(
            query=request.query,
            answer=answer.content,
            provider_name=answer.provider_name,
            model_name=answer.model_name,
            sources=map_search_results(
                list(answer.sources)
            ),
        )
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error


@app.post(
    "/ingest",
    response_model=IngestResponse,
    tags=["Ingestion"],
)
def ingest(
    request: IngestRequest,
    application: KnowledgeApplicationDependency,
) -> IngestResponse:
    try:
        source_path = (
            Path(request.path)
            if request.path is not None
            else None
        )

        result = application.ingest(source_path)

        return map_ingestion_result(result)
    except FileNotFoundError as error:
        raise HTTPException(
            status_code=404,
            detail=str(error),
        ) from error
    except (RuntimeError, ValueError) as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error


@app.post(
    "/rebuild",
    response_model=IngestResponse,
    tags=["Ingestion"],
)
def rebuild(
    request: IngestRequest,
    application: KnowledgeApplicationDependency,
) -> IngestResponse:
    try:
        source_path = (
            Path(request.path)
            if request.path is not None
            else None
        )

        result = application.rebuild(source_path)

        return map_ingestion_result(result)
    except FileNotFoundError as error:
        raise HTTPException(
            status_code=404,
            detail=str(error),
        ) from error
    except (RuntimeError, ValueError) as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error