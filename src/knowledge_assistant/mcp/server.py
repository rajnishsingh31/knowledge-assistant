from time import perf_counter
from typing import Any

from mcp.server import MCPServer

from knowledge_assistant.bootstrap import (
    create_application,
)
from knowledge_assistant.config import (
    get_settings,
)
from knowledge_assistant.mcp.serializers import (
    serialize_generated_answer,
    serialize_index_stats,
    serialize_search_result,
)
from knowledge_assistant.models import (
    StartupTimings,
)


def _create_application():
    """Create the application used by the MCP server."""

    startup_started = perf_counter()

    settings_started = perf_counter()
    settings = get_settings()

    settings_loading_ms = (
        perf_counter() - settings_started
    ) * 1000

    construction_started = perf_counter()

    application = create_application(
        settings=settings,
    )

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

    return application


application = _create_application()

mcp = MCPServer(
    "Knowledge Assistant"
)


@mcp.tool()
def search_documents(
    query: str,
    limit: int = 5,
) -> list[dict[str, Any]]:
    """Search indexed documents for relevant passages."""

    normalized_query = query.strip()

    if not normalized_query:
        raise ValueError(
            "query cannot be empty"
        )

    if limit <= 0:
        raise ValueError(
            "limit must be greater than zero"
        )

    results = application.search(
        query=normalized_query,
        limit=limit,
    )

    return [
        serialize_search_result(result)
        for result in results
    ]


@mcp.tool()
def answer_question(
    query: str,
    limit: int = 5,
) -> dict[str, Any]:
    """Answer a question using indexed documents."""

    normalized_query = query.strip()

    if not normalized_query:
        raise ValueError(
            "query cannot be empty"
        )

    if limit <= 0:
        raise ValueError(
            "limit must be greater than zero"
        )

    answer = application.ask(
        query=normalized_query,
        limit=limit,
    )

    return serialize_generated_answer(
        answer
    )


@mcp.tool()
def get_index_stats() -> dict[str, Any]:
    """Return statistics about the indexed knowledge base."""

    return serialize_index_stats(
        application.stats()
    )


@mcp.tool()
def inspect_index(
    limit: int = 10,
) -> list[dict[str, object]]:
    """Inspect raw records stored in the vector index."""

    if limit <= 0:
        raise ValueError(
            "limit must be greater than zero"
        )

    return application.inspect(
        limit=limit,
    )