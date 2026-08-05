from typing import Annotated

from fastapi import Depends, Request

from knowledge_assistant.application import (
    KnowledgeAssistantApplication,
)


def get_knowledge_application(
    request: Request,
) -> KnowledgeAssistantApplication:
    """Return the application created during API startup."""

    application = getattr(
        request.app.state,
        "knowledge_application",
        None,
    )

    if application is None:
        raise RuntimeError(
            "Knowledge Assistant application is not initialized"
        )

    return application


KnowledgeApplicationDependency = Annotated[
    KnowledgeAssistantApplication,
    Depends(get_knowledge_application),
]