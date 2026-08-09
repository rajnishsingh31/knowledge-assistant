import logging

from knowledge_assistant.mcp.server import (
    mcp,
)


logger = logging.getLogger(__name__)


def main() -> None:
    logger.warning(
        "Starting Knowledge Assistant MCP server over stdio"
    )

    mcp.run()


if __name__ == "__main__":
    main()