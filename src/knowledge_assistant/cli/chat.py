from knowledge_assistant.agent.formatting import (
    AgentConsoleFormatter,
)
from knowledge_assistant.conversation import (
    ConversationHistory,
)


def run_chat(
    agent_runtime,
) -> None:
    history = ConversationHistory(
        max_messages=8,
    )

    print(
        "Knowledge Assistant Chat"
    )
    print(
        "Type 'exit' or 'quit' to stop."
    )

    while True:
        try:
            query = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not query:
            continue

        if query.lower() in {
            "exit",
            "quit",
        }:
            break

        response = agent_runtime.run(
            query=query,
            history=history,
        )

        print()
        print(
            AgentConsoleFormatter.format_response(
                response=response,
                include_trace=False,
            )
        )